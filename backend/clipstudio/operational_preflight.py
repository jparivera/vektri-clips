"""Pure planning checks; deliberately no provider, credential or upload transport.

All observations (source, ledger, export, provenance and recorded approval) must
be built by the caller from current local state, never trusted from an HTTP body.
This module does no I/O, reads no clock/environment, and reserves no money. Its
fingerprints bind reviewable values; they are not signatures or authorization.
Local caps below are planning limits, not a claim about any provider's quotas.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_SAFE_INTEGER = 2**53 - 1
Nonnegative = Annotated[int, Field(strict=True, ge=0, le=MAX_SAFE_INTEGER)]
Money = Nonnegative  # integer micro-USD: 1 USD == 1,000,000 micro-USD
Timestamp = Annotated[int, Field(strict=True, ge=0, le=253402300799)]
Identifier = Annotated[str, Field(min_length=1, max_length=120, pattern=r'^[A-Za-z0-9_.:/-]+$')]
Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
Permission = Literal['unknown', 'permission-recorded', 'owned', 'licensed', 'synthetic']
TransferField = Literal['query_text', 'transcript', 'sampled_frames', 'audio_clip', 'video_clip']


class StrictPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, allow_inf_nan=False)


class SourceObservation(StrictPlan):
    """Server-observed selected source identity and probe result; no path/URL."""
    project_id: Identifier
    source_sha256: Digest
    duration_ms: int = Field(strict=True, gt=0, le=86_400_000)
    size_bytes: int = Field(strict=True, gt=0, le=10**12)
    has_video: bool
    has_audio: bool


class CloudAnalysisPlan(StrictPlan):
    project_id: Identifier
    source_sha256: Digest
    provider: Identifier
    model: Identifier
    clip_start_ms: int = Field(strict=True, ge=0, le=86_400_000)
    clip_end_ms: int = Field(strict=True, gt=0, le=86_400_000)
    transmitted_fields: list[TransferField] = Field(min_length=1, max_length=5)
    # Caller-supplied ceilings for a future provider-specific request builder.
    # They do not pretend to infer a model tokenizer or image/audio token cost.
    input_tokens_max: int = Field(strict=True, ge=1, le=2_000_000)
    output_tokens_max: int = Field(strict=True, ge=1, le=32_768)
    frame_count_max: int = Field(strict=True, ge=0, le=900)
    frame_interval_ms: int | None = Field(default=None, strict=True, ge=100, le=86_400_000)
    transfer_bytes_max: int = Field(strict=True, ge=1, le=2_147_483_648)

    @model_validator(mode='after')
    def bounds(self):
        if not 0 < self.clip_end_ms - self.clip_start_ms <= 7_200_000:
            raise ValueError('Select an ordered source clip of at most two hours.')
        if len(set(self.transmitted_fields)) != len(self.transmitted_fields):
            raise ValueError('Transfer fields must be distinct.')
        if 'query_text' not in self.transmitted_fields:
            raise ValueError('Explicitly include query_text in the transfer scope.')
        if 'sampled_frames' in self.transmitted_fields:
            if not self.frame_count_max or self.frame_interval_ms is None:
                raise ValueError('Frame transfer needs a positive frame cap and sampling interval.')
        elif self.frame_count_max or self.frame_interval_ms is not None:
            raise ValueError('Frame estimates require sampled_frames in the transfer scope.')
        return self


class UnitPrice(StrictPlan):
    micro_usd: Money
    units: int = Field(strict=True, ge=1, le=1_000_000_000)


class PriceSchedule(StrictPlan):
    """Locally supplied quote evidence. Zero rates must be explicit, never inferred."""
    provider: Identifier
    model: Identifier
    currency: Literal['USD'] = 'USD'
    provenance_url: str = Field(min_length=10, max_length=1000)
    provenance_sha256: Digest
    observed_at_unix: Timestamp
    expires_at_unix: Timestamp
    request_micro_usd: Money | None = None
    input_tokens: UnitPrice | None = None
    output_tokens: UnitPrice | None = None
    frames: UnitPrice | None = None
    media_milliseconds: UnitPrice | None = None
    transfer_bytes: UnitPrice | None = None

    @field_validator('provenance_url')
    @classmethod
    def public_reference_only(cls, value):
        url = urlsplit(value)
        if (url.scheme != 'https' or not url.hostname or url.username or url.password
                or url.query or url.fragment or any(c.isspace() for c in value)):
            raise ValueError('Use an HTTPS price-reference URL without credentials, query or fragment.')
        return value

    @model_validator(mode='after')
    def ordered(self):
        if self.expires_at_unix <= self.observed_at_unix:
            raise ValueError('The supplied price evidence needs an explicit later expiry.')
        return self


class BudgetSnapshot(StrictPlan):
    per_job_limit_micro_usd: Money
    total_limit_micro_usd: Money
    spent_micro_usd: Money
    reserved_micro_usd: Money
    observed_at_unix: Timestamp


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _ceil_charge(quantity, price):
    charge = (quantity * price.micro_usd + price.units - 1) // price.units
    if charge > MAX_SAFE_INTEGER:
        raise ValueError('Quoted cost exceeds the supported JSON-safe integer micro-USD range.')
    return charge


def _clock(value):
    if type(value) is not int or not 0 <= value <= 253402300799:
        raise ValueError('now_unix must be an integer Unix timestamp.')
    return value


def cloud_analysis_preflight(request, source, price_schedule, budget, *, now_unix,
                             maximum_price_age_seconds=86_400):
    """Validate and price a bounded proposal; always return can_execute=False.

    Usage values are explicit proposed upper bounds, not measured tokens/bytes.
    A future adapter must build the exact scoped payload, measure it and enforce
    all ceilings before a transfer. It must also reconcile actual billable usage.
    Price evidence is supplied locally and is not independently fetched/verified.
    """
    now = _clock(now_unix)
    if type(maximum_price_age_seconds) is not int or not 1 <= maximum_price_age_seconds <= 2_592_000:
        raise ValueError('Local price-evidence maximum age must be 1–2,592,000 seconds.')
    plan = CloudAnalysisPlan.model_validate(request)
    observed = SourceObservation.model_validate(source)
    ledger = BudgetSnapshot.model_validate(budget)
    prices = PriceSchedule.model_validate(price_schedule) if price_schedule is not None else None
    blockers = []
    if (plan.project_id, plan.source_sha256) != (observed.project_id, observed.source_sha256):
        blockers.append('selected_source_mismatch')
    if plan.clip_end_ms > observed.duration_ms:
        blockers.append('clip_exceeds_source')
    scope = set(plan.transmitted_fields)
    if scope & {'sampled_frames', 'video_clip'} and not observed.has_video:
        blockers.append('selected_source_has_no_video')
    if 'audio_clip' in scope and not observed.has_audio:
        blockers.append('selected_source_has_no_audio')
    duration = plan.clip_end_ms - plan.clip_start_ms
    frames = 0
    if 'sampled_frames' in scope:
        # Conservative endpoint-inclusive upper count for uniform sampling.
        frames = (duration + plan.frame_interval_ms - 1) // plan.frame_interval_ms + 1
        if frames > plan.frame_count_max:
            blockers.append('sampling_exceeds_frame_cap')
    quantities = {'input_tokens': plan.input_tokens_max, 'output_tokens': plan.output_tokens_max,
                  'frames': plan.frame_count_max,
                  'media_milliseconds': duration * len(scope & {'audio_clip', 'video_clip'}),
                  'transfer_bytes': plan.transfer_bytes_max}
    if ledger.observed_at_unix > now or now - ledger.observed_at_unix > 60:
        blockers.append('budget_snapshot_stale')
    available = ledger.total_limit_micro_usd - ledger.spent_micro_usd - ledger.reserved_micro_usd
    if available < 0:
        blockers.append('budget_already_overcommitted')
    breakdown = {}
    estimate = None
    if prices is None:
        blockers.append('price_schedule_missing')
    else:
        if (prices.provider, prices.model) != (plan.provider, plan.model):
            blockers.append('price_model_mismatch')
        if (prices.observed_at_unix > now or now >= prices.expires_at_unix
                or now - prices.observed_at_unix > maximum_price_age_seconds):
            blockers.append('price_schedule_stale')
        missing = []
        if prices.request_micro_usd is None:
            missing.append('request')
        else:
            breakdown['request'] = prices.request_micro_usd
        for dimension, quantity in quantities.items():
            rate = getattr(prices, dimension)
            if quantity and rate is None:
                missing.append(dimension)
            else:
                breakdown[dimension] = _ceil_charge(quantity, rate) if quantity else 0
        if missing:
            blockers.extend('price_missing_' + dimension for dimension in missing)
        else:
            estimate = sum(breakdown.values())
            if estimate > MAX_SAFE_INTEGER:
                raise ValueError('Total quoted cost exceeds the supported JSON-safe integer micro-USD range.')
            if estimate > ledger.per_job_limit_micro_usd:
                blockers.append('per_job_budget_exceeded')
            if estimate > available:
                blockers.append('total_budget_exceeded')
    planning_valid = not blockers
    # These cannot be overridden by user-supplied booleans or a fabricated key.
    blockers.extend(['provider_disabled', 'account_approval_missing', 'data_transfer_not_approved',
                     'usage_bounds_not_enforced', 'budget_not_reserved'])
    binding = {'request': plan.model_dump(), 'source': observed.model_dump(),
               'price_schedule': prices.model_dump() if prices else None, 'budget': ledger.model_dump(),
               'maximum_price_age_seconds': maximum_price_age_seconds}
    return {'schema_version': 1, 'kind': 'cloud_analysis_preflight', 'simulation_only': True,
            'can_execute': False,
            'planning_valid': planning_valid, 'blockers': blockers,
            'plan_fingerprint': _fingerprint(binding), 'evaluated_at_unix': now,
            'provider': plan.provider, 'model': plan.model,
            'selected_source': {'project_id': plan.project_id, 'sha256': plan.source_sha256,
                                'start_ms': plan.clip_start_ms, 'end_ms': plan.clip_end_ms},
            'transmitted_fields': list(plan.transmitted_fields),
            'usage_upper_bounds': {**quantities, 'uniform_sample_count_upper_bound': frames},
            'estimated_max_micro_usd': estimate, 'cost_breakdown_micro_usd': breakdown,
            'remaining_total_micro_usd': max(0, available), 'currency': 'USD',
            'price_evidence': prices.model_dump() if prices else None,
            'data_transmitted': False, 'spend_reserved_micro_usd': 0, 'paid_usage_micro_usd': 0,
            'limitations': ['Supplied prices and usage ceilings require provider-specific verification.',
                            'Budget values are a declared simulation, not an authoritative spend ledger.',
                            'video_clip scopes only video; audio requires separate audio_clip scope.',
                            'Media time prices the aggregate duration of separately selected audio/video payloads.',
                            'Each priced dimension is rounded up independently to whole micro-USD.',
                            'This snapshot is not an approval, atomic reservation or execution guarantee.']}


class DestinationDraft(StrictPlan):
    """Same public fields as PublicationDraft; caller validates IANA zone upstream.

    No timezone database, credential store or external platform is queried here.
    Schedule metadata does not implement an automatic posting scheduler.
    """
    project_id: Identifier
    export_id: Identifier
    destination: Literal['manual', 'youtube', 'tiktok', 'instagram'] = 'manual'
    account: str = Field(default='', max_length=200)
    caption: str = Field(default='', max_length=5000)
    visibility: Literal['private', 'unlisted', 'public'] = 'private'
    schedule: str | None = Field(default=None, max_length=100)
    timezone: str = Field(min_length=1, max_length=100)
    permission_status: Permission = 'unknown'

    @field_validator('schedule')
    @classmethod
    def date_text(cls, value):
        if value is not None:
            datetime.fromisoformat(value)
        return value


class ManualApprovalObservation(StrictPlan):
    draft: DestinationDraft
    asset_fingerprint: Digest
    confirm_manual_handoff: bool
    rights_acknowledgment: str = Field(min_length=1, max_length=1000)
    confirmed_at_unix: Timestamp

    @field_validator('rights_acknowledgment')
    @classmethod
    def acknowledgment(cls, value):
        if not value.strip():
            raise ValueError('A nonblank operator rights acknowledgment is required.')
        return value


class ExportObservation(StrictPlan):
    """Current server-observed file/DB checks, never client certification."""
    project_id: Identifier
    export_id: Identifier
    revision: int = Field(strict=True, ge=1)
    current_project_revision: int = Field(strict=True, ge=1)
    asset_fingerprint: Digest
    source_sha256: Digest
    current_source_sha256: Digest
    expected_video_sha256: Digest
    observed_video_sha256: Digest
    full_decode_passed: bool
    dimensions_passed: bool
    package_crc_passed: bool
    artifact_hashes_passed: bool
    duration_error_ms: int = Field(strict=True, ge=0)
    duration_tolerance_ms: int = Field(strict=True, ge=0, le=150)
    destination_metadata_validated: bool
    checked_at_unix: Timestamp


class ProvenanceObservation(StrictPlan):
    source_sha256: Digest
    permission_status: Permission
    attribution: str = Field(default='', max_length=2000)


def publishing_preflight(draft, export, provenance, approval=None, *, now_unix):
    """Report manual readiness from trusted observations; remote posting stays off.

    The caller must recheck actual export/package files, current DB revision and
    draft timezone before building ExportObservation. Passing an old manifest's
    success flag without current hashes is insufficient. No filesystem path,
    token, credentials, account scopes or transport can be supplied to execute.
    """
    now = _clock(now_unix)
    selected = DestinationDraft.model_validate(draft)
    observed = ExportObservation.model_validate(export) if export is not None else None
    rights = ProvenanceObservation.model_validate(provenance) if provenance is not None else None
    reviewed = ManualApprovalObservation.model_validate(approval) if approval is not None else None
    blockers, warnings = [], []
    if observed is None:
        blockers.append('verified_export_missing')
    else:
        if (selected.project_id, selected.export_id) != (observed.project_id, observed.export_id):
            blockers.append('selected_export_mismatch')
        if observed.revision != observed.current_project_revision:
            blockers.append('export_revision_stale')
        if observed.source_sha256 != observed.current_source_sha256:
            blockers.append('export_source_changed')
        if observed.expected_video_sha256 != observed.observed_video_sha256:
            blockers.append('export_video_hash_changed')
        for name in ('full_decode_passed', 'dimensions_passed', 'package_crc_passed', 'artifact_hashes_passed'):
            if not getattr(observed, name):
                blockers.append('export_' + name.removesuffix('_passed') + '_unverified')
        if observed.duration_error_ms > observed.duration_tolerance_ms:
            blockers.append('export_duration_out_of_tolerance')
        if observed.checked_at_unix > now or now - observed.checked_at_unix > 60:
            blockers.append('export_observation_stale')
        if not observed.destination_metadata_validated:
            blockers.append('destination_metadata_unvalidated')
    if selected.destination != 'manual' and not selected.account.strip():
        blockers.append('destination_account_missing')
    if rights is None:
        blockers.append('source_provenance_missing')
    else:
        if observed and rights.source_sha256 != observed.current_source_sha256:
            blockers.append('source_provenance_mismatch')
        if selected.permission_status != rights.permission_status:
            blockers.append('draft_permission_status_stale')
        if rights.permission_status == 'unknown':
            warnings.append('Permission is unknown; an operator acknowledgment does not establish legal rights.')
        if not rights.attribution.strip():
            warnings.append('No attribution is recorded; review source terms before manual upload.')
    if reviewed is None:
        blockers.append('manual_approval_missing')
    else:
        if not reviewed.confirm_manual_handoff:
            blockers.append('manual_handoff_not_confirmed')
        if reviewed.draft.model_dump() != selected.model_dump():
            blockers.append('approved_draft_changed')
        if observed and reviewed.asset_fingerprint != observed.asset_fingerprint:
            blockers.append('approved_asset_changed')
        if reviewed.confirmed_at_unix > now:
            blockers.append('approval_timestamp_invalid')
    remote_blockers = ['publishing_transport_unsupported', 'account_authorization_unverified',
                       'account_scopes_unverified', 'remote_publication_not_approved']
    binding = {'draft': selected.model_dump(), 'export': observed.model_dump() if observed else None,
               'provenance': rights.model_dump() if rights else None,
               'approval': reviewed.model_dump() if reviewed else None}
    return {'schema_version': 1, 'kind': 'publishing_preflight', 'simulation_only': True,
            'can_publish': False,
            'manual_handoff_ready': not blockers, 'manual_blockers': blockers,
            'remote_blockers': remote_blockers, 'warnings': warnings,
            'plan_fingerprint': _fingerprint(binding), 'evaluated_at_unix': now,
            'destination': selected.destination, 'visibility': selected.visibility,
            'scheduled_metadata_only': selected.schedule is not None,
            'account_connected': False, 'data_transmitted': False, 'publishing_attempted': False,
            'limitations': ['Manual approval does not authorize an account, remote upload or automatic schedule.',
                            'Provider-specific metadata, account scopes and external quotas are unverified.',
                            'Readiness is a local snapshot and must be rechecked after any edit or file change.']}
