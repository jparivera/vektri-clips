"""Read-only local observations for non-executing operational preflight.

The DB/WAL reader uses a bounded private temporary copy so observing an idle
library cannot create its original WAL/SHM. Media verification only reads files:
no decoder, worker, external account, credential or transport is invoked.
"""
from __future__ import annotations

import contextlib
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import stat
import time
import zipfile

from . import config, db
from .operational_preflight import (
    BudgetSnapshot, CloudAnalysisPlan, DestinationDraft, ExportObservation,
    ManualApprovalObservation, Money, PriceSchedule, ProvenanceObservation,
    SourceObservation, StrictPlan, cloud_analysis_preflight, publishing_preflight,
)
from .retention import reference_snapshot
from .schemas import PublicationDraft


JSON_LIMIT = 8 * 1024**2
READ_LIMIT = 16 * 1024**3
TIME_LIMIT = 30
ARCHIVE_METADATA_LIMIT = 256 * 1024
ARTIFACT_NAMES = {'video.mp4', 'clean.mp4', 'captions.srt', 'captions.vtt',
                  'captions.ass', 'thumbnail.jpg'}
HEX = re.compile(r'^[a-f0-9]{64}$')


class DeclaredBudgetScenario(StrictPlan):
    """Hypothetical operator numbers, never the application's spend ledger."""
    per_job_limit_micro_usd: Money
    total_limit_micro_usd: Money
    spent_micro_usd: Money
    reserved_micro_usd: Money


class CloudPreflightScenario(StrictPlan):
    request: CloudAnalysisPlan
    price_schedule: PriceSchedule | None = None
    budget: DeclaredBudgetScenario


class VerificationBlocked(ValueError):
    """Short safe blocker code; never exposes a filesystem path or file content."""


class _ReadBudget:
    def __init__(self):
        self.remaining = READ_LIMIT
        self.deadline = time.monotonic() + TIME_LIMIT
        self.observed = []

    def take(self, count=0):
        self.remaining -= count
        if self.remaining < 0:
            raise VerificationBlocked('verification_byte_limit_exceeded')
        if time.monotonic() > self.deadline:
            raise VerificationBlocked('verification_time_limit_exceeded')


class _BoundedArchiveReader:
    """Limit ZIP metadata before ZipFile allocates/parses its central directory."""
    def __init__(self, handle, size, budget):
        self.handle, self.size, self.budget = handle, size, budget
        self.metadata_remaining = ARCHIVE_METADATA_LIMIT
        self.metadata_phase = True

    def read(self, size=-1):
        requested = self.size - self.tell() if size < 0 else size
        if requested < 0:
            raise VerificationBlocked('invalid_archive_read')
        if self.metadata_phase:
            if requested > self.metadata_remaining:
                raise VerificationBlocked('package_metadata_limit_exceeded')
            self.metadata_remaining -= requested
        elif requested > 1024**2:
            raise VerificationBlocked('package_read_limit_exceeded')
        if requested > self.budget.remaining:
            raise VerificationBlocked('verification_byte_limit_exceeded')
        self.budget.take()
        data = self.handle.read(requested)
        self.budget.take(len(data))
        return data

    def seek(self, offset, whence=0):
        return self.handle.seek(offset, whence)

    def tell(self):
        return self.handle.tell()

    def seekable(self):
        return True


def _meta(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


@contextlib.contextmanager
def _project_file(pid, relative, budget):
    """Descriptor-relative opens refuse symlinks and special files at every level."""
    config.project_dir(pid)  # Validate the application's project identifier.
    if not isinstance(relative, str) or '\\' in relative or '\x00' in relative:
        raise VerificationBlocked('unsafe_project_artifact_path')
    path = PurePosixPath(relative)
    if (path.is_absolute() or str(path) != relative or not path.parts
            or path.parts[0] not in ('source', 'artifacts')
            or any(part.startswith('.') for part in path.parts)):
        raise VerificationBlocked('unsafe_project_artifact_path')
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    descriptor = None
    try:
        for part in (*config.DATA.absolute().parts[1:], 'projects', pid, *path.parts[:-1]):
            if part in ('.', '..'):
                raise VerificationBlocked('unsafe_project_artifact_path')
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = _meta(os.fstat(descriptor))
        if not stat.S_ISREG(before[2]) or before[3] > budget.remaining:
            raise VerificationBlocked('project_artifact_not_regular_or_too_large')
        budget.take()
        with os.fdopen(descriptor, 'rb', closefd=False) as handle:
            yield handle, before
        if _meta(os.fstat(descriptor)) != before:
            raise VerificationBlocked('project_artifact_changed_during_verification')
        budget.observed.append((pid, relative, before))
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _hash(handle, budget, expected_size=None):
    digest, count = hashlib.sha256(), 0
    while chunk := handle.read(1024**2):
        count += len(chunk)
        budget.take(len(chunk))
        if expected_size is not None and count > expected_size:
            raise VerificationBlocked('artifact_size_mismatch')
        digest.update(chunk)
    if expected_size is not None and count != expected_size:
        raise VerificationBlocked('artifact_size_mismatch')
    return digest.hexdigest(), count


def _json(handle, budget):
    data = handle.read(JSON_LIMIT + 1)
    budget.take(len(data))
    if len(data) > JSON_LIMIT:
        raise VerificationBlocked('manifest_size_limit_exceeded')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise VerificationBlocked('manifest_not_an_object')
    return value


def _find(snapshot, table, identifier):
    value = next((row for row in snapshot[table] if row.get('id') == identifier), None)
    if value is None:
        raise KeyError(table[:-1].capitalize() + ' not found')
    return value


def _milliseconds(value, *, ceiling=False):
    try:
        number = Decimal(str(value)) * 1000
        if not number.is_finite() or number < 0:
            raise ValueError
        return int(number.to_integral_value(rounding=ROUND_CEILING if ceiling else ROUND_FLOOR))
    except (ValueError, InvalidOperation):
        raise VerificationBlocked('invalid_recorded_duration') from None


def _observe_source(project, budget):
    source = project['source']
    if not isinstance(source, dict) or not HEX.fullmatch(str(source.get('sha256', ''))):
        raise VerificationBlocked('source_not_prepared')
    with _project_file(project['id'], source.get('path'), budget) as (handle, before):
        actual, size = _hash(handle, budget, before[3])
    if actual != source['sha256']:
        raise VerificationBlocked('source_hash_changed')
    metadata = source.get('metadata', {})
    observed = SourceObservation(project_id=project['id'], source_sha256=actual,
                                 duration_ms=_milliseconds(metadata.get('duration')),
                                 size_bytes=size, has_video=metadata.get('video'),
                                 has_audio=metadata.get('audio'))
    return source, observed


def _unchanged_files(budget):
    # Reopen names to detect replacement after a successful descriptor read.
    observations = list(budget.observed)
    for pid, path, expected in observations:
        with _project_file(pid, path, budget) as (_, current):
            if current != expected:
                raise VerificationBlocked('project_artifact_changed_during_verification')


def cloud_preflight_for_project(pid, scenario):
    payload = CloudPreflightScenario.model_validate(scenario)
    if payload.request.project_id != pid:
        raise ValueError('The scenario project must match the selected project URL.')
    budget = _ReadBudget()
    initial = reference_snapshot()
    project = _find(initial, 'projects', pid)
    try:
        source, observed = _observe_source(project, budget)
        _unchanged_files(budget)
    except (OSError, VerificationBlocked, ValueError):
        raise db.Conflict('Current source integrity could not be verified; no data was transmitted.') from None
    latest = _find(reference_snapshot(), 'projects', pid)
    budget.take()
    if latest['source'] != source:
        raise db.Conflict('Source metadata changed during preflight; request a fresh scenario.')
    now = int(time.time())
    result = cloud_analysis_preflight(payload.request, observed, payload.price_schedule,
        BudgetSnapshot(**payload.budget.model_dump(), observed_at_unix=now), now_unix=now)
    result['scenario_notice'] = ('All supplied prices, balances and usage ceilings are hypothetical operator '
                                'inputs, not verified vendor prices, an account balance or an actual spend ledger.')
    result['source_integrity_checked'] = True
    return result


def _observe_export(project, exported, source, current_source, timeline, budget):
    pid, folder = project['id'], exported['artifact_dir']
    with _project_file(pid, folder + '/manifest.json', budget) as (handle, _):
        manifest = _json(handle, budget)
    if manifest != exported['manifest']:
        raise VerificationBlocked('disk_and_database_manifest_differ')
    if (manifest.get('project_id') != pid or manifest.get('revision') != exported['revision']
            or manifest.get('fingerprint') != exported['fingerprint']
            or manifest.get('source_sha256') != current_source.source_sha256
            or manifest.get('source_provenance') != source.get('provenance')
            or manifest.get('timeline') != timeline):
        raise VerificationBlocked('export_identity_or_provenance_changed')
    files = manifest.get('files')
    if (not isinstance(files, dict) or 'video.mp4' not in files
            or not set(files) <= ARTIFACT_NAMES or manifest.get('video') != 'video.mp4'):
        raise VerificationBlocked('unexpected_export_artifacts')
    hashes = {}
    for name, expected in files.items():
        if (not isinstance(expected, dict) or type(expected.get('bytes')) is not int
                or expected['bytes'] < 0 or not HEX.fullmatch(str(expected.get('sha256', '')))):
            raise VerificationBlocked('invalid_manifest_artifact')
        with _project_file(pid, folder + '/' + name, budget) as (handle, _):
            actual, _ = _hash(handle, budget, expected['bytes'])
        if actual != expected['sha256']:
            raise VerificationBlocked('export_artifact_hash_changed')
        hashes[name] = actual
    with _project_file(pid, folder + '/package.zip', budget) as (handle, package_meta):
        archive_reader = _BoundedArchiveReader(handle, package_meta[3], budget)
        with zipfile.ZipFile(archive_reader) as package:
            archive_reader.metadata_phase = False
            entries = package.infolist()
            names = [item.filename for item in entries]
            if (len(names) != len(set(names)) or set(names) != set(files) | {'manifest.json'}
                    or any(item.flag_bits & 1 or stat.S_ISLNK(item.external_attr >> 16)
                           or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                           for item in entries)):
                raise VerificationBlocked('unexpected_package_members')
            for item in entries:
                if item.file_size > budget.remaining:
                    raise VerificationBlocked('package_expansion_limit_exceeded')
                with package.open(item) as member:
                    if item.filename == 'manifest.json':
                        if _json(member, budget) != manifest:
                            raise VerificationBlocked('package_manifest_changed')
                    else:
                        digest, _ = _hash(member, budget, files[item.filename]['bytes'])
                        if digest != hashes[item.filename]:
                            raise VerificationBlocked('package_artifact_hash_changed')
            # Reading each member to EOF above checks its ZIP CRC as well.
    verification = manifest.get('verification', {})
    return ExportObservation(
        project_id=pid, export_id=exported['id'], revision=exported['revision'],
        current_project_revision=project['revision'], asset_fingerprint=exported['fingerprint'],
        source_sha256=manifest['source_sha256'], current_source_sha256=current_source.source_sha256,
        expected_video_sha256=files['video.mp4']['sha256'], observed_video_sha256=hashes['video.mp4'],
        full_decode_passed=verification.get('full_decode') == 'passed',
        dimensions_passed=verification.get('dimensions') == 'passed',
        package_crc_passed=True, artifact_hashes_passed=True,
        duration_error_ms=_milliseconds(verification.get('duration_error_seconds'), ceiling=True),
        duration_tolerance_ms=_milliseconds(verification.get('duration_tolerance_seconds')),
        destination_metadata_validated=True, checked_at_unix=int(time.time()))


def publication_preflight(publication_id):
    budget = _ReadBudget()
    initial = reference_snapshot()
    publication = _find(initial, 'publications', publication_id)
    project = _find(initial, 'projects', publication['project_id'])
    draft = DestinationDraft.model_validate(PublicationDraft.model_validate(publication['draft']).model_dump())
    source = project['source']
    provenance = source.get('provenance', {})
    rights = ProvenanceObservation(source_sha256=source['sha256'],
        permission_status=provenance.get('permission_status', 'unknown'),
        attribution=provenance.get('attribution', ''))
    stored_approval = publication['approval'] if publication['status'] == 'ready-for-manual-upload' else None
    reviewed = None
    if stored_approval:
        reviewed = ManualApprovalObservation(
            **{key: stored_approval[key] for key in ('draft', 'asset_fingerprint',
                                                   'confirm_manual_handoff', 'rights_acknowledgment')},
            confirmed_at_unix=int(stored_approval['confirmed_at']))
    verification_blocker = None
    observed = None
    exported = next((row for row in initial['exports'] if row['id'] == publication['export_id']), None)
    try:
        if not exported or exported['project_id'] != project['id']:
            raise VerificationBlocked('selected_export_missing')
        if (draft.project_id != publication['project_id'] or draft.export_id != publication['export_id']
                or exported['revision'] != publication['revision']
                or exported['fingerprint'] != publication['asset_fingerprint']):
            raise VerificationBlocked('publication_export_binding_changed')
        source, current_source = _observe_source(project, budget)
        saved_revision = next((row for row in initial['revisions']
                               if row['project_id'] == project['id'] and row['revision'] == exported['revision']), None)
        if saved_revision is None:
            raise VerificationBlocked('export_timeline_revision_missing')
        observed = _observe_export(project, exported, source, current_source, saved_revision['timeline'], budget)
        _unchanged_files(budget)
    except VerificationBlocked as exc:
        verification_blocker = str(exc)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError):
        verification_blocker = 'export_files_or_metadata_unverifiable'
    latest = reference_snapshot()
    if (_find(latest, 'publications', publication_id) != publication
            or _find(latest, 'projects', project['id']) != project
            or (exported and _find(latest, 'exports', exported['id']) != exported)):
        raise db.Conflict('Publication or project changed during preflight; review the current export again.')
    result = publishing_preflight(draft, observed, rights, reviewed, now_unix=int(time.time()))
    if verification_blocker:
        result['manual_handoff_ready'] = False
        result['manual_blockers'].append(verification_blocker)
    result['current_file_integrity_verified'] = verification_blocker is None and observed is not None
    result['verification_basis'] = (
        'Fresh source/artifact hashes and ZIP CRC bound to the stored manifest; '
        'recorded full-decode evidence is evaluated only for unchanged bytes.'
        if result['current_file_integrity_verified'] else
        'Current source/export verification was incomplete or failed; inspect the manual blockers. '
        'No successful current-file verification is claimed.')
    result['filesystem_writes_to_project'] = False
    return result


def read_cloud_scenario(path):
    """CLI reads only its explicitly supplied bounded JSON scenario, then validates."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = _meta(os.fstat(descriptor))
        if not stat.S_ISREG(before[2]):
            raise ValueError('Cloud preflight scenario must be a regular file, not a symlink or special file.')
        if before[3] > 64 * 1024:
            raise ValueError('Cloud preflight scenario exceeds 64 KiB.')
        with os.fdopen(descriptor, 'rb', closefd=False) as source:
            data = source.read(64 * 1024 + 1)
        if _meta(os.fstat(descriptor)) != before or len(data) != before[3]:
            raise ValueError('Cloud preflight scenario changed while being read.')
    finally:
        os.close(descriptor)
    return CloudPreflightScenario.model_validate(json.loads(data)).model_dump()
