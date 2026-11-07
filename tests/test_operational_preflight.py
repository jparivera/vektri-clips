"""Pure, deterministic tests: no live provider, account, media job or credentials."""
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clipstudio.operational_preflight import (
    CloudAnalysisPlan, DestinationDraft, PriceSchedule, cloud_analysis_preflight,
    publishing_preflight,
)


NOW = 1_800_000_000
SHA = 'a' * 64
VIDEO_SHA = 'b' * 64
ASSET = 'c' * 64


def cloud_values():
    plan = {'project_id': 'project1', 'source_sha256': SHA, 'provider': 'example-provider',
            'model': 'example-model', 'clip_start_ms': 1_000, 'clip_end_ms': 9_000,
            'transmitted_fields': ['query_text', 'sampled_frames'],
            'input_tokens_max': 1_001, 'output_tokens_max': 101,
            'frame_count_max': 5, 'frame_interval_ms': 2_000, 'transfer_bytes_max': 100_001}
    source = {'project_id': 'project1', 'source_sha256': SHA, 'duration_ms': 20_000,
              'size_bytes': 300_000, 'has_video': True, 'has_audio': True}
    # Deliberately fictitious unit prices used only for exact arithmetic tests.
    prices = {'provider': 'example-provider', 'model': 'example-model',
              'provenance_url': 'https://example.invalid/model-prices',
              'provenance_sha256': 'd' * 64, 'observed_at_unix': NOW - 100,
              'expires_at_unix': NOW + 100, 'request_micro_usd': 5,
              'input_tokens': {'micro_usd': 7, 'units': 1_000},
              'output_tokens': {'micro_usd': 9, 'units': 100},
              'frames': {'micro_usd': 2, 'units': 1},
              'transfer_bytes': {'micro_usd': 1, 'units': 100_000}}
    budget = {'per_job_limit_micro_usd': 35, 'total_limit_micro_usd': 100,
              'spent_micro_usd': 50, 'reserved_micro_usd': 15, 'observed_at_unix': NOW}
    return plan, source, prices, budget


def cloud(*, mutate=None, **kwargs):
    values = cloud_values()
    if mutate:
        mutate(*values)
    return cloud_analysis_preflight(*values, now_unix=NOW, **kwargs)


def publication_values():
    draft = {'project_id': 'project1', 'export_id': 'export1', 'destination': 'manual',
             'account': '', 'caption': 'Reviewed caption', 'visibility': 'private',
             'schedule': None, 'timezone': 'Australia/Perth', 'permission_status': 'unknown'}
    export = {'project_id': 'project1', 'export_id': 'export1', 'revision': 2,
              'current_project_revision': 2, 'asset_fingerprint': ASSET,
              'source_sha256': SHA, 'current_source_sha256': SHA,
              'expected_video_sha256': VIDEO_SHA, 'observed_video_sha256': VIDEO_SHA,
              'full_decode_passed': True, 'dimensions_passed': True,
              'package_crc_passed': True, 'artifact_hashes_passed': True,
              'duration_error_ms': 2, 'duration_tolerance_ms': 150,
              'destination_metadata_validated': True, 'checked_at_unix': NOW}
    rights = {'source_sha256': SHA, 'permission_status': 'unknown', 'attribution': 'Source credit'}
    approval = {'draft': copy.deepcopy(draft), 'asset_fingerprint': ASSET,
                'confirm_manual_handoff': True,
                'rights_acknowledgment': 'I reviewed the recorded permission status, which remains unknown.',
                'confirmed_at_unix': NOW - 1}
    return draft, export, rights, approval


def publication(*, mutate=None):
    values = publication_values()
    if mutate:
        mutate(*values)
    return publishing_preflight(*values, now_unix=NOW)


def test_exact_rounding_and_equal_budget_limits_do_not_enable_execution():
    result = cloud()
    assert result['estimated_max_micro_usd'] == 35
    assert result['cost_breakdown_micro_usd'] == {
        'request': 5, 'input_tokens': 8, 'output_tokens': 10, 'frames': 10,
        'media_milliseconds': 0, 'transfer_bytes': 2}
    assert result['remaining_total_micro_usd'] == 35
    assert result['usage_upper_bounds']['uniform_sample_count_upper_bound'] == 5
    assert result['planning_valid'] is True
    assert result['simulation_only'] is True and result['can_execute'] is False
    assert result['blockers'] == ['provider_disabled', 'account_approval_missing',
                                  'data_transfer_not_approved', 'usage_bounds_not_enforced',
                                  'budget_not_reserved']
    assert result['spend_reserved_micro_usd'] == result['paid_usage_micro_usd'] == 0
    assert result['data_transmitted'] is False


@pytest.mark.parametrize(('key', 'value', 'blocker'), [
    ('per_job_limit_micro_usd', 34, 'per_job_budget_exceeded'),
    ('reserved_micro_usd', 16, 'total_budget_exceeded'),
    ('spent_micro_usd', 101, 'budget_already_overcommitted'),
    ('observed_at_unix', NOW + 1, 'budget_snapshot_stale'),
    ('observed_at_unix', NOW - 61, 'budget_snapshot_stale'),
])
def test_budget_guards(key, value, blocker):
    result = cloud(mutate=lambda p, s, r, b: b.update({key: value}))
    assert blocker in result['blockers'] and result['planning_valid'] is False
    assert result['remaining_total_micro_usd'] >= 0


@pytest.mark.parametrize(('field', 'value'), [('observed_at_unix', NOW + 1),
                                           ('observed_at_unix', NOW - 86_401),
                                           ('expires_at_unix', NOW)])
def test_stale_or_future_price_evidence_blocks(field, value):
    result = cloud(mutate=lambda p, s, r, b: r.update({field: value}))
    assert 'price_schedule_stale' in result['blockers']
    assert result['planning_valid'] is False


def test_missing_prices_never_imply_free_usage():
    p, s, _, b = cloud_values()
    result = cloud_analysis_preflight(p, s, None, b, now_unix=NOW)
    assert result['estimated_max_micro_usd'] is None
    assert 'price_schedule_missing' in result['blockers']


@pytest.mark.parametrize('dimension', ['request_micro_usd', 'input_tokens', 'output_tokens', 'frames', 'transfer_bytes'])
def test_every_consumed_dimension_requires_explicit_pricing(dimension):
    result = cloud(mutate=lambda p, s, r, b: r.pop(dimension))
    assert result['estimated_max_micro_usd'] is None
    assert any(x.startswith('price_missing_') for x in result['blockers'])


def test_explicit_zero_rate_and_integer_ceiling_for_media_duration():
    p, s, r, b = cloud_values()
    p.update(transmitted_fields=['query_text', 'audio_clip'], frame_count_max=0,
             frame_interval_ms=None, clip_end_ms=9_001)
    r['media_milliseconds'] = {'micro_usd': 3, 'units': 1_000}
    r['input_tokens'] = {'micro_usd': 0, 'units': 1_000}
    b.update(per_job_limit_micro_usd=100, total_limit_micro_usd=200)
    result = cloud_analysis_preflight(p, s, r, b, now_unix=NOW)
    assert result['cost_breakdown_micro_usd']['media_milliseconds'] == 25
    assert result['cost_breakdown_micro_usd']['input_tokens'] == 0
    assert result['planning_valid'] is True


@pytest.mark.parametrize(('target', 'field', 'value', 'blocker'), [
    ('source', 'project_id', 'other', 'selected_source_mismatch'),
    ('source', 'source_sha256', 'f' * 64, 'selected_source_mismatch'),
    ('source', 'duration_ms', 8_999, 'clip_exceeds_source'),
    ('source', 'has_video', False, 'selected_source_has_no_video'),
    ('prices', 'model', 'other', 'price_model_mismatch'),
    ('plan', 'frame_count_max', 4, 'sampling_exceeds_frame_cap'),
])
def test_source_identity_and_sampling_and_model_guards(target, field, value, blocker):
    values = cloud_values()
    values[{'plan': 0, 'source': 1, 'prices': 2}[target]][field] = value
    result = cloud_analysis_preflight(*values, now_unix=NOW)
    assert blocker in result['blockers'] and result['planning_valid'] is False


@pytest.mark.parametrize(('field', 'value'), [
    ('clip_start_ms', True), ('clip_end_ms', 2.5), ('frame_count_max', 901),
    ('input_tokens_max', 2_000_001), ('output_tokens_max', 32_769),
    ('transfer_bytes_max', 2_147_483_649), ('clip_end_ms', 1_000),
    ('clip_end_ms', 7_201_001), ('transmitted_fields', ['query_text', 'query_text']),
    ('transmitted_fields', ['full_source']), ('frame_interval_ms', None),
    ('api_key', 'not-an-accepted-field'), ('account_approved', True),
])
def test_invalid_or_secret_shaped_plan_fields_are_rejected(field, value):
    p, *_ = cloud_values()
    p[field] = value
    with pytest.raises(ValidationError):
        CloudAnalysisPlan.model_validate(p)


@pytest.mark.parametrize('url', ['http://example.invalid/prices', 'https://user:secret@example.invalid/prices',
                                 'https://example.invalid/prices?key=secret', 'https://example.invalid/prices#secret'])
def test_price_reference_cannot_embed_credentials(url):
    _, _, prices, _ = cloud_values()
    prices['provenance_url'] = url
    with pytest.raises(ValidationError):
        PriceSchedule.model_validate(prices)


@pytest.mark.parametrize('value', [True, 1.1, -1, float('nan'), float('inf'), '43'])
def test_money_is_strict_nonnegative_integer(value):
    with pytest.raises(ValidationError):
        cloud(mutate=lambda p, s, r, b: b.update(per_job_limit_micro_usd=value))


def test_large_integer_price_arithmetic_never_uses_float_or_overflows():
    p, s, r, b = cloud_values()
    p['input_tokens_max'] = 2_000_000
    r['input_tokens'] = {'micro_usd': 10**10, 'units': 3}
    result = cloud_analysis_preflight(p, s, r, b, now_unix=NOW)
    expected = (2_000_000 * 10**10 + 2) // 3
    assert result['cost_breakdown_micro_usd']['input_tokens'] == expected
    assert type(result['estimated_max_micro_usd']) is int
    assert 'per_job_budget_exceeded' in result['blockers']


def test_audio_and_video_payloads_are_priced_as_separate_durations():
    p, s, r, b = cloud_values()
    p.update(transmitted_fields=['query_text', 'video_clip', 'audio_clip'],
             frame_count_max=0, frame_interval_ms=None)
    r['media_milliseconds'] = {'micro_usd': 1, 'units': 1_000}
    result = cloud_analysis_preflight(p, s, r, b, now_unix=NOW)
    assert result['usage_upper_bounds']['media_milliseconds'] == 16_000
    assert result['cost_breakdown_micro_usd']['media_milliseconds'] == 16


def test_money_input_and_computed_output_cannot_lose_precision_in_json_clients():
    p, s, r, b = cloud_values()
    b['total_limit_micro_usd'] = 2**53
    with pytest.raises(ValidationError):
        cloud_analysis_preflight(p, s, r, b, now_unix=NOW)
    b['total_limit_micro_usd'] = 2**53 - 1
    r['input_tokens'] = {'micro_usd': 2**53 - 1, 'units': 1}
    with pytest.raises(ValueError, match='JSON-safe integer'):
        cloud_analysis_preflight(p, s, r, b, now_unix=NOW)


def test_fingerprints_bind_scope_price_budget_and_do_not_mutate_arguments():
    original = cloud_values(); saved = copy.deepcopy(original)
    first = cloud_analysis_preflight(*original, now_unix=NOW)
    assert original == saved
    assert first == cloud_analysis_preflight(*original, now_unix=NOW)
    original[0]['clip_start_ms'] += 1
    assert first['plan_fingerprint'] != cloud_analysis_preflight(*original, now_unix=NOW)['plan_fingerprint']
    original = cloud_values(); original[2]['input_tokens']['micro_usd'] += 1
    assert first['plan_fingerprint'] != cloud_analysis_preflight(*original, now_unix=NOW)['plan_fingerprint']
    original = cloud_values(); original[3]['reserved_micro_usd'] += 1
    assert first['plan_fingerprint'] != cloud_analysis_preflight(*original, now_unix=NOW)['plan_fingerprint']


def test_unknown_rights_with_recorded_manual_review_preserves_manual_handoff():
    result = publication()
    assert result['manual_handoff_ready'] is True and result['manual_blockers'] == []
    assert result['can_publish'] is False and result['publishing_attempted'] is False
    assert result['account_connected'] is False and result['data_transmitted'] is False
    assert 'account_scopes_unverified' in result['remote_blockers']
    assert any('does not establish legal rights' in x for x in result['warnings'])


@pytest.mark.parametrize(('field', 'value', 'blocker'), [
    ('export_id', 'other', 'selected_export_mismatch'),
    ('current_project_revision', 3, 'export_revision_stale'),
    ('current_source_sha256', 'f' * 64, 'export_source_changed'),
    ('observed_video_sha256', 'f' * 64, 'export_video_hash_changed'),
    ('full_decode_passed', False, 'export_full_decode_unverified'),
    ('dimensions_passed', False, 'export_dimensions_unverified'),
    ('package_crc_passed', False, 'export_package_crc_unverified'),
    ('artifact_hashes_passed', False, 'export_artifact_hashes_unverified'),
    ('duration_error_ms', 151, 'export_duration_out_of_tolerance'),
    ('checked_at_unix', NOW + 1, 'export_observation_stale'),
    ('checked_at_unix', NOW - 61, 'export_observation_stale'),
    ('destination_metadata_validated', False, 'destination_metadata_unvalidated'),
])
def test_current_actual_export_observations_are_required(field, value, blocker):
    result = publication(mutate=lambda d, e, p, a: e.update({field: value}))
    assert result['manual_handoff_ready'] is False and blocker in result['manual_blockers']


@pytest.mark.parametrize(('field', 'value'), [('caption', 'different'), ('account', 'different'),
    ('destination', 'youtube'), ('visibility', 'public'), ('timezone', 'Etc/UTC'),
    ('schedule', '2027-01-01T09:00:00'), ('permission_status', 'owned')])
def test_every_reviewed_destination_field_is_bound(field, value):
    result = publication(mutate=lambda d, e, p, a: d.update({field: value}))
    assert not result['manual_handoff_ready']
    assert 'approved_draft_changed' in result['manual_blockers']


@pytest.mark.parametrize(('field', 'value', 'blocker'), [
    ('asset_fingerprint', 'f' * 64, 'approved_asset_changed'),
    ('confirm_manual_handoff', False, 'manual_handoff_not_confirmed'),
    ('confirmed_at_unix', NOW + 1, 'approval_timestamp_invalid'),
])
def test_approval_identity_and_confirmation(field, value, blocker):
    result = publication(mutate=lambda d, e, p, a: a.update({field: value}))
    assert not result['manual_handoff_ready'] and blocker in result['manual_blockers']


def test_missing_export_provenance_or_approval_never_passes():
    d, e, p, a = publication_values()
    for args, blocker in [((d, None, p, a), 'verified_export_missing'),
                          ((d, e, None, a), 'source_provenance_missing'),
                          ((d, e, p, None), 'manual_approval_missing')]:
        result = publishing_preflight(*args, now_unix=NOW)
        assert not result['manual_handoff_ready'] and blocker in result['manual_blockers']


def test_remote_destination_can_be_ready_for_manual_upload_only():
    d, e, p, a = publication_values()
    d.update(destination='youtube', account='operator-selected-channel', schedule='2027-01-01T09:00:00')
    a['draft'] = copy.deepcopy(d)
    result = publishing_preflight(d, e, p, a, now_unix=NOW)
    assert result['manual_handoff_ready'] is True and result['scheduled_metadata_only'] is True
    assert result['can_publish'] is False
    assert 'remote_publication_not_approved' in result['remote_blockers']


def test_no_io_or_credentials_are_needed_and_sensitive_review_text_is_not_returned(monkeypatch):
    import builtins
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError('Preflight must perform no filesystem/network operation.')

    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    assert cloud()['can_execute'] is False
    result = publication()
    assert 'Reviewed caption' not in json.dumps(result)
    assert 'I reviewed the recorded' not in json.dumps(result)
    d, *_ = publication_values(); d['access_token'] = 'not-accepted'
    with pytest.raises(ValidationError):
        DestinationDraft.model_validate(d)


@pytest.mark.parametrize('now', [True, 1.5, -1, float('nan')])
def test_explicit_clock_is_validated(now):
    with pytest.raises(ValueError):
        cloud_analysis_preflight(*cloud_values(), now_unix=now)
    with pytest.raises(ValueError):
        publishing_preflight(*publication_values(), now_unix=now)
