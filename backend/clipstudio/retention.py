"""Bounded, read-only storage inventory and advisory cache retention plans.

There is deliberately no apply/delete operation. A valid verification means only
that the observed candidate metadata and protecting database references still
match. It is neither a lock nor authorization for future deletion.
"""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import tempfile
import time
from urllib.parse import unquote

import psutil

from . import config

VERSION = 1
DB_LIMIT = 64 * 1024**2
JSON_LIMIT = 1024**2
ROW_LIMIT = 50000
METADATA_LIMIT = 32 * 1024**2
TIME_LIMIT = 15
HEX32 = re.compile(r"[a-f0-9]{32}\Z")
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
TEMP = re.compile(r"\.([a-f0-9]{64})-([a-f0-9]{32})\Z")
TABLES = {
    'projects': 'id,status,source,analysis,revision',
    'revisions': 'project_id,revision,fingerprint,timeline',
    'stages': 'project_id,name,fingerprint,artifact_dir,result',
    'exports': 'id,project_id,revision,fingerprint,artifact_dir,manifest',
    'publications': 'id,project_id,export_id,asset_fingerprint,revision,draft,status,approval',
    # Heartbeat clocks and logs are observations, not retention references.
    'jobs': 'id,project_id,kind,status,params,result,lease_token,attempts,cancel_requested',
    'uploads': 'id,status,offset,size,filename,project_id,options',
}
JSON_FIELDS = {'source', 'analysis', 'timeline', 'result', 'manifest', 'draft', 'approval', 'params', 'options'}


class UnsafeSnapshot(ValueError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise UnsafeSnapshot('Unsafe or empty relative artifact path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ('..', '.') for p in value.split('/')) or str(path) != value:
        raise UnsafeSnapshot('Unsafe relative artifact path: ' + value[:160])
    return value


def _meta(value):
    return [value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_nlink,
            value.st_mtime_ns, value.st_ctime_ns, value.st_blocks * 512]


@contextlib.contextmanager
def _root_fd(path):
    """Open every component without following a symlink, including the root."""
    current = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in Path(path).absolute().parts[1:]:
            if component in ('.', '..'):
                raise UnsafeSnapshot('Unsafe data directory')
            new = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = new
        yield current
    finally:
        os.close(current)


@contextlib.contextmanager
def _open_fd(root, relative, *, directory=False):
    parts = _relative(relative).split('/')
    parent = os.dup(root)
    descriptor = None
    try:
        for component in parts[:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK |
                             (os.O_DIRECTORY if directory else 0), dir_fd=parent)
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _current_meta(root, relative, *, directory=False):
    with _open_fd(root, relative, directory=directory) as fd:
        return _meta(os.fstat(fd))


class _Budget:
    def __init__(self):
        self.deadline = time.monotonic() + TIME_LIMIT
        self.json_bytes = 0

    def check(self):
        if time.monotonic() > self.deadline:
            raise UnsafeSnapshot('Storage scan exceeded the time budget')

    def decode(self, value):
        self.check()
        self.json_bytes += len(value)
        if self.json_bytes > METADATA_LIMIT or len(value) > JSON_LIMIT:
            raise UnsafeSnapshot('Metadata exceeds the bounded JSON budget')
        result = json.loads(value)
        # Serialization rejects non-finite values and excessively nested data.
        _digest(result)
        return result


def _read_json(root, relative, expected, budget, limit=JSON_LIMIT):
    with _open_fd(root, relative) as fd:
        before = _meta(os.fstat(fd))
        if before != expected or not stat.S_ISREG(before[2]) or before[3] > limit:
            raise UnsafeSnapshot('Metadata file changed or exceeds its size limit: ' + relative)
        data = os.read(fd, limit + 1)
        if len(data) != before[3] or _meta(os.fstat(fd)) != before:
            raise UnsafeSnapshot('Metadata file changed while being read: ' + relative)
    return budget.decode(data)


def _database_snapshot(root, budget):
    """Read a private DB+WAL copy; mode=ro alone can create original WAL/SHM files."""
    with tempfile.TemporaryDirectory(prefix='vektri-retention-') as scratch:
        snapshots = {}
        total = 0
        for name in ('studio.sqlite3', 'studio.sqlite3-wal'):
            try:
                with _open_fd(root, name) as fd:
                    before = _meta(os.fstat(fd))
                    total += before[3]
                    if not stat.S_ISREG(before[2]) or total > DB_LIMIT:
                        raise UnsafeSnapshot('Database snapshot exceeds the 64 MiB budget or is not a regular file')
                    with open(Path(scratch) / name, 'xb') as out:
                        remaining = before[3]
                        while remaining:
                            budget.check()
                            block = os.read(fd, min(1024**2, remaining))
                            if not block:
                                raise UnsafeSnapshot('Database changed during its snapshot')
                            out.write(block)
                            remaining -= len(block)
                    if _meta(os.fstat(fd)) != before:
                        raise UnsafeSnapshot('Database changed during its snapshot; retry when the library is idle')
                    snapshots[name] = before
            except FileNotFoundError:
                if name == 'studio.sqlite3':
                    raise UnsafeSnapshot('Database is missing; inventory cannot establish protecting references')
                snapshots[name] = None
        for name, before in snapshots.items():
            try:
                after = _current_meta(root, name)
            except FileNotFoundError:
                after = None
            if after != before:
                raise UnsafeSnapshot('Database or WAL changed during its snapshot; retry when the library is idle')
        with contextlib.closing(sqlite3.connect(Path(scratch) / 'studio.sqlite3', timeout=0.5)) as conn:
            conn.row_factory = sqlite3.Row
            conn.set_progress_handler(lambda: int(time.monotonic() > budget.deadline), 1000)
            conn.execute('PRAGMA query_only=ON')
            conn.execute('BEGIN')
            if conn.execute('PRAGMA user_version').fetchone()[0] != 1:
                raise UnsafeSnapshot('Unsupported database schema; retention references cannot be established')
            if conn.execute('PRAGMA quick_check(1)').fetchone()[0] != 'ok':
                raise UnsafeSnapshot('Database snapshot failed its integrity check')
            result = {}
            count = 0
            for table, columns in TABLES.items():
                rows = []
                for record in conn.execute(f'SELECT {columns} FROM {table} ORDER BY {columns.split(",")[0]}'):
                    budget.check()
                    count += 1
                    if count > ROW_LIMIT:
                        raise UnsafeSnapshot('Database reference inventory exceeded its row budget')
                    row = dict(record)
                    for key in JSON_FIELDS & row.keys():
                        if row[key] is not None:
                            row[key] = budget.decode(row[key])
                    rows.append(row)
                result[table] = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
            return result


def reference_snapshot(data_dir=None):
    """Bounded decoded database references for other read-only local preflights.

    Raises on missing, unsupported, unreadable, oversized, or changing snapshots.
    Call again after inspecting files to detect reference changes; no lock is held.
    No files are created in the original database directory, even in WAL mode.
    """
    path = config.DATA if data_dir is None else data_dir
    with _root_fd(path) as root:
        return _database_snapshot(root, _Budget())


def _scan(root, max_entries, max_depth, budget):
    entries = {}

    def walk(parent, prefix, depth):
        budget.check()
        with os.scandir(parent) as iterator:
            for entry in iterator:
                budget.check()
                path = prefix + entry.name
                value = entry.stat(follow_symlinks=False)
                if len(entries) >= max_entries:
                    raise UnsafeSnapshot('Storage scan exceeded its entry limit')
                entries[path] = _meta(value)
                if stat.S_ISLNK(value.st_mode):
                    raise UnsafeSnapshot('Symlink refused; no target was followed: ' + path)
                if stat.S_ISDIR(value.st_mode):
                    if depth >= max_depth:
                        raise UnsafeSnapshot('Storage scan exceeded its depth limit: ' + path)
                    with _open_fd(parent, entry.name, directory=True) as child:
                        if _meta(os.fstat(child)) != entries[path]:
                            raise UnsafeSnapshot('Directory changed during traversal: ' + path)
                        walk(child, path + '/', depth + 1)
                elif not stat.S_ISREG(value.st_mode):
                    raise UnsafeSnapshot('Special file refused: ' + path)
    walk(root, '', 0)
    return entries


def _inventory(entries):
    totals = {}
    unique = {}
    for path, meta in entries.items():
        if not stat.S_ISREG(meta[2]):
            continue
        top = path.split('/')[0]
        category = top if top in {'projects', 'models', 'uploads', 'logs'} else ('database' if top.startswith('studio.sqlite3') else 'other')
        groups = ['total', category]
        if category == 'projects':
            parts = path.split('/')
            kind = parts[2] if len(parts) > 2 else ''
            groups.append('project_' + ('originals' if kind == 'source' else 'exports' if len(parts) > 3 and parts[3] == 'package' else 'cache' if kind == 'artifacts' else 'other'))
        inode = tuple(meta[:2])
        for group in groups:
            amount = totals.setdefault(group, {'file_count': 0, 'logical_bytes': 0, 'unique_inode_bytes': 0, 'allocated_inode_bytes': 0})
            amount['file_count'] += 1
            amount['logical_bytes'] += meta[3]
            seen = unique.setdefault(group, set())
            if inode not in seen:
                seen.add(inode)
                amount['unique_inode_bytes'] += meta[3]
                amount['allocated_inode_bytes'] += meta[7]
    for category in ('total', 'projects', 'models', 'uploads', 'logs', 'database', 'other', 'project_originals', 'project_cache', 'project_exports', 'project_other'):
        totals.setdefault(category, {'file_count': 0, 'logical_bytes': 0, 'unique_inode_bytes': 0, 'allocated_inode_bytes': 0})
    return totals


def _intersects(first, second):
    return first == second or first.startswith(second + '/') or second.startswith(first + '/')


def _references(snapshot):
    protected = {'models': {'Installed models'}, 'uploads': {'Uploads and resumable upload data'},
                 'studio.sqlite3': {'Project metadata, transcripts, all saved revisions, exports, and publication records'},
                 'studio.sqlite3-wal': {'Committed database changes'}, 'studio.sqlite3-shm': {'Database coordination state'},
                 'logs': {'Runtime and audit logs are outside the cache retention scope'}}
    projects = {}
    busy = []

    def protect(path, reason):
        protected.setdefault(_relative(path), set()).add(reason)

    def ref(pid, value, reason):
        if not HEX32.fullmatch(pid or ''):
            raise UnsafeSnapshot('Invalid project identifier in protecting database references')
        path = _relative(value)
        protect('projects/' + pid + '/' + path, reason)

    def nested(pid, value, reason, base=None, depth=0):
        if depth > 32:
            raise UnsafeSnapshot('Artifact references exceed the nesting limit')
        if isinstance(value, list):
            for item in value:
                nested(pid, item, reason, base, depth + 1)
        elif isinstance(value, dict):
            local_base = value.get('artifact_dir', base)
            for key, item in value.items():
                if key in {'artifact_dir', 'prepared_dir', 'render_folder'} and item is not None:
                    ref(pid, item, reason)
                elif key == 'ref' and isinstance(item, str):
                    location = unquote(item.split('#', 1)[0])
                    if location.startswith('/api/v1/artifacts/'):
                        fields = location.split('/', 5)
                        if len(fields) != 6:
                            raise UnsafeSnapshot('Malformed artifact evidence URL')
                        ref(fields[4], fields[5], reason)
                    elif location.startswith(('artifacts/', 'source/')):
                        ref(pid, location, reason)
                    elif location and ('/' in location or '.' in location):
                        if local_base and not location.startswith(('/', 'http:', 'https:')):
                            ref(pid, _relative(local_base) + '/' + _relative(location), reason)
                        else:
                            # Unresolved references must never justify removing old evidence.
                            protect('projects/' + pid + '/artifacts', 'Unresolved evidence reference; all project artifacts retained')
                nested(pid, item, reason, local_base, depth + 1)

    for project in snapshot['projects']:
        pid = project['id']
        if not isinstance(pid, str) or not HEX32.fullmatch(pid):
            raise UnsafeSnapshot('Invalid project identifier in database')
        projects[pid] = project
        protect('projects/' + pid + '/source', 'Immutable original media')
        source = project['source']
        if not isinstance(source, dict):
            raise UnsafeSnapshot('Invalid source metadata')
        if source.get('path'):
            ref(pid, source['path'], 'Current original source')
        nested(pid, source, 'Current prepared source')
        nested(pid, project['analysis'], 'Current analysis and evidence')
        if project['status'] in {'importing', 'queued'}:
            busy.append('Project import is active: ' + pid)
            protect('projects/' + pid, 'Active project import')
    exports = {row['id']: row for row in snapshot['exports']}
    for row in snapshot['exports']:
        ref(row['project_id'], row['artifact_dir'], 'Saved export (all revisions retained)')
        nested(row['project_id'], row['manifest'], 'Saved export evidence')
    for row in snapshot['publications']:
        if row['export_id'] not in exports:
            raise UnsafeSnapshot('Publication references a missing export')
        nested(row['project_id'], row['draft'], 'Publication draft evidence')
        nested(row['project_id'], row['approval'], 'Publication approval evidence')
    for row in snapshot['revisions']:
        nested(row['project_id'], row['timeline'], 'Saved revision evidence')
    for row in snapshot['jobs']:
        if row['status'] not in {'queued', 'running', 'succeeded', 'failed', 'cancelled'}:
            raise UnsafeSnapshot('Unknown job status; active work cannot be ruled out safely')
        if row['status'] in {'queued', 'running'}:
            busy.append('Queued or running job: ' + row['id'])
            if row['project_id']:
                protect('projects/' + row['project_id'], 'Queued or running job protects its project')
        if row['project_id']:
            nested(row['project_id'], row['params'], 'Job input artifact')
            nested(row['project_id'], row['result'], 'Job result artifact')
    if any(row['status'] != 'complete' for row in snapshot['uploads']):
        busy.append('Resumable or incomplete uploads are present')
    return projects, protected, busy


def _worker_observation(root, entries, budget):
    workers = []
    uncertain = False
    for path, meta in entries.items():
        if not path.startswith('logs/workers/') or not path.endswith('.json'):
            continue
        try:
            entry = _read_json(root, path, meta, budget, 16384)
            pid, started, beat = entry['pid'], entry['process_started_at'], entry['heartbeat_at']
            if type(pid) is not int or not 0 < pid < 2**31:
                raise ValueError('Invalid worker pid')
            if any(type(v) not in (float, int) or not math.isfinite(v) for v in (started, beat)):
                raise ValueError('Invalid worker timestamps')
            if not 0 <= time.time() - beat <= 10:
                continue
            process = psutil.Process(pid)
            if abs(process.create_time() - started) < .01 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                workers.append({'pid': pid, 'state': entry.get('state', 'unknown')})
        except psutil.NoSuchProcess:
            continue
        except (OSError, ValueError, KeyError, TypeError, psutil.Error):
            uncertain = True
    return {'alive': bool(workers), 'workers': workers, 'uncertain': uncertain}


def _candidate_paths(entries, prefix, budget):
    result = {}
    for index, (path, meta) in enumerate(entries.items()):
        if index % 128 == 0:
            budget.check()
        if path == prefix or path.startswith(prefix + '/'):
            result[path] = meta
    return result


def _candidates(root, entries, snapshot, projects, protected, min_age, budget):
    stages = {(row['project_id'], row['name'], row['fingerprint']): row for row in snapshot['stages']}
    revisions = {}
    for row in snapshot['revisions']:
        revisions.setdefault(row['project_id'], set()).add(_digest(row['timeline']))
    live_tokens = {row['lease_token'] for row in snapshot['jobs'] if row['status'] in {'queued', 'running'}}
    candidates, retained, anchors = [], [], {}
    now = time.time()
    for path, meta in sorted(entries.items()):
        budget.check()
        parts = path.split('/')
        if len(parts) != 5 or parts[0] != 'projects' or parts[2] != 'artifacts' or not stat.S_ISDIR(meta[2]):
            continue
        pid, kind, name = parts[1], parts[3], parts[4]
        if pid not in projects or kind not in {'prepare', 'render'}:
            continue
        if any(_intersects(path, target) for target in protected):
            continue
        tree = _candidate_paths(entries, path, budget)
        tmp = TEMP.fullmatch(name)
        if tmp:
            if tmp[2] in live_tokens or now - max(m[5] / 10**9 for m in tree.values()) < min_age:
                continue
            reason = 'Abandoned worker temporary directory older than the retention age; no active lease token'
            candidate_kind = 'abandoned_temporary'
        else:
            if not HEX64.fullmatch(name):
                continue
            row = stages.get((pid, kind, name))
            if not row or row['artifact_dir'] != '/'.join(parts[2:]):
                continue
            source = projects[pid]['source']
            source_path = 'projects/' + pid + '/' + _relative(source.get('path', ''))
            source_meta = entries.get(source_path)
            sha = source.get('sha256')
            if not source_path.startswith('projects/' + pid + '/source/') or not source_meta or not stat.S_ISREG(source_meta[2]) or not isinstance(sha, str) or not HEX64.fullmatch(sha):
                retained.append({'path': path, 'reason': 'Reproducibility cannot be established from a retained original'})
                continue
            marker = path + '/complete.json'
            if marker not in tree:
                retained.append({'path': path, 'reason': 'Completion receipt missing'})
                continue
            try:
                receipt = _read_json(root, marker, tree[marker], budget)
                checksums = receipt['checksums']
                if not isinstance(checksums, dict) or receipt['fingerprint'] != name or receipt['result'] != row['result']:
                    raise ValueError('Receipt does not match the registered stage')
                expected_files = set()
                for relative, checksum in checksums.items():
                    expected_files.add(path + '/' + _relative(relative))
                    if not isinstance(checksum, str) or not HEX64.fullmatch(checksum):
                        raise ValueError('Invalid receipt checksum')
                actual_files = {p for p, m in tree.items() if stat.S_ISREG(m[2]) and p != marker}
                expected_dirs = {path}
                for file_path in expected_files:
                    parent = file_path.rsplit('/', 1)[0]
                    while parent != path:
                        expected_dirs.add(parent)
                        parent = parent.rsplit('/', 1)[0]
                actual_dirs = {p for p, m in tree.items() if stat.S_ISDIR(m[2])}
                if actual_files != expected_files or actual_dirs != expected_dirs:
                    raise ValueError('Receipt inventory has extra or missing files/directories')
                result = row['result']
                if not isinstance(result, dict) or result.get('sha256' if kind == 'prepare' else 'source_sha256') != sha:
                    raise ValueError('Receipt does not identify the retained original')
                if kind == 'render' and (_digest(result.get('timeline')) not in revisions.get(pid, set()) or result.get('timeline_sha256') != _digest(result.get('timeline'))):
                    raise ValueError('Render recipe is not a retained timeline revision')
            except (ValueError, KeyError, TypeError, RecursionError) as exc:
                retained.append({'path': path, 'reason': str(exc)[:200]})
                continue
            anchors[source_path] = source_meta
            reason = 'Unreferenced registered ' + kind + ' cache with exact receipt inventory and retained source' + (' and timeline revision' if kind == 'render' else '')
            candidate_kind = 'reproducible_cache'
        candidates.append({'path': path, 'kind': candidate_kind, 'reason': reason,
                           'file_count': sum(stat.S_ISREG(m[2]) for m in tree.values()),
                           'logical_bytes': sum(m[3] for m in tree.values() if stat.S_ISREG(m[2])),
                           'metadata_fingerprint': _digest(tree)})
        if len(candidates) > 1000:
            raise UnsafeSnapshot('Candidate count exceeds the bounded review limit')
    return candidates, retained, anchors


def _savings(entries, candidates, budget):
    by_inode = {}
    for path, meta in entries.items():
        budget.check()
        if stat.S_ISREG(meta[2]):
            by_inode.setdefault(tuple(meta[:2]), []).append((path, meta))
    prefixes = {row['path'] for row in candidates}
    estimates = {path: 0 for path in prefixes}
    total = 0
    for links in by_inode.values():
        budget.check()
        # Candidate roots always have five components: projects/id/artifacts/stage/hash.
        owners = []
        for path, _ in links:
            prefix = '/'.join(path.split('/')[:5])
            owners.append(prefix if prefix in prefixes else None)
        meta = links[0][1]
        if len(links) != meta[4] or None in owners or any(row[1] != meta for row in links):
            continue
        estimate = min(meta[3], meta[7])
        total += estimate
        if len(set(owners)) == 1:
            estimates[owners[0]] += estimate
    for candidate in candidates:
        candidate['estimated_reclaimable_bytes'] = estimates[candidate['path']]
    return total


def _seal(plan):
    return _digest({key: plan[key] for key in ('schema_version', 'data_directory', 'settings',
                                              'snapshot_fingerprint', 'review_candidates')})


def plan(data_dir=None, *, max_entries=50000, max_depth=24, min_temp_age_seconds=86400):
    """Return an advisory inventory. Never alter the library or authorize removal."""
    if type(max_entries) is not int or not 1 <= max_entries <= 50000 or type(max_depth) is not int or not 1 <= max_depth <= 24:
        raise ValueError('Traversal limits must be positive and at most 50000 entries / 24 levels')
    if type(min_temp_age_seconds) not in (int, float) or not math.isfinite(min_temp_age_seconds) or min_temp_age_seconds < 86400:
        raise ValueError('Temporary directory retention must be at least 86400 seconds')
    root_path = str(Path(config.DATA if data_dir is None else data_dir).absolute())
    result = {'schema_version': VERSION, 'data_directory': root_path, 'generated_at': time.time(),
              'read_only': True, 'automatic_deletion_supported': False, 'status': 'review_only',
              'maintenance_required': False, 'maintenance_reasons': [], 'blockers': [],
              'settings': {'max_entries': max_entries, 'max_depth': max_depth, 'min_temp_age_seconds': min_temp_age_seconds},
              'inventory': {}, 'inventory_complete': False, 'protected_paths': [], 'retained_caches': [],
              'review_candidates': [], 'estimated_reclaimable_bytes': 0, 'snapshot_fingerprint': None,
              'limitations': ['Advisory metadata snapshot only; no deletion, move, archive, or apply operation exists.',
                             'Receipt inventories are checked; original, model, and cached media bytes are not rehashed.',
                             'Hardlinks are counted once; links outside this inventory or to retained files yield zero estimated savings.',
                             'Estimates exclude directory blocks and cannot predict filesystem deduplication, compression, or future writes.',
                             'Regeneration may differ after software changes; all candidates require operator review.',
                             'Only registered unreferenced prepare/render caches and old known worker temporaries are considered.']}
    budget = _Budget()
    try:
        with _root_fd(root_path) as root:
            snapshot = _database_snapshot(root, budget)
            entries = _scan(root, max_entries, max_depth, budget)
            result['inventory'] = _inventory(entries)
            result['inventory_complete'] = True
            projects, protected, busy = _references(snapshot)
            worker = _worker_observation(root, entries, budget)
            if worker['alive']:
                busy.append('A worker process is live; future maintenance requires stopping it, including when idle')
            if worker['uncertain']:
                busy.append('Worker state could not be established safely')
            result['worker'] = worker
            result['maintenance_reasons'] = busy
            result['maintenance_required'] = bool(busy)
            result['protected_paths'] = [{'path': path, 'reasons': sorted(reasons)} for path, reasons in sorted(protected.items())]
            candidates, retained, anchors = _candidates(root, entries, snapshot, projects, protected, min_temp_age_seconds, budget)
            candidate_metadata = dict(anchors)
            for row in candidates:
                candidate_metadata.update(_candidate_paths(entries, row['path'], budget))
            for path, expected in candidate_metadata.items():
                budget.check()
                if _current_meta(root, path, directory=stat.S_ISDIR(expected[2])) != expected:
                    raise UnsafeSnapshot('Candidate or retained source changed during planning: ' + path)
            if snapshot != _database_snapshot(root, budget):
                raise UnsafeSnapshot('Protecting database references changed during planning; retry when the library is idle')
            result['estimated_reclaimable_bytes'] = _savings(entries, candidates, budget)
            result['review_candidates'] = candidates
            result['retained_caches'] = retained
            result['snapshot_fingerprint'] = _digest({'references': snapshot, 'files': candidate_metadata})
            if busy:
                result['status'] = 'maintenance_required'
    except (OSError, ValueError, TypeError, KeyError, RecursionError, sqlite3.Error) as exc:
        result['status'] = 'blocked'
        result['maintenance_required'] = True
        result['blockers'].append(str(exc)[:500])
        result['review_candidates'] = []
        result['estimated_reclaimable_bytes'] = 0
        result['snapshot_fingerprint'] = None
    result['plan_fingerprint'] = _seal(result)
    return result


def verify_plan(previous, data_dir=None):
    """Refresh a plan read-only; validity is not a deletion authorization or lock."""
    reasons = []
    root_path = str(Path(config.DATA if data_dir is None else data_dir).absolute())
    try:
        if not isinstance(previous, dict) or previous['schema_version'] != VERSION or previous['data_directory'] != root_path:
            raise ValueError('Plan version or library directory does not match')
        if previous['plan_fingerprint'] != _seal(previous):
            raise ValueError('Plan contents changed or its fingerprint is invalid')
        current = plan(root_path, **previous['settings'])
        if previous['snapshot_fingerprint'] is None or current['snapshot_fingerprint'] is None:
            reasons.append('A blocked or incomplete plan cannot be verified')
        elif current['snapshot_fingerprint'] != previous['snapshot_fingerprint'] or current['review_candidates'] != previous['review_candidates']:
            reasons.append('Candidate metadata or protecting database references changed; generate a new plan')
        reasons.extend(current['blockers'])
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        reasons.append(str(exc))
        current = None
    return {'valid': not reasons, 'read_only': True, 'automatic_deletion_supported': False,
            'reasons': reasons, 'current_fingerprint': current['snapshot_fingerprint'] if current else None,
            'maintenance_required': current['maintenance_required'] if current else True,
            'maintenance_reasons': current['maintenance_reasons'] if current else [],
            'meaning': 'Verification is an advisory refresh, not deletion authorization or a lock.'}
