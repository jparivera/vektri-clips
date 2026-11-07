"""Explicit full backup/restore. No cleanup ever deletes originals implicitly."""
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from . import config,db
from .security import checksum,free_space

def backup(destination):
    destination=Path(destination).expanduser().resolve()
    if destination.exists():raise ValueError('Backup destination already exists; choose a new filename')
    if destination.is_relative_to(config.DATA):raise ValueError('Keep backups outside the data directory')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if db.one("SELECT 1 FROM jobs WHERE status IN ('running','queued')"):
        raise ValueError('Finish or cancel pending jobs and stop the app before a full backup')
    if db.one("SELECT 1 FROM uploads WHERE status!='complete'"):
        raise ValueError('Finish incomplete uploads before backup; resumable upload bytes are not included in the archive')
    temp=destination.with_suffix(destination.suffix+'.partial')
    if temp.exists():raise ValueError('Partial backup exists; choose a different name or inspect it first')
    # Exclusive creation protects concurrent backup attempts and keeps private
    # media unreadable by other local users throughout archive construction.
    temp.touch(mode=0o600,exist_ok=False)
    try:
        # Hold write lock to ensure API edits cannot move metadata during snapshot/copy.
        with db.tx() as lock, tempfile.TemporaryDirectory(prefix='vektri-backup-') as d:
            if lock.execute("SELECT 1 FROM jobs WHERE status IN ('running','queued')").fetchone():raise ValueError('A job started; stop the app before backup')
            if lock.execute("SELECT 1 FROM uploads WHERE status!='complete'").fetchone():raise ValueError('Finish incomplete uploads before backup')
            snapshot=Path(d)/'studio.sqlite3'
            src=sqlite3.connect(config.DB);dst=sqlite3.connect(snapshot)
            try:src.backup(dst)
            finally:src.close();dst.close()
            files=[p for folder in ('projects','models') for p in (config.DATA/folder).rglob('*') if p.is_file() and not p.is_symlink() and not any(x.startswith('.') for x in p.relative_to(config.DATA).parts)]
            total=sum(p.stat().st_size for p in files)+snapshot.stat().st_size
            free_space(destination.parent,total)
            manifest={'schema_version':1,'database_version':1,'files':{'studio.sqlite3':checksum(snapshot)}}
            # Cached artifacts can be hard-linked. Store complete regular-file
            # bytes so restore can continue rejecting all archive link entries.
            with tarfile.open(temp,'w:gz',compresslevel=1,dereference=True) as archive:
                archive.add(snapshot,arcname='studio.sqlite3')
                for p in files:
                    rel=str(p.relative_to(config.DATA));manifest['files'][rel]=checksum(p);archive.add(p,arcname=rel,recursive=False)
                m=Path(d)/'backup-manifest.json';m.write_text(json.dumps(manifest,indent=2));archive.add(m,arcname=m.name)
        os.link(temp,destination);temp.unlink()
        return {'path':str(destination),'bytes':destination.stat().st_size,'files':len(manifest['files']),'sha256':checksum(destination),'contains_originals':True}
    except BaseException:
        temp.unlink(missing_ok=True);raise

def restore(archive_path,destination):
    archive_path=Path(archive_path).expanduser().resolve(strict=True)
    destination=Path(destination).expanduser().resolve()
    if destination.exists():raise ValueError('Restore requires a new, nonexistent destination; existing data is never overwritten')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name(destination.name+'.restoring')
    if temp.exists():raise ValueError('A prior restore directory exists; inspect it before retrying')
    temp.mkdir(mode=0o700)
    try:
        with tarfile.open(archive_path,'r:gz') as archive:
            members=archive.getmembers()
            if len(members)>200000:raise ValueError('Archive contains too many files')
            names=set();total=0
            for m in members:
                p=Path(m.name)
                if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk() or not m.isfile():raise ValueError('Unsafe archive member')
                if m.name in names:raise ValueError('Duplicate archive member')
                names.add(m.name);total+=m.size
                if not (m.name in ('studio.sqlite3','backup-manifest.json') or p.parts[0] in ('projects','models')):raise ValueError('Unexpected backup content')
            free_space(destination.parent,total)
            for m in members:
                p=temp/m.name;p.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(m) as src,p.open('xb') as out:shutil.copyfileobj(src,out,1024*1024)
        manifest=json.loads((temp/'backup-manifest.json').read_text())
        if manifest['schema_version']!=1 or manifest['database_version']!=1:raise ValueError('Unsupported backup version')
        if set(manifest['files'])!=names-{'backup-manifest.json'}:raise ValueError('Backup manifest does not match archive')
        for rel,h in manifest['files'].items():
            if checksum(temp/rel)!=h:raise ValueError('Backup checksum mismatch: '+rel)
        conn=sqlite3.connect(temp/'studio.sqlite3')
        try:
            if conn.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Database integrity check failed')
        finally:conn.close()
        os.replace(temp,destination)
        for p in (destination/'projects').glob('*/source/*'):
            if p.is_file():p.chmod(0o444)
        return {'path':str(destination),'verified_files':len(manifest['files']),'next':'Set VEKTRI_DATA_DIR to this path, then start Clip Studio.'}
    except BaseException:
        shutil.rmtree(temp);raise
