"""Bounded DNS-pinned public-media fetches; no cookies, proxies or platform bypass."""
import hashlib
import http.client
import ipaddress
import os
import shutil
import socket
import ssl
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from . import config

class ImportRejected(ValueError): pass

def clean_name(name):
    name=Path(name.replace('\\','/')).name.strip()
    name=''.join(c for c in name if c.isprintable() and c not in '\x00')
    return name[:180] or 'Imported media'

def free_space(path: Path,needed=0):
    if shutil.disk_usage(path).free < config.MIN_FREE_BYTES+needed:
        raise OSError('Not enough free disk space. Free space or choose a larger data directory; originals were preserved.')

def checksum(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def validate_url(url: str):
    if not isinstance(url,str) or len(url)>2048 or any(ord(c)<32 for c in url):
        raise ImportRejected('URL is invalid or too long')
    try:
        p=urlsplit(url)
        port=p.port or (443 if p.scheme=='https' else 80)
    except ValueError:raise ImportRejected('Malformed URL') from None
    if p.scheme not in ('http','https') or not p.hostname or p.username or p.password:
        raise ImportRejected('Use a direct public HTTP(S) media URL without credentials. Local files use the file picker.')
    if port not in (80,443):raise ImportRejected('Only public HTTP/HTTPS ports 80 and 443 are supported')
    host=p.hostname.encode('idna').decode('ascii').lower().rstrip('.')
    if host=='localhost' or host.endswith(('.localhost','.local','.internal')):
        raise ImportRejected('Local and internal hosts are not allowed')
    try:addresses=socket.getaddrinfo(host,port,type=socket.SOCK_STREAM)
    except socket.gaierror:raise ImportRejected('The media host could not be resolved') from None
    ips=sorted({r[4][0] for r in addresses})
    if not ips:raise ImportRejected('The media host has no address')
    for ip in ips:
        addr=ipaddress.ip_address(ip)
        if not addr.is_global or getattr(addr,'ipv4_mapped',None) and not addr.ipv4_mapped.is_global:
            raise ImportRejected('Private, loopback, reserved, link-local and metadata addresses are blocked')
    return p,host,port,ips

class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self,host,port,address,timeout):
        super().__init__(host,port,timeout=timeout,context=ssl.create_default_context())
        self.address=address
    def connect(self):
        # Pin the validated IP while preserving hostname certificate/SNI verification.
        sock=socket.create_connection((self.address,self.port),self.timeout)
        self.sock=self._context.wrap_socket(sock,server_hostname=self.host)

class PinnedHTTP(http.client.HTTPConnection):
    def __init__(self,host,port,address,timeout):
        super().__init__(host,port,timeout=timeout);self.address=address
    def connect(self):self.sock=socket.create_connection((self.address,self.port),self.timeout)

def fetch_public_media(url: str,destination: Path,check=lambda:None,max_bytes=None,deadline_seconds=300):
    max_bytes=max_bytes or config.MAX_URL
    started=time.monotonic(); original=url
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.download')
    try:
        for redirect in range(6):
            check()
            if time.monotonic()-started>deadline_seconds:raise ImportRejected('Download time limit exceeded')
            p,host,port,ips=validate_url(url)
            # A host may advertise IPv6 when this laptop has only IPv4 routing.
            # Retry only already-validated addresses, with the same host/SNI.
            conn=None;response=None;last_error=None
            path=urlunsplit(('', '',p.path or '/',p.query,''))
            for address in sorted(ips,key=lambda ip:(ipaddress.ip_address(ip).version,ip))[:8]:
                check()
                remaining=deadline_seconds-(time.monotonic()-started)
                if remaining<=0:raise ImportRejected('Download time limit exceeded')
                conn=(PinnedHTTPS if p.scheme=='https' else PinnedHTTP)(host,port,address,min(15,remaining))
                try:
                    conn.request('GET',path,headers={'User-Agent':'VektriClipStudio/0.1 direct-media-import','Accept':'video/*,audio/*,application/octet-stream','Accept-Encoding':'identity'})
                    response=conn.getresponse();break
                except (OSError,http.client.HTTPException) as exc:
                    last_error=exc;conn.close()
            if response is None:raise ImportRejected('Could not connect to the public media host: '+str(last_error))
            try:
                if response.status in (301,302,303,307,308):
                    location=response.getheader('Location')
                    if not location:raise ImportRejected('Redirect has no destination')
                    next_url=urljoin(url,location)
                    if p.scheme=='https' and urlsplit(next_url).scheme!='https':raise ImportRejected('HTTPS downgrade redirects are blocked')
                    url=next_url
                    continue
                if response.status!=200:
                    raise ImportRejected(f'Media host returned HTTP {response.status}. Use an authorized local file if access is restricted.')
                content_type=(response.getheader('Content-Type') or '').split(';')[0].lower()
                if not (content_type.startswith(('video/','audio/')) or content_type in ('application/octet-stream','binary/octet-stream')):
                    raise ImportRejected('This URL is not a direct media file. Platform pages and playlists are unsupported; import an authorized local file.')
                if content_type in ('application/vnd.apple.mpegurl','audio/x-mpegurl'):
                    raise ImportRejected('Streaming playlists are unsupported; use a standalone media file')
                if response.getheader('Content-Encoding') not in (None,'identity'):raise ImportRejected('Compressed transfers are unsupported')
                size=response.getheader('Content-Length')
                if size and (not size.isdigit() or int(size)>max_bytes):raise ImportRejected('Download exceeds the 512 MiB URL import limit')
                free_space(destination.parent,min(int(size or 0),max_bytes))
                total=0
                with temporary.open('xb') as f:
                    while True:
                        check()
                        if time.monotonic()-started>deadline_seconds:raise ImportRejected('Download time limit exceeded')
                        b=response.read1(256*1024)
                        if not b:break
                        total+=len(b)
                        if total>max_bytes:raise ImportRejected('Download size limit exceeded')
                        free_space(destination.parent,len(b))
                        f.write(b)
                    f.flush();os.fsync(f.fileno())
                if total==0:raise ImportRejected('The host returned an empty file')
                if size and total!=int(size):raise ImportRejected('Download interrupted; declared and received sizes differ')
                os.replace(temporary,destination)
                return {'original_url':original,'resolved_url':url,'bytes':total,'content_type':content_type,'redirects':redirect,'adapter':'direct-public-media-v1'}
            finally:conn.close()
        raise ImportRejected('Too many redirects; maximum is five')
    finally:
        temporary.unlink(missing_ok=True)

def authorized_artifact(project_id: str,relative: str):
    root=config.project_dir(project_id).resolve()
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts or any(part.startswith('.') for part in p.parts):raise ImportRejected('Artifact path is not allowed')
    if not p.parts or p.parts[0] not in ('source','artifacts'):raise ImportRejected('Artifact is not a project output')
    target=root/p
    if any(part.is_symlink() for part in [target,*target.parents] if part!=root.parent):raise ImportRejected('Symlink artifacts are not allowed')
    if not target.resolve().is_relative_to(root) or not target.is_file():raise FileNotFoundError('Artifact not found')
    if target.suffix.lower() not in ('.mp4','.webm','.mkv','.mov','.wav','.mp3','.flac','.ogg','.bin','.jpg','.jpeg','.png','.json','.srt','.vtt','.zip','.ass'):
        raise ImportRejected('Artifact type is not served')
    return target
