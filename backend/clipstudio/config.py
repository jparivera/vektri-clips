from pathlib import Path
import os

ROOT = Path(os.environ.get('VEKTRI_APP_DIR', Path(__file__).resolve().parents[2])).expanduser().resolve()
DATA = Path(os.environ.get('VEKTRI_DATA_DIR', ROOT / 'data')).expanduser().resolve()
DB = DATA / 'studio.sqlite3'
PORT = int(os.environ.get('VEKTRI_PORT', '8741'))
MAX_UPLOAD = int(os.environ.get('VEKTRI_MAX_UPLOAD_BYTES', str(4 * 1024**3)))
MAX_URL = min(MAX_UPLOAD, 512 * 1024**2)
LEASE_SECONDS = 20
MAX_ATTEMPTS = 3
MIN_FREE_BYTES = 256 * 1024**2

def initialize():
    DATA.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ('projects', 'uploads', 'models', 'logs'):
        (DATA / name).mkdir(exist_ok=True, mode=0o700)

def project_dir(project_id: str) -> Path:
    if len(project_id) != 32 or any(c not in '0123456789abcdef' for c in project_id):
        raise ValueError('Invalid project identifier')
    return DATA / 'projects' / project_id
