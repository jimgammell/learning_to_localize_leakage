from pathlib import Path
import hashlib
from functools import partial

def get_sha256_hash(path: Path, block_size: int = 4096) -> str:
    assert path.exists()
    sha256_hash = hashlib.sha256()
    with open(path, 'rb') as f:
        for byte_block in iter(partial(f.read, 256), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()