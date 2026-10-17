"""Check the published file manifest, JSON finiteness and solver syntax.

This is a lightweight package check; it does not rerun the long integrations.
Run from any directory with Python 3. No third-party packages are required.
"""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent


def check_finite(value, location):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_finite(item, f"{location}[{index}]")


def main():
    manifest = json.loads((ROOT / 'MANIFEST.sha256.json').read_text(encoding='utf-8'))
    for relative, expected in manifest.items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
    original_root = ROOT / 'code_revision2'
    original_manifest = json.loads((original_root / 'manifest.sha256.json').read_text(encoding='utf-8'))
    for relative, expected in original_manifest.items():
        if hashlib.sha256((original_root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Inner SHA256 mismatch: {relative}")
    scripts = sorted((ROOT / 'code_revision2').glob('*.py'))
    for path in scripts:
        compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
    outputs = sorted((ROOT / 'code_revision2' / 'results').glob('*.json'))
    for path in outputs:
        check_finite(json.loads(path.read_text(encoding='utf-8')), path.name)
    print(f"PASS: {len(manifest)} file hashes; {len(scripts)} Python scripts; "
          f"{len(outputs)} finite JSON output files.")


if __name__ == '__main__':
    main()
