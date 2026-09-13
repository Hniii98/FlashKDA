"""Verify source provenance for the native fused header.

Original sources are audit inputs, not compilation units. Missing references
are downloaded from the pinned commit; pass --reference-dir to use a local copy.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path,
                        default=ROOT / ".cache/fused-validation/header-rewrite/upstream")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "flash_kda/fused_registry.json").read_text())
    header = (ROOT / "csrc/smxx/fwd_kernel_fused.cuh").read_text()
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    checked = set()
    for variant in manifest["variants"]:
        assert variant["entry"] in header, variant["entry"]
        path = args.reference_dir / (variant["name"] + ".inc")
        if not path.exists():
            url = ("https://raw.githubusercontent.com/flashinfer-ai/flashinfer/"
                   + manifest["commit"] + "/" + variant["source"])
            with urlopen(url, timeout=60) as response:
                data = response.read()
            assert hashlib.sha256(data).hexdigest() == variant["sha256"], url
            path.write_bytes(data)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == variant["sha256"], path
        checked.add(path)
    print(f"Verified {len(checked)} upstream reference bodies from {manifest['commit']}")
    print("The adapted header is validated with tools/check_fused_port.py and torch/FLA tests.")


if __name__ == "__main__":
    main()
