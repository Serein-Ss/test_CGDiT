"""Verify the versioned input snapshot without importing the ML environment."""

import hashlib
import json
from pathlib import Path


def verify(root: Path) -> list[str]:
    manifest = json.loads((root / "data/manifest.json").read_text())
    errors = []
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"Missing: {item['path']}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            errors.append(f"Changed: {item['path']}")
    return errors


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    errors = verify(root)
    if errors:
        raise SystemExit("\n".join(errors))
    print("All versioned input hashes match data/manifest.json")


if __name__ == "__main__":
    main()
