"""Heuristic code-only release checks; not a privacy or license certification."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST.json"
ROOT_FILES = {"README.md", "README_zh.md", "SPEC.md", "pyproject.toml", ".gitignore",
              ".gitattributes", "LICENSE", "LICENSE.md", "CITATION.cff", MANIFEST}
ALLOWED_SUFFIXES = {"src": {".py"}, "tests": {".py"}, "scripts": {".py", ".sh"},
                    "configs": {".yaml", ".yml"}, "docs": {".md"}, ".github": {".yml", ".yaml"}}
DEV_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
            ".ruff_cache", "build", "dist", ".idea", ".vscode"}
PATTERNS = {
    "GitHub credential pattern": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "API credential pattern": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "private-key marker": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "quoted credential assignment": re.compile(
        r"\b(?:api[_-]?key|password|passwd|secret|access[_-]?token)\s*[:=]\s*[\"'][A-Za-z0-9_+/.=-]{12,}[\"']",
        re.IGNORECASE),
    "private Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9_.-]+[\\/]"),
    "private Unix user path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
}


def inspect_tree(root: Path):
    inventory, problems, ignored = [], [], []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in DEV_DIRS or part.endswith(".egg-info") for part in rel.parts):
            if path.is_file():
                ignored.append(rel.as_posix())
            continue
        if path.is_symlink():
            problems.append(f"Symlink not permitted: {rel.as_posix()}")
            continue
        if not path.is_file():
            continue
        name = rel.as_posix()
        if name == MANIFEST:
            continue
        if len(rel.parts) == 1:
            permitted = name in ROOT_FILES
        else:
            permitted = path.suffix in ALLOWED_SUFFIXES.get(rel.parts[0], set())
        if not permitted or path.name.endswith((".local.yaml", ".local.yml")):
            problems.append(f"Non-release file: {name}")
            continue
        if path.stat().st_size > 1_000_000:
            problems.append(f"Unexpectedly large source/document: {name}")
            continue
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"Not UTF-8 source/document: {name}")
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                # Never echo the possible sensitive value.
                problems.append(f"{label}: {name}")
        inventory.append({"path": name, "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest()})
    return inventory, problems, ignored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true",
                        help="Regenerate inventory only after manual review of intentional edits")
    args = parser.parse_args(argv)
    inventory, problems, ignored = inspect_tree(ROOT)
    manifest_path = ROOT / MANIFEST
    if not inventory:
        problems.append("Release tree is empty.")
    if problems:
        for problem in problems:
            print("FAIL:", problem)
        return 1
    if args.write_manifest:
        manifest = {"schema_version": 1, "prepared_date": "2026-10-21",
                    "scope": "Code-only local release candidate; no clinical results or patient records",
                    "license_status": "Author selection pending; no open-source license assumed",
                    "files": inventory}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {MANIFEST}; this is not manual privacy/license approval.")
    elif not manifest_path.is_file():
        print(f"FAIL: {MANIFEST} is missing; review files before --write-manifest.")
        return 1
    else:
        try:
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"FAIL: Manifest cannot be read ({type(exc).__name__}).")
            return 1
        if saved.get("files") != inventory:
            print("FAIL: Inventory/checksums differ from the release manifest. "
                  "Review intentional edits before regenerating it.")
            return 1
    print(f"PASS: {len(inventory)} source/document files; no prohibited file types "
          "or configured secret/path patterns detected.")
    if ignored:
        print(f"NOTE: {len(ignored)} local development artifacts ignored; do not upload them.")
    print("Manual privacy review, code-ownership confirmation and license selection remain necessary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
