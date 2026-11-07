"""Create (or continue) a Zenodo draft, upload files, and set the metadata. Never publishes: review the draft in the
Zenodo web interface and press Publish there.

Token: Zenodo -> Account -> Applications -> Personal access tokens -> New token with scopes deposit:write and
deposit:actions; pass it in the environment variable ZENODO_TOKEN (never on the command line or in a file in the repo).

  export ZENODO_TOKEN=...   # in the shell only
  python release/zenodo_upload.py --metadata release/zenodo_dataset.json --files release/out_v4/zenodo/* release/raw_zenodo/*.tar.gz
  python release/zenodo_upload.py --metadata ... --deposition 1234567 --files more_files...    # continue a draft
  --sandbox uses sandbox.zenodo.org (separate account and token) for a test run.

The first call reserves the DOI (printed; also shown in the draft) so that it can go into the paper before publishing.
Files already in the draft with the same name and size are skipped, so an interrupted upload can simply be rerun.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="JSON with the Zenodo 'metadata' object")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--deposition", type=int, help="id of an existing draft to continue")
    ap.add_argument("--sandbox", action="store_true")
    a = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN") or sys.exit("set ZENODO_TOKEN")
    api = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
    s = requests.Session(); s.headers["Authorization"] = f"Bearer {token}"

    if a.deposition:
        r = s.get(f"{api}/deposit/depositions/{a.deposition}")
    else:
        r = s.post(f"{api}/deposit/depositions", json={})
    r.raise_for_status(); dep = r.json()
    print("draft", dep["id"], dep["links"]["html"])

    meta = json.load(open(a.metadata))
    meta.setdefault("prereserve_doi", True)
    r = s.put(f"{api}/deposit/depositions/{dep['id']}", json={"metadata": meta})
    if not r.ok:
        sys.exit(f"metadata rejected: {r.status_code} {r.text}")
    dep = r.json()
    print("reserved DOI", dep["metadata"].get("prereserve_doi", {}).get("doi"))

    existing = {f["filename"]: f for f in s.get(f"{api}/deposit/depositions/{dep['id']}/files").json()}
    bucket = dep["links"]["bucket"]
    for p in map(Path, a.files):
        if not p.is_file():
            continue
        size = p.stat().st_size
        if p.name in existing and existing[p.name]["filesize"] == size:
            print("skip (already uploaded)", p.name); continue
        print(f"upload {p.name} ({size / 1e9:.2f} GB)", flush=True)
        with open(p, "rb") as f:
            r = s.put(f"{bucket}/{p.name}", data=f)
        r.raise_for_status()
        if r.json().get("checksum", "").split(":")[-1] != md5(p):
            sys.exit(f"checksum mismatch for {p.name}: delete it in the draft and rerun")
    print("done; review and publish in the web interface:", dep["links"]["html"])


if __name__ == "__main__":
    main()
