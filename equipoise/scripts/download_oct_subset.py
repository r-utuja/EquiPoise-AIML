"""Download a balanced subset of the Kermany retinal OCT dataset from Kaggle (M10).

The Kaggle archive is one ~11.6 GB zip in cloud storage that accepts HTTP range requests.
Instead of downloading all of it, this script:
1. asks the Kaggle API for the archive's (signed) download URL;
2. reads only the zip's central directory (its index, at the end of the file) through a
   seekable range-request file object, which gives every file name, offset and size;
3. draws a seeded balanced subset from the official train split
   (`vision.per_class_train` per class) plus the whole official test split;
4. fetches just those images by byte range (in parallel, resumable) into
   `data/external/kermany_oct/<split>/<class>/`, and writes `manifest.csv` there.

These images are NOT linked to Protocol T patients: M10 is a screening front-end demo.
Nothing under data/ is committed.

Credentials: the Kaggle token via KAGGLE_API_TOKEN or ~/.kaggle/access_token (or a legacy
~/.kaggle/kaggle.json). The script never prints the token or the signed URL.

Usage:
    python scripts/download_oct_subset.py
"""

from __future__ import annotations

import io
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from equipoise.config import load_config

LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")   # zip local file header (30 bytes)


class SignedUrl:
    """The archive URL from the Kaggle API; refreshed when it expires (HTTP 403)."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self._lock = threading.Lock()
        self.url = self._fetch()

    def _fetch(self) -> str:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

        api = KaggleApi()
        api.authenticate()
        owner, slug = self.dataset.split("/")
        with api.build_kaggle_client() as kaggle:
            req = ApiDownloadDatasetRequest()
            req.owner_slug, req.dataset_slug = owner, slug
            resp = kaggle.datasets.dataset_api_client.download_dataset(req)
        url = resp.url
        resp.close()          # we only needed the redirect target, not the 11.6 GB body
        return url

    def refresh(self) -> None:
        with self._lock:
            self.url = self._fetch()


def get_range(src: SignedUrl, start: int, end: int, session: requests.Session,
              retries: int = 5) -> bytes:
    """Bytes [start, end] (inclusive) of the archive, with retries and URL refresh."""
    for attempt in range(retries):
        try:
            r = session.get(src.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code == 403:
                src.refresh()
                raise RuntimeError("signed URL expired; refreshed")
            if r.status_code != 206:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


class RangeFile(io.RawIOBase):
    """Read-only, seekable file object over HTTP range requests (enough for zipfile)."""

    def __init__(self, src: SignedUrl, size: int) -> None:
        self.src, self.size, self.pos = src, size, 0
        self.session = requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = get_range(self.src, self.pos, end, self.session)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def archive_size(src: SignedUrl) -> int:
    r = requests.get(src.url, headers={"Range": "bytes=0-0"}, timeout=60)
    return int(r.headers["Content-Range"].split("/")[1])


def parse(infos: list[zipfile.ZipInfo], prefix: str, classes: list[str]) -> pd.DataFrame:
    """Keep `<prefix>{train,test}/<CLASS>/<CLASS>-<patient>-<n>.jpeg` and parse the parts."""
    rows = []
    for zi in infos:
        if not zi.filename.startswith(prefix) or zi.is_dir():
            continue
        parts = zi.filename[len(prefix):].split("/")
        if len(parts) != 3 or parts[0] not in ("train", "test") or parts[1] not in classes:
            continue
        if not parts[2].lower().endswith((".jpeg", ".jpg", ".png")):
            continue
        rows.append({"name": zi.filename, "split": parts[0], "cls": parts[1],
                     "file": parts[2], "patient_id": parts[2].split("-")[1],
                     "offset": zi.header_offset, "csize": zi.compress_size,
                     "method": zi.compress_type, "bytes": zi.file_size})
    return pd.DataFrame(rows)


def choose(files: pd.DataFrame, per_class: int, use_test: bool, seed: int) -> pd.DataFrame:
    """Balanced train subset + official test split. The official split is NOT patient-
    disjoint (hundreds of patients have scans in both), so training images are drawn only
    from patients who have no scan in the test split; otherwise test scores would be
    inflated by near-duplicate scans of the same eye."""
    rng = np.random.default_rng(seed)
    test_patients = set(files.loc[files.split == "test", "patient_id"]) if use_test else set()
    train = files[(files.split == "train") & ~files.patient_id.isin(test_patients)]
    parts = []
    for _cls, g in train.sort_values("name").groupby("cls"):
        k = min(per_class, len(g))
        parts.append(g.iloc[np.sort(rng.choice(len(g), k, replace=False))])
    if use_test:
        parts.append(files[files.split == "test"].sort_values("name"))
    return pd.concat(parts, ignore_index=True)


_LOCAL = threading.local()


def fetch_member(src: SignedUrl, row, dest: Path) -> str:
    """Read one member by byte range: local header + compressed data, then inflate."""
    if dest.is_file() and dest.stat().st_size == row.bytes:
        return "cached"
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    try:
        head = get_range(src, row.offset, row.offset + LOCAL_HEADER.size - 1, _LOCAL.session)
        sig, *_rest, name_len, extra_len = LOCAL_HEADER.unpack(head)
        if sig != b"PK\x03\x04":
            return "failed: bad local header"
        start = row.offset + LOCAL_HEADER.size + name_len + extra_len
        data = get_range(src, start, start + row.csize - 1, _LOCAL.session)
        if row.method == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        elif row.method != zipfile.ZIP_STORED:
            return f"failed: compression method {row.method}"
        if len(data) != row.bytes:
            return "failed: size mismatch"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return "ok"
    except Exception as e:
        return f"failed: {type(e).__name__}: {e}"


def main() -> None:
    cfg = load_config()
    v = cfg.raw["vision"]
    root = cfg.paths.root / v["raw_dir"]
    index_cache = root / "_zip_index.csv"
    src = SignedUrl(v["kaggle_dataset"])
    if index_cache.is_file():
        files = pd.read_csv(index_cache, dtype={"patient_id": str})
    else:
        size = archive_size(src)
        print(f"archive: {size / 1e9:.1f} GB; reading its central directory only ...",
              flush=True)
        with zipfile.ZipFile(io.BufferedReader(RangeFile(src, size), 1 << 20)) as zf:
            infos = zf.infolist()
        print(f"  {len(infos)} entries in the archive", flush=True)
        files = parse(infos, v["kaggle_prefix"], list(v["classes"]))
        root.mkdir(parents=True, exist_ok=True)
        files.to_csv(index_cache, index=False)
    if files.empty:
        raise SystemExit(f"no files under prefix {v['kaggle_prefix']!r}")
    print("available:", files.groupby(["split", "cls"]).size().to_dict(), flush=True)
    pick = choose(files, int(v["per_class_train"]), bool(v["use_official_test"]), cfg.seed)
    pick["path"] = [str(Path(s) / c / f) for s, c, f in zip(pick.split, pick.cls, pick.file,
                                                            strict=True)]
    print(f"subset: {len(pick)} images ({pick.groupby(['split', 'cls']).size().to_dict()}), "
          f"{pick.csize.sum() / 1e6:.0f} MB to fetch", flush=True)
    status = {}
    with ThreadPoolExecutor(int(v["download_workers"])) as pool:
        futs = {pool.submit(fetch_member, src, row, root / row.path): row.path
                for row in pick.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            status[futs[fut]] = fut.result()
            if i % 500 == 0 or i == len(futs):
                bad = sum(not s.startswith(("ok", "cached")) for s in status.values())
                print(f"  {i}/{len(futs)} done, {bad} failed", flush=True)
    pick["status"] = pick.path.map(status)
    ok = pick.status.str.startswith(("ok", "cached"))
    pick.loc[ok, ["split", "cls", "patient_id", "path", "name"]].to_csv(
        root / "manifest.csv", index=False)
    print(f"manifest: {int(ok.sum())} images -> {root / 'manifest.csv'}")
    if (~ok).any():
        print(f"{int((~ok).sum())} failed; rerun to retry. First: {pick.status[~ok].iloc[0]}")


if __name__ == "__main__":
    main()
