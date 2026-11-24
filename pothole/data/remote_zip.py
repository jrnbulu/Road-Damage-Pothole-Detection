"""Download one file out of a huge remote ZIP without downloading all of it.

RDD2022 on Figshare is a single 13 GB zip that holds one zip per country
(RDD2022/India.zip is ~527 MB). A ZIP keeps its index at the end, so we read
the index with HTTP Range requests, find where India.zip sits, and download
only those bytes, in resumable chunks, with a CRC check at the end.
"""
import io
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

USER_AGENT = "pothole-capstone/0.1 (python-urllib)"
CHUNK = 16 * 1024 * 1024


def _get_range(url: str, start: int, end: int, retries: int = 6) -> bytes:
    """GET bytes [start, end] inclusive. The URL is re-requested every time because
    Figshare redirects to a signed link that expires within seconds."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status != 206:
                    raise IOError(f"Server ignored the Range header (HTTP {r.status})")
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"Short read: {len(data)} of {end - start + 1} bytes")
            return data
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"\n  retry in {wait}s ({e})")
            time.sleep(wait)


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP with a read-ahead buffer (enough for zipfile to read the index)."""

    def __init__(self, url: str, block: int = 1024 * 1024, size: int | None = None):
        self.url, self.block, self.pos = url, block, 0
        self.size = size if size is not None else _remote_size(url)
        self._cache_start, self._cache = -1, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cs, ce = self._cache_start, self._cache_start + len(self._cache)
        if not (cs <= self.pos and self.pos + n <= ce):
            end = min(self.size, self.pos + max(n, self.block)) - 1
            self._cache_start, self._cache = self.pos, _get_range(self.url, self.pos, end)
            cs = self.pos
        out = self._cache[self.pos - cs: self.pos - cs + n]
        self.pos += len(out)
        return out


def download_member(url: str, member: str, out_path: Path) -> Path:
    """Save one *stored* (uncompressed) member of a remote ZIP to out_path. Resumable."""
    remote = HttpRangeFile(url)
    with zipfile.ZipFile(remote) as zf:
        info = zf.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        raise IOError(f"{member} is compressed; this helper only copies stored members")

    # The local header has its own name/extra lengths; read them to find where the data starts.
    remote.seek(info.header_offset)
    sig, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", remote.read(30))
    if sig != 0x04034B50:
        raise IOError("Bad local file header")
    data_start = info.header_offset + 30 + name_len + extra_len
    total = info.compress_size

    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(out_path.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    if done > total:
        part.unlink()
        done = 0
    print(f"Downloading {member} ({total / 1e6:.0f} MB) out of the {remote.size / 1e9:.1f} GB archive")
    with part.open("ab") as f:
        while done < total:
            a = data_start + done
            b = data_start + min(total, done + CHUNK) - 1
            f.write(_get_range(url, a, b))
            done = f.tell()
            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()

    crc = 0
    with part.open("rb") as f:
        while chunk := f.read(CHUNK):
            crc = zlib.crc32(chunk, crc)
    if crc & 0xFFFFFFFF != info.CRC:
        part.unlink()
        raise IOError("CRC check failed; the partial file was deleted, run again")
    part.replace(out_path)
    print(f"Verified and saved: {out_path}")
    return out_path
