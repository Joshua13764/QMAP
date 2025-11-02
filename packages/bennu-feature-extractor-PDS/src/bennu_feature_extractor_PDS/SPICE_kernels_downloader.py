# src/bennu_feature_extractor_PDS/SPICE_kernels_downloader.py
from __future__ import annotations

import multiprocessing
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.utils.FS_environment_factory import \
    FSEnvironmentFactory
from bennu_feature_extractor.step_base import StepBase
from joblib import delayed
from requests.adapters import HTTPAdapter
from tqdm_joblib import ParallelPbar
from urllib3.util.retry import Retry


# ---------- Domain models ----------
@dataclass(frozen=True)
class KernelItem:
    url: str
    subdir: str
    filename: str
    out_path: Path
    group: str          # ck/spk/pck/...
    remote_size: int    # 0 if unknown


@dataclass(frozen=True)
class PlanSummary:
    items: List[KernelItem]
    total_bytes: int
    skipped_bytes: int
    bytes_by_group: Dict[str, int]


# ---------- 1) MK parsing (Single Responsibility) ----------
_ALLOWED_TOPS = (
    "ck/",
    "spk/",
    "pck/",
    "lsk/",
    "sclk/",
    "fk/",
    "ik/",
    "dsk/",
    "mk/")


class MKParser:
    @staticmethod
    def extract_kernel_rels(mk_text: str) -> List[str]:
        """
        Strict, line-based parse of KERNELS_TO_LOAD += ( ... ) blocks.
        Keeps only quoted entries that begin with allowed top-level dirs.
        """
        rels: List[str] = []
        in_block = False
        for raw in mk_text.splitlines():
            line = raw.strip()
            if not in_block and re.search(
                    r"\bKERNELS_TO_LOAD\b", line, flags=re.I):
                in_block = True
            if in_block:
                for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', line):
                    p = (m.group(1) or m.group(2)).strip()
                    p = p.replace("\\", "/")
                    p = re.sub(r"^\$KERNELS/", "", p, flags=re.I)
                    p = re.sub(r"^(\./)+", "", p)
                    p = re.sub(r"^(\.\./)+", "", p)
                    if (
                        p.lower().startswith(_ALLOWED_TOPS)
                        and "spice_kernels" not in p.lower()
                        and "`" not in p
                    ):
                        rels.append(p)
                if ")" in line:
                    in_block = False
        # de-dup, keep order
        out, seen = [], set()
        for p in rels:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out


# ---------- 2) URL mapping (Single Responsibility) ----------
class URLMapper:
    KEY = "/spice_kernels/"

    @staticmethod
    def mk_root(mk_url: str) -> str:
        low = mk_url.lower()
        i = low.rfind(URLMapper.KEY)
        if i == -1:
            raise ValueError(
                "MK URL does not contain /spice_kernels/: " + mk_url)
        return mk_url[: i + len(URLMapper.KEY)]

    @staticmethod
    def rel_dir(file_url: str) -> str:
        low = file_url.lower()
        i = low.rfind(URLMapper.KEY)
        if i == -1:
            return ""
        tail = file_url[i + len(URLMapper.KEY):]
        slash = tail.rfind("/")
        return "" if slash == -1 else tail[:slash]

    @staticmethod
    def safe_join(root: str, rel: str) -> str:
        """Avoid stripping root; avoid duplicating 'spice_kernels'."""
        url = root.rstrip("/") + "/" + rel.lstrip("/")
        double = "spice_kernels/spice_kernels"
        if double in url.lower():
            parts = url.split("/spice_kernels/")
            url = parts[0] + "/spice_kernels/" + "/".join(parts[2:])
        return url


# ---------- 3) Size probing (Single Responsibility) ----------
class SizeProbe:
    def __init__(self, session: requests.Session, timeout: int):
        self._s = session
        self._t = timeout

    def content_length(self, url: str) -> int:
        # Try HEAD
        try:
            h = self._s.head(url, allow_redirects=True, timeout=self._t)
            v = h.headers.get("Content-Length")
            if v and v.isdigit():
                sz = int(v)
                if sz > 0:
                    return sz
        except Exception:
            pass
        # Fallback: 1-byte range GET → Content-Range: bytes 0-0/TOTAL
        try:
            r = self._s.get(
                url,
                headers={
                    "Range": "bytes=0-0"},
                timeout=self._t)
            cr = r.headers.get("Content-Range", "")
            m = re.search(r"/(\d+)$", cr)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return 0


# ---------- 4) Download planning (Single Responsibility) ----------
class DownloadPlanner:
    def __init__(self, download_path: Path, overwrite: bool,
                 size_probe: SizeProbe):
        self.root = download_path
        self.overwrite = overwrite
        self.sizer = size_probe

    def plan(self, urls: Iterable[str]) -> PlanSummary:
        items: List[KernelItem] = []
        skipped = 0
        by_group: Dict[str, int] = {}
        total = 0

        for u in urls:
            subdir = URLMapper.rel_dir(u)
            fname = (u.rsplit("/", 1)[-1]) or "download"
            out_path = (self.root / subdir / fname)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            size = self.sizer.content_length(u)
            exists = out_path.exists() and not self.overwrite
            if exists:
                if size > 0:
                    skipped += size
                continue

            group = (subdir.split("/", 1)[0] if subdir else "other") or "other"
            by_group[group] = by_group.get(group, 0) + max(size, 0)
            total += max(size, 0)

            items.append(
                KernelItem(
                    url=u, subdir=subdir, filename=fname,
                    out_path=out_path, group=group, remote_size=size
                )
            )

        return PlanSummary(items=items, total_bytes=total,
                           skipped_bytes=skipped, bytes_by_group=by_group)


# ---------- 5) Downloader (Single Responsibility) ----------
class FileDownloader:
    def __init__(self, session: requests.Session, chunk_size: int,
                 timeout: int, overwrite: bool, logger):
        self._s = session
        self._chunk = chunk_size
        self._t = timeout
        self._overwrite = overwrite
        self._log = logger

    def __call__(self, item: KernelItem) -> Path:
        out_path = item.out_path
        if out_path.exists() and not self._overwrite:
            return out_path

        tmp = out_path.with_suffix(out_path.suffix + ".part")
        try:
            with self._s.get(item.url, stream=True, timeout=self._t) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=self._chunk):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
        return out_path


# ---------- 6) Orchestrator Step (Facade; composes the above) ----------
@dataclass
class SPICEKernelGrabber(StepBase):
    """
    SPICE mirror step following SOLID:
      - MKParser: parse MKs → rel paths
      - URLMapper: map to absolute URLs and local destinations
      - SizeProbe: estimate Content-Length
      - DownloadPlanner: decide what to fetch & summarize sizes
      - FileDownloader: fetch one file
      - Orchestrator (this class): compose with joblib + tqdm_joblib progress
    """
    DownloadPath: str
    MkUrls: List[str]
    ExtraUrls: Optional[List[str]] = None
    TimeoutSeconds: int = 60
    Overwrite: bool = False
    Workers: int = field(
        default_factory=lambda: max(
            1, multiprocessing.cpu_count() - 2))
    ChunkSize: int = 16 * 1024 * 1024  # 16 MiB

    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self):
        os.makedirs(self.DownloadPath, exist_ok=True)
        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        self._session = s

    # ---- StepBase API ----
    def get_hash(self) -> int:
        return hash((
            self.DownloadPath, tuple(self.MkUrls), tuple(self.ExtraUrls or []),
            self.TimeoutSeconds, self.Overwrite, self.Workers, self.ChunkSize
        ))

    def run(self, env: FSEnvironment) -> FSEnvironment:
        download_root = Path(self.DownloadPath)

        # 1) Collect absolute URLs from MKs
        urls: List[str] = []
        for mk_url in self.MkUrls:
            # ensure MK itself is stored under mk/
            mk_dir = download_root / "mk"
            mk_dir.mkdir(parents=True, exist_ok=True)
            mk_file = mk_dir / (mk_url.rsplit("/", 1)[-1] or "metakernel.tm")
            if not mk_file.exists() or self.Overwrite:
                self.logger.info(f"Downloading MK: {mk_url}")
                self._fetch_to(mk_url, mk_file)

            mk_text = mk_file.read_text(encoding="utf-8", errors="ignore")
            rels = MKParser.extract_kernel_rels(mk_text)
            root = URLMapper.mk_root(mk_url)
            self.logger.info(f"Found {len(rels)} kernels in MK: {mk_url}")
            urls.extend(
                URLMapper.safe_join(
                    root, r.replace(
                        "\\", "/")) for r in rels)

        if self.ExtraUrls:
            urls.extend(self.ExtraUrls)

        # 2) Plan (skip existing; estimate sizes)
        planner = DownloadPlanner(
            download_root, self.Overwrite, SizeProbe(
                self._session, self.TimeoutSeconds))
        plan = planner.plan(urls)

        self.logger.info(
            f"Plan: {len(plan.items)} files; ~{plan.total_bytes /
                                               (1 << 30):.2f} GB "
            f"(already on disk: {plan.skipped_bytes / (1 << 30):.2f} GB)"
        )
        for g, b in sorted(plan.bytes_by_group.items()):
            self.logger.info(f"  {g:>4}: {b / (1 << 30):.2f} GB")

        # 3) Download with a single global progress bar (ParallelPbar)

        if plan.items:
            desc = f"Downloading SPICE ({plan.total_bytes /
                                         (1 << 30):.2f} GB, {len(plan.items)} files)"
            downloader = FileDownloader(
                self._session,
                self.ChunkSize,
                self.TimeoutSeconds,
                self.Overwrite,
                self.logger)

            try:
                ParallelPbar(desc)(n_jobs=self.Workers)(
                    delayed(downloader)(item) for item in plan.items
                )
            except Exception as e:
                print("Got an error:", e)
                self.run(env)  # Re-run usually fixes download errors

        else:
            self.logger.info(
                "Nothing to download — everything already present.")

        # 4) Return environment
        return FSEnvironment.merge([
            env,
            FSEnvironmentFactory.from_folder(download_root, extensions=None),
        ])

    # ---- small helper ----
    def _fetch_to(self, url: str, dst: Path):
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            with self._session.get(url, stream=True, timeout=self.TimeoutSeconds) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=self.ChunkSize):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dst)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass


# Backwards-compat alias
SPICEFromMK = SPICEKernelGrabber
__all__ = ["SPICEKernelGrabber", "SPICEFromMK"]
