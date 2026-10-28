"""Shared competition interface. Distributed to participants.

Model contract
--------------
Your submission is a GitHub repository with a ``model.py`` that defines ``class Model``, plus
the trained model files it loads (nothing is trained on the organizers' side):

    class Model:
        def __init__(self):
            # Load weights here. Runs once per run in an untimed setup phase
            # (subject to a setup timeout). The working directory is the
            # directory of your file, and that directory is on sys.path.
            # Use paths relative to your file, never absolute ones:
            #     self.net = load(Path(__file__).parent / "weights.pt")
            ...

        def predict(self, x: Sample) -> int:
            # Return a class id in 0..N_CLASSES-1, or ABSTAIN (-1).
            ...

* ``predict`` receives one ``Sample`` at a time, strictly sequentially, and is
  timed individually. Only the ``predict`` call is timed.
* Labels are never passed to the model, not even during warmup.
* Any other return value (bool, float, None, out-of-range int, ...) or an
  exception is recorded as an invalid output / exception and scored as a wrong answer.
* ``x.series`` is a read-only float32 array of shape (C, L) that may contain NaN.
  ``x.meta`` is a dict of str -> int | float | str.

Subclassing ``BaseModel`` is optional.
"""
from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ABSTAIN = -1
N_CLASSES = 8  # number of classes in the competition (ids 0..N_CLASSES-1)

META_PREFIX = "meta__"  # column prefix used for meta fields in exported files


@dataclass(frozen=True)
class Sample:
    series: np.ndarray  # float32, shape (C, L), read-only, may contain NaN
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        arr = np.asarray(self.series, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Sample.series must have shape (C, L), got {arr.shape}")
        arr = arr.view()  # read-only view; does not change the caller's array flags
        arr.setflags(write=False)
        meta = {}
        for k, v in dict(self.meta).items():
            if not isinstance(k, str):
                raise TypeError(f"meta keys must be str, got {k!r}")
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                raise TypeError(f"meta[{k!r}] must be int, float or str, got {type(v).__name__}")
            meta[k] = v
        object.__setattr__(self, "series", arr)
        object.__setattr__(self, "meta", meta)


class BaseModel:
    """Optional base class. Subclasses must implement ``predict``."""

    def __init__(self):
        pass

    def predict(self, x: Sample) -> int:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class Dataset:
    """Exported dataset. ``ds[j]`` returns the ``Sample`` for row j, in the same
    form the model receives at test time; ``ds.labels[j]`` is its label.
    ``series`` is the full (N, C, L) array for vectorized training."""

    series: np.ndarray
    labels: np.ndarray
    sample_index: np.ndarray
    meta: dict  # field name -> (N,) array

    def __len__(self) -> int:
        return len(self.series)

    def meta_at(self, j: int) -> dict:
        return {k: v[j].item() for k, v in self.meta.items()}

    def __getitem__(self, j: int) -> Sample:
        return Sample(self.series[j], self.meta_at(j))

    def __iter__(self):
        return (self[j] for j in range(len(self)))


def load_dataset(path, mmap: bool = True) -> Dataset:
    """Read an exported ``.npz`` or ``.parquet`` file.

    For ``.npz`` files written by the exporter (uncompressed), ``mmap=True`` maps
    the arrays from disk instead of loading them, so very large files work.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        arrays = _read_parquet(path)
    else:
        arrays = _read_npz(path, mmap)
    series = arrays.pop("series")
    if series.dtype != np.float32 or series.ndim != 3:
        raise ValueError(f"{path}: 'series' must be float32 (N, C, L), got {series.dtype} {series.shape}")
    if series.flags.writeable:
        series.setflags(write=False)
    meta = {k[len(META_PREFIX):]: v for k, v in arrays.items() if k.startswith(META_PREFIX)}
    n = len(series)
    labels = arrays.get("labels", np.full(n, -1, dtype=np.int64))
    sample_index = arrays.get("sample_index", np.arange(n, dtype=np.int64))
    return Dataset(series=series, labels=np.asarray(labels), sample_index=np.asarray(sample_index), meta=meta)


def _read_npz(path: Path, mmap: bool) -> dict:
    out = {}
    with zipfile.ZipFile(path) as zf, open(path, "rb") as f:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            name = info.filename[:-4]
            arr = _mmap_member(path, f, info) if mmap and info.compress_type == zipfile.ZIP_STORED else None
            if arr is None:
                with zf.open(info) as m:
                    arr = np.lib.format.read_array(m, allow_pickle=False)
            out[name] = arr
    return out


def _mmap_member(path: Path, f, info: zipfile.ZipInfo):
    f.seek(info.header_offset)
    local = f.read(30)
    n_name, n_extra = struct.unpack("<HH", local[26:30])
    f.seek(info.header_offset + 30 + n_name + n_extra)
    version = np.lib.format.read_magic(f)
    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = read_header(f)
    if dtype.hasobject or 0 in shape or shape == ():
        return None
    return np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape, order="F" if fortran else "C")


def _read_parquet(path: Path) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading .parquet files requires pyarrow") from e
    table = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    c, l = int(md["n_channels"]), int(md["length"])
    out = {}
    for name in table.column_names:
        col = table.column(name).combine_chunks()
        if name == "series":
            out[name] = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, c, l)
        else:
            out[name] = col.to_numpy(zero_copy_only=False)
            if out[name].dtype == object:
                out[name] = out[name].astype(str)
    return out
