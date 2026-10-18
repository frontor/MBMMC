#!/usr/bin/env python3
"""
Verified conversion of GSE178169 hg19 single-CpG methylation-ratio bigWigs
into an hg38 multi-sample beta matrix.

Why this script is stricter than a point-wise pyliftover workflow
-----------------------------------------------------------------
Treat GSE178169 as an example. Its processed files are hg19 bigWigs containing a methylation ratio
at single-CpG resolution. A CpG must therefore be lifted as the complete
2-bp interval [C,G), not merely as one coordinate. Lifting only the C base
can create an off-by-one target coordinate in reverse-orientation chain
segments and does not prove that the target locus is still a CpG.

This script:
  1. checks whether each bigWig interval is anchored on the C or G base by
     comparison with an indexed hg19 FASTA;
  2. validates source CpG sequence;
  3. writes every source CpG as a 2-bp BED interval;
  4. uses the UCSC liftOver executable with -multiple and -minMatch=1.0;
  5. retains only uniquely mapped 2-bp intervals whose hg38 sequence is CG;
  6. conservatively drops hg19->hg38 target collisions by default;
  7. supports union, intersection, or minimum-sample-fraction feature sets;
  8. writes detailed JSON QC statistics.

Important limitation
--------------------
The GEO bigWigs contain methylation ratios, not methylated/unmethylated read
counts or coverage. A minimum-depth filter cannot be reconstructed from these
files. Missing CpGs must remain missing and must never be converted to beta=0.

Dependencies
------------
  mamba create -n bw_trans -c conda-forge -c bioconda \
      python=3.11 pybigwig pysam ucsc-liftover
  mamba activate bw_trans

Prepare indexed FASTAs:
  samtools faidx hg19.fa
  samtools faidx hg38.fa

Recommended command
-------------------
  python bw_to_hg38_beta_matrix_verified.py \
      --bw_dir ./GSE178169_RAW \
      --bw_glob '*.bw' \
      --chain ./hg19ToHg38.over.chain.gz \
      --source_fasta ./hg19.fa \
      --target_fasta ./hg38.fa \
      --outdir ./GSE178169_hg38_verified \
      --beta_scale raw_beta \
      --input_anchor auto \
      --feature_mode min_fraction \
      --min_sample_fraction 0.80 \
      --matrix ./GSE178169_hg38_beta_matrix.tsv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


STANDARD_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


@dataclass
class SampleSpec:
    sample: str
    beta_bw: str


@dataclass
class SampleStats:
    sample: str
    beta_bw: str
    detected_anchor: str = ""
    anchor_sites_checked: int = 0
    anchor_c_at_start: int = 0
    anchor_g_at_start: int = 0
    intervals_seen: int = 0
    interval_widths: Dict[str, int] = None
    finite_values: int = 0
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    skipped_invalid_value: int = 0
    skipped_wide_interval: int = 0
    skipped_invalid_source_cpg: int = 0
    source_cpg_written_to_bed: int = 0
    liftover_unmapped: int = 0
    liftover_malformed_output: int = 0
    skipped_nonstandard_target: int = 0
    skipped_target_not_2bp: int = 0
    skipped_invalid_target_cpg: int = 0
    skipped_multi_target: int = 0
    unique_lifted_records: int = 0
    collision_groups: int = 0
    collision_records: int = 0
    final_records: int = 0

    def __post_init__(self) -> None:
        if self.interval_widths is None:
            self.interval_widths = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a rigorously validated hg38 beta matrix from GSE178169 "
            "hg19 single-CpG methylation-ratio bigWigs."
        )
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--samplesheet", help="CSV/TSV containing columns sample,beta_bw.")
    src.add_argument("--bw_dir", help="Directory containing GSE178169 .bw files.")
    p.add_argument("--bw_glob", default="*.bw", help="Glob used with --bw_dir.")
    p.add_argument("--chain", required=True, help="UCSC hg19ToHg38.over.chain.gz.")
    p.add_argument("--source_fasta", required=True, help="Indexed hg19 FASTA.")
    p.add_argument("--target_fasta", required=True, help="Indexed hg38 FASTA.")
    p.add_argument("--liftover_bin", default="liftOver", help="UCSC liftOver executable.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--matrix", default="", help="Default: <outdir>/hg38_beta_matrix.tsv.gz")
    p.add_argument("--features", default="", help="Default: <outdir>/hg38_features.tsv.gz")
    p.add_argument("--report_json", default="", help="Default: <outdir>/processing_report.json")
    p.add_argument("--missing", default="NA")
    p.add_argument("--digits", type=int, default=6)
    p.add_argument(
        "--beta_scale",
        choices=["raw_beta", "percent"],
        default="raw_beta",
        help="GSE178169 is expected to use raw_beta (0..1).",
    )
    p.add_argument(
        "--input_anchor",
        choices=["auto", "c_at_start", "g_at_start"],
        default="auto",
        help=(
            "Meaning of bigWig interval start. auto compares positions with hg19 FASTA; "
            "g_at_start also catches a common 1-based-to-bedGraph off-by-one error."
        ),
    )
    p.add_argument(
        "--anchor_check_sites",
        type=int,
        default=20000,
        help="Number of sites sampled per file for anchor detection.",
    )
    p.add_argument(
        "--anchor_min_fraction",
        type=float,
        default=0.90,
        help="Minimum sampled CpG fraction required for automatic anchor selection.",
    )
    p.add_argument(
        "--max_interval_width",
        type=int,
        default=2,
        help="Intervals wider than this are rejected; single-CpG tracks should be width 1 or 2.",
    )
    p.add_argument(
        "--collision_policy",
        choices=["drop", "mean", "first"],
        default="drop",
        help="Policy when distinct hg19 CpGs map to the same hg38 CpG.",
    )
    p.add_argument(
        "--feature_mode",
        choices=["union", "intersection", "min_fraction"],
        default="min_fraction",
    )
    p.add_argument(
        "--min_sample_fraction",
        type=float,
        default=0.80,
        help="Used when --feature_mode min_fraction.",
    )
    p.add_argument(
        "--chroms",
        default=",".join(STANDARD_CHROMS),
        help="Comma-separated target chromosomes retained.",
    )
    p.add_argument(
        "--bw_window",
        type=int,
        default=5_000_000,
        help="Chunk size for reading bigWig intervals.",
    )
    p.add_argument(
        "--fasta_chunk",
        type=int,
        default=1_000_000,
        help="FASTA sequence cache chunk size.",
    )
    p.add_argument("--keep_work", action="store_true")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.digits < 0:
        raise SystemExit("--digits must be non-negative.")
    if args.anchor_check_sites < 1:
        raise SystemExit("--anchor_check_sites must be positive.")
    if not (0.5 <= args.anchor_min_fraction <= 1.0):
        raise SystemExit("--anchor_min_fraction must be between 0.5 and 1.")
    if args.max_interval_width < 1:
        raise SystemExit("--max_interval_width must be positive.")
    if not (0.0 < args.min_sample_fraction <= 1.0):
        raise SystemExit("--min_sample_fraction must be in (0,1].")
    if args.bw_window < 1 or args.fasta_chunk < 1000:
        raise SystemExit("--bw_window must be positive and --fasta_chunk must be >=1000.")
    if any(ch in args.missing for ch in ("\t", "\n", "\r")):
        raise SystemExit("--missing must not contain tab/newline characters.")


def open_text(path: Path | str, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="" if "t" in mode else None)
    return open(path, mode, newline="" if "t" in mode else None)


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom)
    if chrom.lower().startswith("chr"):
        core = chrom[3:]
    else:
        core = chrom
    if core in {"M", "MT", "m", "mt"}:
        return "chrM"
    if core.lower() == "x":
        return "chrX"
    if core.lower() == "y":
        return "chrY"
    return f"chr{core}"


def chrom_order(chrom: str) -> int:
    c = normalize_chrom(chrom)[3:]
    if c.isdigit():
        return int(c)
    if c == "X":
        return 23
    if c == "Y":
        return 24
    if c in {"M", "MT"}:
        return 25
    return 1000


def sample_file_id(sample: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample).strip("._") or "sample"
    digest = hashlib.sha1(sample.encode("utf-8")).hexdigest()[:10]
    return f"{safe}.{digest}"


def infer_delimiter(path: Path) -> str:
    with open_text(path, "rt") as fh:
        line = fh.readline()
    return "\t" if line.count("\t") > line.count(",") else ","


def resolve_relative(value: str, base: Path) -> str:
    p = Path(value.strip())
    return str(p if p.is_absolute() else base / p)


def read_samples(args: argparse.Namespace) -> List[SampleSpec]:
    out: List[SampleSpec] = []
    if args.samplesheet:
        path = Path(args.samplesheet)
        delim = infer_delimiter(path)
        with open_text(path, "rt") as fh:
            rdr = csv.DictReader(fh, delimiter=delim)
            if not rdr.fieldnames or not {"sample", "beta_bw"}.issubset(rdr.fieldnames):
                raise SystemExit("samplesheet must contain sample and beta_bw columns.")
            for row in rdr:
                sample = (row.get("sample") or "").strip()
                bw = (row.get("beta_bw") or "").strip()
                if not sample or not bw:
                    continue
                if any(ch in sample for ch in ("\t", "\n", "\r")):
                    raise SystemExit(f"Invalid sample name: {sample!r}")
                out.append(SampleSpec(sample, resolve_relative(bw, path.parent)))
    else:
        paths = sorted(Path(args.bw_dir).glob(args.bw_glob))
        if not paths:
            raise SystemExit(f"No files matched {args.bw_dir}/{args.bw_glob}")
        for p in paths:
            name = p.name
            for suffix in (".bigWig", ".bigwig", ".bw"):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            out.append(SampleSpec(name, str(p)))

    seen = set()
    for s in out:
        if s.sample in seen:
            raise SystemExit(f"Duplicate sample name: {s.sample}")
        seen.add(s.sample)
        if not Path(s.beta_bw).is_file():
            raise SystemExit(f"Missing bigWig for {s.sample}: {s.beta_bw}")
    if not out:
        raise SystemExit("No samples found.")
    return out


class FastaChunkReader:
    """Small LRU cache for sequential 2-bp FASTA queries."""

    def __init__(self, path: str, chunk_size: int, max_chunks: int = 8):
        try:
            import pysam
        except Exception as e:
            raise SystemExit("Missing pysam. Install it from bioconda/conda-forge.") from e
        self.fa = pysam.FastaFile(path)
        self.refs = set(self.fa.references)
        self.lengths = dict(zip(self.fa.references, self.fa.lengths))
        self.chunk_size = int(chunk_size)
        self.max_chunks = int(max_chunks)
        self.cache: OrderedDict[Tuple[str, int], str] = OrderedDict()
        self.alias_cache: Dict[str, str] = {}

    def close(self) -> None:
        self.fa.close()

    def resolve(self, chrom: str) -> str:
        if chrom in self.alias_cache:
            return self.alias_cache[chrom]
        n = normalize_chrom(chrom)
        candidates = [chrom, n, n[3:] if n.startswith("chr") else f"chr{n}"]
        for c in candidates:
            if c in self.refs:
                self.alias_cache[chrom] = c
                return c
        raise KeyError(f"Chromosome {chrom!r} not found in FASTA.")

    def fetch(self, chrom: str, start: int, end: int) -> str:
        if start < 0 or end <= start:
            return ""
        ref = self.resolve(chrom)
        if end > self.lengths[ref]:
            return ""
        chunk0 = (start // self.chunk_size) * self.chunk_size
        # Two-base requests can very rarely cross a chunk boundary.
        if end > chunk0 + self.chunk_size:
            return self.fa.fetch(ref, start, end).upper()
        key = (ref, chunk0)
        if key not in self.cache:
            chunk_end = min(chunk0 + self.chunk_size, self.lengths[ref])
            self.cache[key] = self.fa.fetch(ref, chunk0, chunk_end).upper()
            self.cache.move_to_end(key)
            while len(self.cache) > self.max_chunks:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        seq = self.cache[key]
        return seq[start - chunk0:end - chunk0]


def iter_bw_intervals(bw, chrom: str, window: int) -> Iterator[Tuple[int, int, float]]:
    chrom_len = int(bw.chroms(chrom))
    for w0 in range(0, chrom_len, window):
        w1 = min(w0 + window, chrom_len)
        intervals = bw.intervals(chrom, w0, w1)
        if not intervals:
            continue
        for start, end, value in intervals:
            # An interval overlapping a window boundary can be returned twice.
            if int(start) < w0:
                continue
            yield int(start), int(end), value


def normalize_beta(value, scale: str) -> float:
    try:
        v = float(value)
    except Exception:
        return math.nan
    if not math.isfinite(v):
        return math.nan
    if scale == "percent":
        v /= 100.0
    if v < -1e-6 or v > 1.0 + 1e-6:
        return math.nan
    return min(1.0, max(0.0, v))


def inspect_and_choose_anchor(
    spec: SampleSpec,
    args: argparse.Namespace,
    source_fa: FastaChunkReader,
    keep_chroms: set[str],
    stats: SampleStats,
) -> str:
    import pyBigWig

    bw = pyBigWig.open(spec.beta_bw)
    widths = Counter()
    checked = c_at = g_at = 0
    try:
        for bw_chrom in bw.chroms().keys():
            nchrom = normalize_chrom(bw_chrom)
            if nchrom not in keep_chroms:
                continue
            for start, end, value in iter_bw_intervals(bw, bw_chrom, args.bw_window):
                widths[end - start] += 1
                beta = normalize_beta(value, args.beta_scale)
                if not math.isfinite(beta):
                    continue
                c_seq = source_fa.fetch(nchrom, start, start + 2)
                g_seq = source_fa.fetch(nchrom, start - 1, start + 1) if start > 0 else ""
                checked += 1
                c_at += int(c_seq == "CG")
                g_at += int(g_seq == "CG")
                if checked >= args.anchor_check_sites:
                    break
            if checked >= args.anchor_check_sites:
                break
    finally:
        bw.close()

    stats.anchor_sites_checked = checked
    stats.anchor_c_at_start = c_at
    stats.anchor_g_at_start = g_at
    stats.interval_widths = {str(k): int(v) for k, v in sorted(widths.items())}

    if checked == 0:
        raise SystemExit(f"{spec.sample}: no finite standard-chromosome values found.")

    if args.input_anchor != "auto":
        anchor = args.input_anchor
    else:
        c_frac = c_at / checked
        g_frac = g_at / checked
        threshold = args.anchor_min_fraction
        if c_frac >= threshold and c_frac > g_frac:
            anchor = "c_at_start"
        elif g_frac >= threshold and g_frac > c_frac:
            anchor = "g_at_start"
        else:
            raise SystemExit(
                f"{spec.sample}: cannot determine CpG anchor. "
                f"C-at-start={c_at}/{checked} ({c_frac:.3%}), "
                f"G-at-start={g_at}/{checked} ({g_frac:.3%}). "
                "Inspect the bigWig/FASTA assembly or set --input_anchor explicitly."
            )
    stats.detected_anchor = anchor
    return anchor


def work_paths(outdir: Path, sample: str) -> Dict[str, Path]:
    sid = sample_file_id(sample)
    wd = outdir / "intermediate" / sid
    return {
        "dir": wd,
        "source_bed": wd / "source_cpg.bed",
        "mapped_bed": wd / "mapped.multiple.bed",
        "unmapped_bed": wd / "unmapped.bed",
        "mapped_id": wd / "mapped.by_id.tsv",
        "mapped_id_sorted": wd / "mapped.by_id.sorted.tsv",
        "target_raw": wd / "target.raw.tsv",
        "target_sorted": wd / "target.raw.sorted.tsv",
        "per_sample": outdir / "per_sample_hg38" / f"{sid}.hg38.beta.tsv.gz",
    }


def write_source_cpg_bed(
    spec: SampleSpec,
    anchor: str,
    args: argparse.Namespace,
    source_fa: FastaChunkReader,
    keep_chroms: set[str],
    bed_path: Path,
    stats: SampleStats,
) -> None:
    import pyBigWig

    bed_path.parent.mkdir(parents=True, exist_ok=True)
    bw = pyBigWig.open(spec.beta_bw)
    rid = 0
    width_counter = Counter()
    value_min = math.inf
    value_max = -math.inf
    try:
        with open(bed_path, "wt") as out:
            for bw_chrom in bw.chroms().keys():
                nchrom = normalize_chrom(bw_chrom)
                if nchrom not in keep_chroms:
                    continue
                for start, end, value in iter_bw_intervals(bw, bw_chrom, args.bw_window):
                    stats.intervals_seen += 1
                    width = end - start
                    width_counter[width] += 1
                    if width < 1 or width > args.max_interval_width:
                        stats.skipped_wide_interval += 1
                        continue

                    beta = normalize_beta(value, args.beta_scale)
                    if not math.isfinite(beta):
                        stats.skipped_invalid_value += 1
                        continue
                    stats.finite_values += 1
                    value_min = min(value_min, beta)
                    value_max = max(value_max, beta)

                    c0 = start if anchor == "c_at_start" else start - 1
                    if c0 < 0 or source_fa.fetch(nchrom, c0, c0 + 2) != "CG":
                        stats.skipped_invalid_source_cpg += 1
                        continue

                    # BED4 name carries both a numeric record ID and the beta value.
                    name = f"{rid}~{beta:.12g}"
                    out.write(f"{nchrom}\t{c0}\t{c0 + 2}\t{name}\n")
                    rid += 1
                    stats.source_cpg_written_to_bed += 1
    finally:
        bw.close()

    stats.interval_widths = {str(k): int(v) for k, v in sorted(width_counter.items())}
    stats.value_min = None if value_min == math.inf else value_min
    stats.value_max = None if value_max == -math.inf else value_max

    if stats.source_cpg_written_to_bed == 0:
        raise SystemExit(f"{spec.sample}: no validated source CpGs were written.")
    valid_source_fraction = stats.source_cpg_written_to_bed / max(
        1, stats.source_cpg_written_to_bed + stats.skipped_invalid_source_cpg
    )
    if valid_source_fraction < args.anchor_min_fraction:
        raise SystemExit(
            f"{spec.sample}: only {valid_source_fraction:.2%} of candidate positions "
            "validate as hg19 CpG sites. Coordinate convention or FASTA is likely wrong."
        )


def run_liftover(paths: Dict[str, Path], args: argparse.Namespace) -> None:
    liftover = shutil.which(args.liftover_bin) or (
        args.liftover_bin if Path(args.liftover_bin).is_file() else None
    )
    if not liftover:
        raise SystemExit(
            f"Cannot find UCSC liftOver executable: {args.liftover_bin}. "
            "Install ucsc-liftover or pass --liftover_bin."
        )
    cmd = [
        str(liftover),
        "-minMatch=1.0",
        "-multiple",
        str(paths["source_bed"]),
        str(args.chain),
        str(paths["mapped_bed"]),
        str(paths["unmapped_bed"]),
    ]
    subprocess.run(cmd, check=True)


def count_unmapped(path: Path) -> int:
    n = 0
    if not path.exists():
        return n
    with open(path, "rt") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                n += 1
    return n


def parse_liftover_output(paths: Dict[str, Path], stats: SampleStats) -> None:
    with open(paths["mapped_bed"], "rt") as inp, open(paths["mapped_id"], "wt") as out:
        for line in inp:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4 or "~" not in fields[3]:
                stats.liftover_malformed_output += 1
                continue
            try:
                rid_s, beta_s = fields[3].split("~", 1)
                rid = int(rid_s)
                start = int(fields[1])
                end = int(fields[2])
                beta = float(beta_s)
            except Exception:
                stats.liftover_malformed_output += 1
                continue
            out.write(f"{rid}\t{fields[0]}\t{start}\t{end}\t{beta:.12g}\n")
    stats.liftover_unmapped = count_unmapped(paths["unmapped_bed"])


def external_sort(in_path: Path, out_path: Path, keys: List[str]) -> None:
    sort_bin = shutil.which("sort")
    if not sort_bin:
        raise SystemExit("GNU/coreutils sort is required for large WGBS files.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sort_bin, "-T", str(out_path.parent), "-t", "\t", *keys, str(in_path)]
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    with open(out_path, "wt") as out:
        subprocess.run(cmd, check=True, stdout=out, env=env)


def select_unique_targets(
    paths: Dict[str, Path],
    target_fa: FastaChunkReader,
    keep_chroms: set[str],
    stats: SampleStats,
) -> None:
    """Group liftOver -multiple records by source ID and retain one valid CpG target."""
    with open(paths["mapped_id_sorted"], "rt") as inp, open(paths["target_raw"], "wt") as out:
        current_id: Optional[int] = None
        group: List[Tuple[str, int, int, float]] = []

        def emit(rows: List[Tuple[str, int, int, float]]) -> None:
            if not rows:
                return
            valid: Dict[Tuple[str, int, int], float] = {}
            for chrom_raw, start, end, beta in rows:
                chrom = normalize_chrom(chrom_raw)
                if chrom not in keep_chroms:
                    stats.skipped_nonstandard_target += 1
                    continue
                if end - start != 2:
                    stats.skipped_target_not_2bp += 1
                    continue
                if target_fa.fetch(chrom, start, end) != "CG":
                    stats.skipped_invalid_target_cpg += 1
                    continue
                valid[(chrom, start, end)] = beta

            if len(valid) == 0:
                return
            if len(valid) > 1:
                stats.skipped_multi_target += 1
                return
            (chrom, start, _end), beta = next(iter(valid.items()))
            pos1 = start + 1
            label = f"{chrom}_{pos1}"
            out.write(
                f"{chrom_order(chrom)}\t{chrom}\t{pos1}\t{label}\t{beta:.12g}\n"
            )
            stats.unique_lifted_records += 1

        for line in inp:
            if not line.strip():
                continue
            r = line.rstrip("\n").split("\t")
            rid = int(r[0])
            row = (r[1], int(r[2]), int(r[3]), float(r[4]))
            if current_id is None:
                current_id = rid
            if rid != current_id:
                emit(group)
                group = []
                current_id = rid
            group.append(row)
        emit(group)


def collapse_target_collisions(
    paths: Dict[str, Path], args: argparse.Namespace, stats: SampleStats
) -> None:
    paths["per_sample"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["target_sorted"], "rt") as inp, open_text(paths["per_sample"], "wt") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["sort_key", "chrom", "pos", "chr_pos", "beta"])

        current_key = None
        rows: List[List[str]] = []

        def emit(group: List[List[str]]) -> None:
            if not group:
                return
            if len(group) > 1:
                stats.collision_groups += 1
                stats.collision_records += len(group)
                if args.collision_policy == "drop":
                    return
            if args.collision_policy == "first" or len(group) == 1:
                beta = float(group[0][4])
            else:
                beta = sum(float(r[4]) for r in group) / len(group)
            r0 = group[0]
            writer.writerow([r0[0], r0[1], r0[2], r0[3], f"{beta:.{args.digits}f}"])
            stats.final_records += 1

        for line in inp:
            if not line.strip():
                continue
            r = line.rstrip("\n").split("\t")
            key = (r[0], r[1], r[2], r[3])
            if current_key is None:
                current_key = key
            if key != current_key:
                emit(rows)
                rows = []
                current_key = key
            rows.append(r)
        emit(rows)


def iter_per_sample(path: Path) -> Iterator[Tuple[int, int, str, str, str]]:
    with open_text(path, "rt") as fh:
        header = next(fh, None)
        for line in fh:
            if not line.strip():
                continue
            r = line.rstrip("\n").split("\t")
            yield int(r[0]), int(r[2]), r[1], r[3], r[4]


def build_features(
    samples: Sequence[SampleSpec],
    paths_by_sample: Dict[str, Dict[str, Path]],
    features_path: Path,
    args: argparse.Namespace,
) -> Dict[str, int | float]:
    n_samples = len(samples)
    if args.feature_mode == "union":
        min_count = 1
    elif args.feature_mode == "intersection":
        min_count = n_samples
    else:
        min_count = max(1, math.ceil(args.min_sample_fraction * n_samples))

    streams = []
    for sample_idx, s in enumerate(samples):
        p = paths_by_sample[s.sample]["per_sample"]
        streams.append(
            (
                (sk, pos, chrom, label, sample_idx)
                for sk, pos, chrom, label, _beta in iter_per_sample(p)
            )
        )

    merged = heapq.merge(
        *streams, key=lambda x: (x[0], x[1], x[2], x[3], x[4])
    )
    features_path.parent.mkdir(parents=True, exist_ok=True)
    total_union = kept = 0

    with open_text(features_path, "wt") as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(["sort_key", "chrom", "pos", "chr_pos", "n_samples", "sample_fraction"])

        current_key = None
        sample_ids: set[int] = set()

        def emit(key, ids: set[int]) -> None:
            nonlocal total_union, kept
            if key is None:
                return
            total_union += 1
            count = len(ids)
            if count < min_count:
                return
            sk, pos, chrom, label = key
            w.writerow([sk, chrom, pos, label, count, f"{count / n_samples:.6f}"])
            kept += 1

        for sk, pos, chrom, label, sample_idx in merged:
            key = (sk, pos, chrom, label)
            if current_key is None:
                current_key = key
            if key != current_key:
                emit(current_key, sample_ids)
                sample_ids = set()
                current_key = key
            sample_ids.add(sample_idx)
        emit(current_key, sample_ids)

    if kept == 0:
        raise SystemExit(
            f"No features passed feature_mode={args.feature_mode}; "
            "lower --min_sample_fraction or inspect per-sample QC."
        )
    return {
        "n_samples": n_samples,
        "minimum_samples_required": min_count,
        "union_features": total_union,
        "retained_features": kept,
    }


def iter_features(path: Path) -> Iterator[Tuple[int, int, str, str]]:
    with open_text(path, "rt") as fh:
        next(fh, None)
        for line in fh:
            if not line.strip():
                continue
            r = line.rstrip("\n").split("\t")
            yield int(r[0]), int(r[2]), r[1], r[3]


def write_matrix(
    samples: Sequence[SampleSpec],
    paths_by_sample: Dict[str, Dict[str, Path]],
    features_path: Path,
    matrix_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, int]]:
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_stats: Dict[str, Dict[str, int]] = {}

    with open_text(matrix_path, "wt") as out:
        out.write("Sample")
        for _sk, _pos, _chrom, label in iter_features(features_path):
            out.write("\t" + label)
        out.write("\n")

        for s in samples:
            sample_iter = iter_per_sample(paths_by_sample[s.sample]["per_sample"])
            cur = next(sample_iter, None)
            observed = missing = 0
            out.write(s.sample)

            for f_sk, f_pos, f_chrom, f_label in iter_features(features_path):
                fkey = (f_sk, f_pos, f_chrom, f_label)
                value = args.missing
                while cur is not None:
                    ckey = (cur[0], cur[1], cur[2], cur[3])
                    if ckey >= fkey:
                        break
                    cur = next(sample_iter, None)
                if cur is not None and (cur[0], cur[1], cur[2], cur[3]) == fkey:
                    value = cur[4]
                    observed += 1
                    cur = next(sample_iter, None)
                else:
                    missing += 1
                out.write("\t" + value)
            out.write("\n")
            matrix_stats[s.sample] = {"observed": observed, "missing": missing}

    return matrix_stats


def process_sample(
    spec: SampleSpec,
    args: argparse.Namespace,
    source_fa: FastaChunkReader,
    target_fa: FastaChunkReader,
    keep_chroms: set[str],
    outdir: Path,
) -> Tuple[SampleStats, Dict[str, Path]]:
    stats = SampleStats(sample=spec.sample, beta_bw=spec.beta_bw)
    paths = work_paths(outdir, spec.sample)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    anchor = inspect_and_choose_anchor(spec, args, source_fa, keep_chroms, stats)
    write_source_cpg_bed(spec, anchor, args, source_fa, keep_chroms, paths["source_bed"], stats)
    run_liftover(paths, args)
    parse_liftover_output(paths, stats)

    external_sort(paths["mapped_id"], paths["mapped_id_sorted"], ["-k1,1n"])
    select_unique_targets(paths, target_fa, keep_chroms, stats)
    external_sort(
        paths["target_raw"],
        paths["target_sorted"],
        ["-k1,1n", "-k2,2", "-k3,3n", "-k4,4"],
    )
    collapse_target_collisions(paths, args, stats)

    if not args.keep_work:
        for key in (
            "source_bed", "mapped_bed", "unmapped_bed", "mapped_id",
            "mapped_id_sorted", "target_raw", "target_sorted"
        ):
            try:
                paths[key].unlink()
            except FileNotFoundError:
                pass
        try:
            paths["dir"].rmdir()
        except OSError:
            pass
    return stats, paths


def main() -> None:
    args = parse_args()
    validate_args(args)

    try:
        import pyBigWig  # noqa: F401
    except Exception as e:
        raise SystemExit("Missing pyBigWig. Install it from bioconda/conda-forge.") from e

    for required in (args.chain, args.source_fasta, args.target_fasta):
        if not Path(required).is_file():
            raise SystemExit(f"Required file not found: {required}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    samples = read_samples(args)
    keep_chroms = {
        normalize_chrom(x.strip()) for x in args.chroms.split(",") if x.strip()
    }

    matrix_path = Path(args.matrix) if args.matrix else outdir / "hg38_beta_matrix.tsv.gz"
    features_path = Path(args.features) if args.features else outdir / "hg38_features.tsv.gz"
    report_path = (
        Path(args.report_json) if args.report_json else outdir / "processing_report.json"
    )

    source_fa = FastaChunkReader(args.source_fasta, args.fasta_chunk)
    target_fa = FastaChunkReader(args.target_fasta, args.fasta_chunk)
    stats_list: List[SampleStats] = []
    paths_by_sample: Dict[str, Dict[str, Path]] = {}

    try:
        for i, spec in enumerate(samples, 1):
            print(f"[{i}/{len(samples)}] {spec.sample}", file=sys.stderr, flush=True)
            stats, paths = process_sample(
                spec, args, source_fa, target_fa, keep_chroms, outdir
            )
            stats_list.append(stats)
            paths_by_sample[spec.sample] = paths
    finally:
        source_fa.close()
        target_fa.close()

    feature_stats = build_features(samples, paths_by_sample, features_path, args)
    matrix_stats = write_matrix(
        samples, paths_by_sample, features_path, matrix_path, args
    )

    report = {
        "dataset": "GSE178169",
        "input": [asdict(s) for s in samples],
        "outputs": {
            "matrix": str(matrix_path),
            "features": str(features_path),
            "per_sample_directory": str(outdir / "per_sample_hg38"),
        },
        "parameters": {
            "chain": args.chain,
            "source_fasta": args.source_fasta,
            "target_fasta": args.target_fasta,
            "beta_scale": args.beta_scale,
            "input_anchor": args.input_anchor,
            "anchor_min_fraction": args.anchor_min_fraction,
            "max_interval_width": args.max_interval_width,
            "collision_policy": args.collision_policy,
            "feature_mode": args.feature_mode,
            "min_sample_fraction": args.min_sample_fraction,
            "chroms": sorted(keep_chroms, key=chrom_order),
            "liftover_command_policy": "-multiple -minMatch=1.0 on complete 2-bp CpG intervals",
        },
        "scientific_notes": [
            "The GSE178169 bigWigs contain methylation ratios but no read depth.",
            "No minimum-depth filter can be applied to these processed bigWigs.",
            "Each CpG is lifted as a complete 2-bp interval and target hg38 sequence must be CG.",
            "Missing values are emitted as the configured missing token and never converted to zero.",
            "The default feature set retains CpGs observed in at least 80% of samples.",
            "Distinct source CpGs converging on one target CpG are dropped by default.",
        ],
        "feature_stats": feature_stats,
        "matrix_observation_stats": matrix_stats,
        "sample_stats": [asdict(s) for s in stats_list],
    }
    with open(report_path, "wt") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "matrix": str(matrix_path),
                "features": str(features_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

