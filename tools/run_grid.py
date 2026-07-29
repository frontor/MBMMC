from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run explicit candidates or Cartesian parameter grids for the RF, "
            "crossNN, and MPCNet trainers. Commands and logs are recorded for reproducibility."
        )
    )
    parser.add_argument("--config", required=True, help="YAML or JSON grid configuration.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model key to run; may be repeated or comma-separated. Default: all models.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--resume", action="store_true", help="Skip non-empty output directories.")
    parser.add_argument("--limit", type=int, default=0, help="Run at most N candidates per model; 0 means all.")
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Keep unresolved ${ENV_VAR} placeholders instead of raising an error.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch modules. Default: current interpreter.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError("Config must be a mapping containing a top-level 'models' mapping.")
    return data


def expand_env(value: Any, allow_unresolved: bool) -> Any:
    if isinstance(value, dict):
        return {k: expand_env(v, allow_unresolved) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, allow_unresolved) for v in value]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in os.environ:
            return os.environ[name]
        if allow_unresolved:
            return match.group(0)
        raise ValueError(
            f"Environment variable {name!r} is required by the grid config. "
            f"Export it first or pass --allow-unresolved for command preview."
        )

    return ENV_PATTERN.sub(repl, value)


def safe_token(value: Any) -> str:
    token = str(value).strip().replace(os.sep, "-")
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", token)
    return token.strip("-")[:48] or "value"


def candidate_id_from_args(args: dict[str, Any]) -> str:
    parts = [f"{safe_token(k)}-{safe_token(v)}" for k, v in args.items()]
    raw = "__".join(parts)
    if len(raw) <= 140:
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return raw[:120] + "__" + digest


def expand_candidates(model_spec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    fixed = dict(model_spec.get("fixed_args") or {})
    explicit = model_spec.get("candidates")
    grid = model_spec.get("grid")

    if explicit is not None and grid is not None:
        raise ValueError("A model may define either 'candidates' or 'grid', not both.")

    expanded: list[tuple[str, dict[str, Any]]] = []
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("'candidates' must be a list.")
        for index, item in enumerate(explicit, start=1):
            if not isinstance(item, dict):
                raise ValueError("Every candidate must be a mapping.")
            item = dict(item)
            candidate_id = str(item.pop("id", f"candidate_{index:03d}"))
            candidate_args = dict(item.pop("args", item))
            merged = {**fixed, **candidate_args}
            expanded.append((candidate_id, merged))
        return expanded

    if grid is not None:
        if not isinstance(grid, dict) or not grid:
            raise ValueError("'grid' must be a non-empty mapping.")
        keys = list(grid)
        values: list[Iterable[Any]] = []
        for key in keys:
            choices = grid[key]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Grid value for {key!r} must be a non-empty list.")
            values.append(choices)
        for combo in itertools.product(*values):
            varied = dict(zip(keys, combo))
            merged = {**fixed, **varied}
            expanded.append((candidate_id_from_args(varied), merged))
        return expanded

    return [("default", fixed)]


def value_to_cli(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def command_for(module: str, args: dict[str, Any], python_executable: str) -> list[str]:
    command = [python_executable, "-m", module]
    for key, value in args.items():
        if value is None or value is False:
            continue
        option = "--" + key.replace("_", "-")
        # The retained trainers use underscore option names. Preserve them.
        option = "--" + key
        if value is True:
            command.append(option)
        else:
            command.extend([option, value_to_cli(value)])
    return command


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def selected_models(raw: list[str], available: set[str]) -> list[str]:
    if not raw:
        return sorted(available)
    requested: list[str] = []
    for item in raw:
        requested.extend(x.strip() for x in item.split(",") if x.strip())
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Available: {sorted(available)}")
    return requested


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = expand_env(load_config(config_path), args.allow_unresolved)
    models = config["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("'models' must be a non-empty mapping.")

    output_root = Path(config.get("output_root", "outputs/grid_runs"))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.jsonl"
    commit = git_commit()

    for model_key in selected_models(args.model, set(models)):
        spec = models[model_key]
        if not isinstance(spec, dict):
            raise ValueError(f"Model spec {model_key!r} must be a mapping.")
        module = str(spec["module"])
        output_arg = str(spec.get("output_arg", "outdir"))
        candidates = expand_candidates(spec)
        if args.limit > 0:
            candidates = candidates[: args.limit]

        for candidate_id, candidate_args in candidates:
            run_dir = output_root / model_key / safe_token(candidate_id)
            if args.resume and run_dir.exists() and any(run_dir.iterdir()):
                print(f"[SKIP] {model_key}/{candidate_id}: non-empty output directory")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            candidate_args = dict(candidate_args)
            candidate_args.setdefault(output_arg, str(run_dir))
            command = command_for(module, candidate_args, args.python)
            command_text = shlex.join(command)
            (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
            (run_dir / "resolved_args.json").write_text(
                json.dumps(candidate_args, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            started = dt.datetime.now(dt.timezone.utc)
            record = {
                "model": model_key,
                "candidate_id": candidate_id,
                "module": module,
                "command": command,
                "command_text": command_text,
                "output_dir": str(run_dir),
                "config": str(config_path),
                "git_commit": commit,
                "python": sys.version,
                "platform": platform.platform(),
                "started_at_utc": started.isoformat(),
                "dry_run": bool(args.dry_run),
            }
            print(f"[{'DRY-RUN' if args.dry_run else 'RUN'}] {model_key}/{candidate_id}")
            print(command_text)

            if args.dry_run:
                record.update({"status": "dry-run", "returncode": None})
                append_jsonl(manifest_path, record)
                continue

            log_path = run_dir / "training.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                returncode = process.wait()

            finished = dt.datetime.now(dt.timezone.utc)
            record.update(
                {
                    "finished_at_utc": finished.isoformat(),
                    "elapsed_seconds": (finished - started).total_seconds(),
                    "returncode": returncode,
                    "status": "completed" if returncode == 0 else "failed",
                    "log": str(log_path),
                }
            )
            append_jsonl(manifest_path, record)
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, command)

    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
