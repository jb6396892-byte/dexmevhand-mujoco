from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EnvPaths:
    project_root: Path
    dexmv_sim: Path
    dexmv_learn: Path
    conda_env: str
    conda_bin: Path
    ld_library_path: str
    ld_preload: str
    pycache_prefix: str


def load_dotenv(path: Path | None = None, *, override: bool = False) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        env_path = PROJECT_ROOT / ".env.example"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def get_env_paths() -> EnvPaths:
    load_dotenv()
    return EnvPaths(
        project_root=Path(os.environ.get("FROMREALHAND_ROOT", PROJECT_ROOT)).expanduser().resolve(),
        dexmv_sim=Path(os.environ.get("DEXMV_SIM", "/home/smgbro/dexmv-sim")).expanduser().resolve(),
        dexmv_learn=Path(os.environ.get("DEXMV_LEARN", "/home/smgbro/dexmv-learn")).expanduser().resolve(),
        conda_env=os.environ.get("CONDA_ENV", "dexmv"),
        conda_bin=Path(os.environ.get("CONDA_BIN", "/home/smgbro/miniconda3/bin/conda")).expanduser(),
        ld_library_path=os.environ.get(
            "LD_LIBRARY_PATH",
            "/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu",
        ),
        ld_preload=os.environ.get(
            "LD_PRELOAD",
            "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        ),
        pycache_prefix=os.environ.get("PYTHONPYCACHEPREFIX", "/tmp/fromrealhand-pycache"),
    )


def configure_runtime_paths() -> EnvPaths:
    env = get_env_paths()
    additions = [
        env.project_root / "src",
        env.dexmv_sim,
        env.dexmv_learn,
        env.dexmv_learn / "mjrl",
    ]
    for path in reversed(additions):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    os.environ.setdefault("LD_LIBRARY_PATH", env.ld_library_path)
    os.environ.setdefault("LD_PRELOAD", env.ld_preload)
    os.environ.setdefault("PYTHONPYCACHEPREFIX", env.pycache_prefix)
    return env


def project_path(*parts: str) -> Path:
    return get_env_paths().project_root.joinpath(*parts)


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path
