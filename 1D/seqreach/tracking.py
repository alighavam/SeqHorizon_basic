"""Weights & Biases tracking, behind a small facade.

The rest of the codebase never imports ``wandb``.  It talks to ``Tracker``, which is a
no-op when tracking is disabled or the package is missing, so training, the ablation
grid, and the tests all keep working unchanged on a machine that has never heard of W&B.

Naming convention, so runs stay findable once there are hundreds of them:

* run **name**  = ``<exp_name>_s<seed>`` -- identical to the ``runs/`` directory.
* run **group** = whatever ``wandb.group`` says (e.g. ``ablations-3M``); seeds and
  ablations of one campaign then collapse into a single row in the UI.
* model **artifact** = ``<exp_name>_s<seed>`` of type ``model``, versioned, tagged
  ``step-<N>`` and ``latest``, with the full config attached as metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, f"{key}."))
        else:
            out[key] = list(v) if isinstance(v, tuple) else v
    return out


class Tracker:
    """Thin wrapper around a single W&B run.  Inert unless ``cfg.wandb.enabled``."""

    def __init__(self, cfg: Config, run_dir: str | Path, job_type: str | None = None,
                 name: str | None = None, extra_config: dict | None = None,
                 resume: str | None = None):
        self.cfg = cfg
        self.run = None
        self.run_dir = Path(run_dir)
        w = cfg.wandb
        if not w.enabled or w.mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - depends on the environment
            print(f"[wandb] disabled: {exc}.  pip install wandb")
            return

        self._wandb = wandb
        config = _flat(cfg.to_dict())
        config.update(extra_config or {})
        self.run = wandb.init(
            project=w.project,
            entity=w.entity,
            name=name or cfg.run_name,
            group=w.group,
            job_type=job_type or w.job_type,
            tags=list(w.tags) or None,
            notes=w.notes or None,
            mode=w.mode,
            config=config,
            dir=str(self.run_dir),
            resume="allow" if resume else None,
            id=resume,
            save_code=w.save_code,
        )
        print(f"[wandb] {self.run.url or w.mode} ({w.project}/{name or cfg.run_name})")

    # ------------------------------------------------------------------ writes
    @property
    def active(self) -> bool:
        return self.run is not None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        clean = {k: v for k, v in data.items() if v is not None}
        self.run.log(clean, step=step)

    def log_video(self, path: str | Path, step: int | None = None,
                  key: str = "rollout", caption: str = "") -> None:
        if self.run is None:
            return
        vid = self._wandb.Video(str(path), caption=caption or None, format="gif")
        self.run.log({key: vid}, step=step)

    def log_images(self, paths: dict[str, str | Path], step: int | None = None) -> None:
        if self.run is None:
            return
        imgs = {k: self._wandb.Image(str(p)) for k, p in paths.items() if Path(p).exists()}
        if imgs:
            self.run.log(imgs, step=step)

    def summary(self, data: dict[str, Any]) -> None:
        if self.run is None:
            return
        self.run.summary.update({k: v for k, v in data.items() if v is not None})

    def save_checkpoint(self, path: str | Path, step: int | None = None,
                        metadata: dict | None = None) -> None:
        """Upload the checkpoint as a versioned model artifact."""
        path = Path(path)
        if self.run is None or not path.exists():
            return
        art = self._wandb.Artifact(
            name=self.cfg.run_name, type="model",
            metadata={"config": self.cfg.to_dict(), "global_step": step,
                      **(metadata or {})},
        )
        art.add_file(str(path), name="checkpoint.pt")
        cfg_file = path.parent / "config.yaml"
        if cfg_file.exists():
            art.add_file(str(cfg_file), name="config.yaml")
        aliases = ["latest"] + ([f"step-{step}"] if step else [])
        self.run.log_artifact(art, aliases=aliases)
        print(f"[wandb] logged model artifact {self.cfg.run_name} ({aliases})")

    def save_files(self, paths: list[str | Path]) -> None:
        """Attach files (figures, results.json, ...) to the run's Files tab."""
        if self.run is None:
            return
        for p in paths:
            if Path(p).exists():
                self.run.save(str(p), base_path=str(Path(p).parent), policy="now")

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


def read_train_log(run_dir: str | Path) -> list[dict]:
    """Parse ``train_log.jsonl`` -- used to backfill an already-finished run."""
    log = Path(run_dir) / "train_log.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
