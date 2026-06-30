"""Model and data loading.

The trained checkpoint can only be instantiated with the ``Delphi`` / ``DelphiConfig``
classes from the desired fork, and the eval batch is assembled by that clone's ``get_batch``. 
Rather than vendoring a possibly-stale copy, we put ``paths.delphi_repo`` on ``sys.path`` and 
import ``model`` / ``utils`` from it, forwarding extra ``get_batch`` kwargs from the config.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .config import EvalConfig


@dataclass
class DelphiBundle:
    """Everything inference needs: the loaded model plus the fork's helper functions."""

    model: Any
    model_module: ModuleType
    utils_module: ModuleType
    device: str

    def get_p2i(self, data: np.ndarray) -> np.ndarray:
        return self.utils_module.get_p2i(data)

    def get_batch(self, ix, data, p2i, **kwargs):
        return self.utils_module.get_batch(ix, data, p2i, **kwargs)


# --------------------------------------------------------------------------- #
# device                                                                      #
# --------------------------------------------------------------------------- #
def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# importing the user's Delphi fork                                            #
# --------------------------------------------------------------------------- #
def import_delphi(delphi_repo: str | Path) -> tuple[ModuleType, ModuleType]:
    """Import ``model`` and ``utils`` from the user's Delphi fork."""
    repo = Path(delphi_repo).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"delphi_repo does not exist: {repo}")
    for required in ("model.py", "utils.py"):
        if not (repo / required).exists():
            raise FileNotFoundError(f"{required} not found in delphi_repo {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # Drop any previously-imported same-named modules from a different repo.
    for name in ("model", "utils"):
        if name in sys.modules and getattr(sys.modules[name], "__file__", "").startswith(str(repo)) is False:
            del sys.modules[name]
    model_module = importlib.import_module("model")
    utils_module = importlib.import_module("utils")
    return model_module, utils_module


# --------------------------------------------------------------------------- #
# checkpoint                                                                   #
# --------------------------------------------------------------------------- #
def load_model(cfg: EvalConfig) -> DelphiBundle:
    """Load the checkpoint exactly as upstream does, returning a ready-to-eval model."""
    import torch

    cfg.require_paths("delphi_repo", "ckpt")
    device = resolve_device(cfg.inference.device)
    model_module, utils_module = import_delphi(cfg.paths.delphi_repo)

    checkpoint = torch.load(cfg.paths.ckpt, map_location=device, weights_only=False)
    conf = model_module.DelphiConfig(**checkpoint["model_args"])
    model = model_module.Delphi(conf)
    state_dict = _strip_compile_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return DelphiBundle(model=model, model_module=model_module, utils_module=utils_module, device=device)


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove the ``_orig_mod.`` prefix left by ``torch.compile`` (nanoGPT convention)."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


# --------------------------------------------------------------------------- #
# data                                                                         #
# --------------------------------------------------------------------------- #
def load_data(bin_path: str | Path) -> np.ndarray:
    """Read a Delphi ``.bin`` file as an ``(N, 3)`` int64 array [patient_id, age_days, token_id]."""
    arr = np.fromfile(str(bin_path), dtype=np.uint32)
    if arr.size % 3 != 0:
        raise ValueError(f"{bin_path}: length {arr.size} is not divisible by 3 (expected [pid, age, token] triples).")
    return arr.reshape(-1, 3).astype(np.int64)


def make_eval_batch(bundle: DelphiBundle, data: np.ndarray, p2i: np.ndarray, cfg: EvalConfig):
    """Assemble one big eval batch via the fork's ``get_batch`` (mirrors evaluate_auc.main)."""
    inf = cfg.inference
    n = len(p2i)
    subset = n if inf.dataset_subset_size in (-1, None) else min(inf.dataset_subset_size, n)
    kwargs: dict[str, Any] = dict(
        select=inf.select,
        block_size=inf.block_size,
        device=bundle.device,
        padding=inf.padding_mode,
        no_event_token_rate=inf.no_event_token_rate,
    )
    kwargs.update(inf.get_batch_kwargs or {})
    return bundle.get_batch(range(subset), data, p2i, **kwargs)
