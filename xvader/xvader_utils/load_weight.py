import torch
import os
from typing import Dict, Tuple, Optional, Iterable
import shutil
from huggingface_hub import hf_hub_download

def _is_raw_state_dict(d: Dict) -> bool:
    """Heuristic: looks like a param-name -> Tensor mapping."""
    if not isinstance(d, dict) or not d:
        return False
    k0, v0 = next(iter(d.items()))
    return isinstance(k0, str) and (isinstance(v0, torch.Tensor) or hasattr(v0, "shape"))

def _maybe_pick_ema(ema_obj):
    """
    Try to extract a raw state_dict from a variety of EMA formats:
      - dict of params
      - {"state_dict": {...}}
      - list/tuple of dicts
      - dicts with a 'shadow' or 'model' key
    """
    if _is_raw_state_dict(ema_obj):
        return ema_obj
    if isinstance(ema_obj, dict):
        for key in ("state_dict", "model", "shadow"):
            if key in ema_obj and _is_raw_state_dict(ema_obj[key]):
                return ema_obj[key]
    if isinstance(ema_obj, (list, tuple)):
        for item in ema_obj:
            if _is_raw_state_dict(item):
                return item
            if isinstance(item, dict):
                for key in ("state_dict", "model", "shadow"):
                    if key in item and _is_raw_state_dict(item[key]):
                        return item[key]
    return None

def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    plen = len(prefix)
    return { (k[plen:] if k.startswith(prefix) else k): v for k, v in sd.items() }

def load_checkpoint_into_model(
    model: torch.nn.Module,
    ckpt_path: str,
    device: str | torch.device = "cpu",
    *,
    strict: bool = True,
    try_prefixes: Iterable[str] = ("module.", "model.", "net.", "encoder_q.", "student."),
    prefer_ema: bool = False,
    verbose: bool = True,
) -> Tuple[Tuple[Iterable[str], Iterable[str]], Dict[str, torch.Tensor]]:
    """
    Load weights into `model` from `ckpt_path`, handling common checkpoint wrappers.

    Returns:
        ((missing_keys, unexpected_keys), loaded_state_dict)

    Raises:
        FileNotFoundError / ValueError on unrecoverable layout issues.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # 1) Find the raw state_dict inside the checkpoint
    sd: Optional[Dict[str, torch.Tensor]] = None

    # Common top-level keys in training checkpoints
    for key in (("ema", "ema_model", "ema_models") if prefer_ema else ()):
        if key in ckpt:
            cand = _maybe_pick_ema(ckpt[key])
            if cand is not None:
                sd = cand
                break

    if sd is None:
        for key in ("model", "state_dict"):
            if key in ckpt and _is_raw_state_dict(ckpt[key]):
                sd = ckpt[key]
                break

    if sd is None and _is_raw_state_dict(ckpt):
        sd = ckpt  # file was saved as a bare state_dict

    # Fallback: try EMA if available and not yet used
    if sd is None and not prefer_ema:
        for key in ("ema", "ema_model", "ema_models"):
            if key in ckpt:
                cand = _maybe_pick_ema(ckpt[key])
                if cand is not None:
                    sd = cand
                    break

    if sd is None:
        raise ValueError(
            f"Could not find a model state_dict in {ckpt_path}. "
            f"Looked for keys like 'model', 'state_dict', 'ema*'. Found top-level keys: {list(ckpt.keys())}"
            if isinstance(ckpt, dict) else
            f"Checkpoint is type {type(ckpt)}; expected dict or state_dict."
        )

    # 2) Try to load as-is; if that fails, progressively strip known prefixes
    def _attempt_load(state_dict):
        try:
            # PyTorch >= 2.0 returns IncompatibleKeys object (missing, unexpected)
            incompatible = model.load_state_dict(state_dict, strict=strict)
            # Normalize return value across versions
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            return True, (missing, unexpected)
        except RuntimeError as e:
            return False, e

    ok, result = _attempt_load(sd)
    if not ok:
        # Try stripping common prefixes (DDP etc.)
        last_error = result
        for p in try_prefixes:
            sd_p = _strip_prefix(sd, p)
            ok, result = _attempt_load(sd_p)
            if ok:
                sd = sd_p
                break
        if not ok:
            # As a final fallback, if strict=True caused failure, retry non-strict just to print a useful report
            try:
                _ = model.load_state_dict(sd, strict=False)
            except Exception:
                pass
            raise RuntimeError(f"Failed to load weights (even after prefix stripping). Last error:\n{last_error}")

    missing, unexpected = result
    if verbose:
        print(f"[load_checkpoint_into_model] Loaded from: {ckpt_path}")
        if prefer_ema:
            print("[load_checkpoint_into_model] Prefer EMA: True")
        # Show a quick sample key:
        k0 = next(iter(sd.keys()))
        print(f"[load_checkpoint_into_model] Example key: {k0}")
        if missing:
            print(f"[load_checkpoint_into_model] Missing keys ({len(missing)}): first 5 -> {missing[:5]}")
        if unexpected:
            print(f"[load_checkpoint_into_model] Unexpected keys ({len(unexpected)}): first 5 -> {unexpected[:5]}")
        if not missing and not unexpected:
            print("[load_checkpoint_into_model] Perfect match ✅")

    return (missing, unexpected), sd


def resolve_checkpoint(checkpoint_arg: str) -> str:
    """
    Resolve a checkpoint string. Supports:
      - local files
      - hf://repo_id/filename
      - hf://repo_id/filename@revision
    Returns a local filepath.
    """
    # 1) If file exists locally → return directly
    if os.path.isfile(checkpoint_arg):
        print(f"[INFO] Using local checkpoint: {checkpoint_arg}")
        return checkpoint_arg

    # 2) HF format: hf://repo_id/filename or hf://repo_id/filename@revision
    if checkpoint_arg.startswith("hf://"):
        spec = checkpoint_arg[len("hf://"):]  # strip prefix
        
        # Extract optional revision
        if "@" in spec:
            path_part, revision = spec.split("@", 1)
        else:
            path_part, revision = spec, None

        # repo_id is everything except the last item
        parts = path_part.split("/")
        repo_id = "/".join(parts[:-1])
        filename = parts[-1]

        print(f"[INFO] Downloading from HuggingFace:")
        print(f"       repo_id = {repo_id}")
        print(f"       filename = {filename}")
        print(f"       revision = {revision}")

        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
        print(f"[INFO] HF checkpoint resolved to local file: {ckpt_path}")
        return ckpt_path

    # 3) Unknown format
    raise ValueError(
        f"Invalid checkpoint argument: {checkpoint_arg}\n"
        f"Expected a local file or hf://repo/filename[@revision]"
    )

def save_checkpoint(url: str, output_path: str):
    cached_path = resolve_checkpoint(url)

    # Handle cases where output_path has no directory (e.g. "checkpoint.pt")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Copy bytes from cached_path → output_path
    shutil.copy2(cached_path, output_path)

    print(f"[INFO] Source checkpoint: {cached_path}")
    print(f"[INFO] Checkpoint copied to: {output_path}")