# By dreamraster · dreaMSCend
"""
hmlcore/shortcut_heads.py
Qwen3-style "Shortcut Heads" — shallow transformer blocks between main layers
that predict tokens further ahead (t+K). Independent of main LM head during FWD.
Trained with ground-truth at target offset. L_total = L_main + Σ(λ·L_shortcut_k)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ShortcutHeadConfig:
    enabled: bool = False
    layer_indices: list[int] = field(default_factory=lambda: [-3, -2])
    offsets: list[int] = field(default_factory=lambda: [2, 3])
    num_layers: int = 2
    hidden_dim: Optional[int] = None
    num_heads: int = 4
    loss_weight: float = 0.1
    freeze_after_sft: bool = True
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if len(self.layer_indices) != len(self.offsets):
            raise ValueError("layer_indices and offsets must match in length.")
        if any(o < 1 for o in self.offsets):
            raise ValueError("Offsets must be >= 1.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if self.loss_weight < 0:
            raise ValueError("loss_weight must be >= 0.")


class ShortcutHead(nn.Module):
    """Shallow transformer + LM head predicting token[t+offset]."""

    def __init__(
        self,
        model_hidden: int,
        vocab_size: int,
        offset: int,
        num_layers: int = 2,
        hidden_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.offset = offset
        sh = hidden_dim or (model_hidden // 2)
        while sh % num_heads != 0:
            sh += 1
        self.input_proj = nn.Linear(model_hidden, sh)
        self.pos_embed = nn.Parameter(torch.zeros(1, 8192, sh))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        el = nn.TransformerEncoderLayer(
            d_model=sh,
            nhead=num_heads,
            dim_feedforward=sh * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(el, num_layers=num_layers)
        self.ln = nn.LayerNorm(sh, eps=1e-5)
        self.output_head = nn.Linear(sh, vocab_size)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        slen = hidden.size(1)
        x = self.input_proj(hidden) + self.pos_embed[:, :slen, :]
        mask = torch.triu(
            torch.ones(slen, slen, dtype=torch.bool, device=hidden.device), diagonal=1
        )
        x = self.ln(self.transformer(src=x, mask=mask))
        logits = self.output_head(x)
        if slen <= self.offset:
            z = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
            return {"logits": logits, "loss": z, "loss_weight": 0.0, "valid_tokens": 0}
        sl = labels[:, self.offset :]
        loss = F.cross_entropy(
            logits[:, : -self.offset].reshape(-1, logits.size(-1)), sl.reshape(-1), ignore_index=-100
        )
        return {
            "logits": logits,
            "loss": loss,
            "loss_weight": 1.0,
            "valid_tokens": (sl != -100).sum().item(),
        }

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class ShortcutManager(nn.Module):
    """Manages shortcut heads via forward hooks on decoder layers."""

    def __init__(
        self,
        model: nn.Module,
        config: ShortcutHeadConfig,
        decoder: nn.Module,
        resolved: list[int],
    ):
        super().__init__()
        self.config = config
        self._decoder = decoder
        self._resolved = resolved
        self._hooks, self._captured = [], {}
        self._labels, self._results = None, {}
        self._attached = False
        mh, vs = self._hsize(model), self._vsize(model)
        self.shortcut_heads = nn.ModuleList()
        self.layer_to_head = {}
        for hi, (li, off) in enumerate(zip(resolved, config.offsets)):
            h = ShortcutHead(
                mh,
                vs,
                off,
                config.num_layers,
                config.hidden_dim,
                config.num_heads,
                config.dropout,
            )
            self.shortcut_heads.append(h)
            self.layer_to_head[li] = hi
            logger.info(
                "  Shortcut #%d: layer=%d → offset=%+d (%d params)",
                hi,
                li,
                off,
                h.num_params(),
            )
        logger.info("Shortcut heads total: %d params", self.total_params())

    @staticmethod
    def _hsize(m: nn.Module) -> int:
        # 1. Search recursively in configs
        def search_hsize(cfg) -> int | None:
            if cfg is None:
                return None
            if isinstance(cfg, dict):
                for a in ("hidden_size", "d_model", "n_embd"):
                    if a in cfg and isinstance(cfg[a], int):
                        return cfg[a]
                for v in cfg.values():
                    res = search_hsize(v)
                    if res is not None:
                        return res
            else:
                for a in ("hidden_size", "d_model", "n_embd"):
                    if hasattr(cfg, a) and isinstance(getattr(cfg, a), int):
                        return getattr(cfg, a)
                for attr in ("text_config", "vision_config", "config"):
                    if hasattr(cfg, attr):
                        res = search_hsize(getattr(cfg, attr))
                        if res is not None:
                            return res
                if hasattr(cfg, "to_dict"):
                    try:
                        res = search_hsize(cfg.to_dict())
                        if res is not None:
                            return res
                    except Exception:
                        pass
            return None

        configs = [
            getattr(m, "config", None),
            getattr(getattr(m, "base_model", None), "config", None),
            getattr(getattr(getattr(m, "base_model", None), "model", None), "config", None),
            getattr(getattr(m, "model", None), "config", None),
        ]
        for c in configs:
            val = search_hsize(c)
            if val is not None:
                return val

        # 2. Try standard HF embedding methods
        for method_name in ("get_input_embeddings", "get_output_embeddings"):
            if hasattr(m, method_name):
                try:
                    layer = getattr(m, method_name)()
                    if layer is not None:
                        if hasattr(layer, "embedding_dim"):
                            return layer.embedding_dim
                        if hasattr(layer, "in_features"):
                            return layer.in_features
                        if hasattr(layer, "out_features"):
                            return layer.out_features
                except Exception:
                    pass

        # 3. Check modules directly
        for name, mod in m.named_modules():
            name_lower = name.lower()
            if "embed_tokens" in name_lower or "wte" in name_lower:
                if hasattr(mod, "embedding_dim"):
                    return mod.embedding_dim
                if hasattr(mod, "in_features"):
                    return mod.in_features
            if "lm_head" in name_lower:
                if hasattr(mod, "in_features"):
                    return mod.in_features

        # 4. Check parameters shape
        for p in m.parameters():
            if p.dim() >= 2:
                return p.shape[-1]

        raise RuntimeError("Cannot resolve hidden size")

    @staticmethod
    def _vsize(m: nn.Module) -> int:
        # 1. Search recursively in configs
        def search_vsize(cfg) -> int | None:
            if cfg is None:
                return None
            if isinstance(cfg, dict):
                if "vocab_size" in cfg and isinstance(cfg["vocab_size"], int):
                    return cfg["vocab_size"]
                for v in cfg.values():
                    res = search_vsize(v)
                    if res is not None:
                        return res
            else:
                if hasattr(cfg, "vocab_size") and isinstance(getattr(cfg, "vocab_size"), int):
                    return getattr(cfg, "vocab_size")
                for attr in ("text_config", "vision_config", "config"):
                    if hasattr(cfg, attr):
                        res = search_vsize(getattr(cfg, attr))
                        if res is not None:
                            return res
                if hasattr(cfg, "to_dict"):
                    try:
                        res = search_vsize(cfg.to_dict())
                        if res is not None:
                            return res
                    except Exception:
                        pass
            return None

        configs = [
            getattr(m, "config", None),
            getattr(getattr(m, "base_model", None), "config", None),
            getattr(getattr(getattr(m, "base_model", None), "model", None), "config", None),
            getattr(getattr(m, "model", None), "config", None),
        ]
        for c in configs:
            val = search_vsize(c)
            if val is not None:
                return val

        # 2. Try standard HF embedding methods
        for method_name in ("get_input_embeddings", "get_output_embeddings"):
            if hasattr(m, method_name):
                try:
                    layer = getattr(m, method_name)()
                    if layer is not None:
                        if hasattr(layer, "num_embeddings"):
                            return layer.num_embeddings
                        if hasattr(layer, "out_features"):
                            return layer.out_features
                except Exception:
                    pass

        # 3. Check modules directly (e.g. for quantized layers)
        for name, mod in m.named_modules():
            name_lower = name.lower()
            if "lm_head" in name_lower or "wte" in name_lower:
                if hasattr(mod, "out_features"):
                    return mod.out_features
                if hasattr(mod, "num_embeddings"):
                    return mod.num_embeddings
            if "embed_tokens" in name_lower:
                if hasattr(mod, "num_embeddings"):
                    return mod.num_embeddings
                if hasattr(mod, "out_features"):
                    return mod.out_features

        # 4. Check class names
        for _, mod in m.named_modules():
            if "lm_head" in mod.__class__.__name__:
                if hasattr(mod, "out_features"):
                    return mod.out_features
                if hasattr(mod, "weight") and getattr(mod.weight, "shape", None) is not None:
                    return mod.weight.shape[0]

        # 5. Check parameters shape
        for p_name, p in m.named_parameters():
            if ("lm_head" in p_name or "embed_tokens" in p_name or "wte" in p_name) and p.dim() == 2:
                return p.shape[0]

        raise RuntimeError("Cannot resolve vocab size")

    def attach(self):
        self.detach()
        self._attached = True
        for li in self._resolved:

            def mk(idx, mgr):
                def fn(m, a, o):
                    if mgr.training:
                        mgr._captured[idx] = o[0] if isinstance(o, tuple) else o

                return fn

            self._hooks.append(self._decoder[li].register_forward_hook(mk(li, self)))

        def ph(m, a, o):
            if self._attached and self.training and self._labels is not None:
                self._compute()

        self._hooks.append(self._decoder[-1].register_forward_hook(ph))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks, self._captured, self._labels, self._results = [], {}, None, {}
        self._attached = False

    def set_labels(self, labels: torch.Tensor):
        self._labels = labels
        self._captured, self._results = {}, {}

    def _compute(self):
        if self._labels is None:
            return
        for li, hi in self.layer_to_head.items():
            if li in self._captured:
                self._results[hi] = self.shortcut_heads[hi](
                    self._captured[li], self._labels
                )

    def shortcut_loss(self) -> torch.Tensor:
        if not self._results and self._labels is not None and self._captured:
            self._compute()
        if not self._results:
            d = next(self.parameters()).device
            return torch.tensor(0.0, device=d, dtype=torch.float32)
        device = next(iter(self._results.values()))["loss"].device
        t = torch.tensor(0.0, device=device)
        for r in self._results.values():
            if r["loss_weight"] > 0 and r["loss"].item() > 0:
                t = t + self.config.loss_weight * r["loss_weight"] * r["loss"]
        return t.to(next(self.parameters()).dtype)

    def shortcut_loss_breakdown(self) -> list[dict]:
        bd = []
        for hi, r in self._results.items():
            bd.append(
                {
                    "head_idx": hi,
                    "layer": self._resolved[hi],
                    "offset": self.shortcut_heads[hi].offset,
                    "loss": r["loss"].item(),
                    "weighted_loss": (
                        self.config.loss_weight * r["loss_weight"] * r["loss"]
                    ).item(),
                    "valid_tokens": r["valid_tokens"],
                }
            )
        return bd

    def freeze(self):
        for h in self.shortcut_heads:
            h.freeze()

    def unfreeze(self):
        for h in self.shortcut_heads:
            h.unfreeze()

    def total_params(self):
        return sum(h.num_params() for h in self.shortcut_heads)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _find_decoder(model: nn.Module) -> tuple[nn.Module, int]:
    """Find decoder layer list. Returns (decoder_module, num_layers)."""
    # 1. Try prioritized hardcoded paths first for speed and precision
    paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("base_model", "model", "model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("base_model", "model", "layers"),
        ("base_model", "model", "model", "model", "blocks"),
        ("base_model", "model", "model", "blocks"),
        ("base_model", "model", "blocks"),
        ("generation_transformer", "model", "layers"),
        ("generation_transformer", "transformer", "h"),
    ]
    for path in paths:
        obj = model
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        else:
            if obj is not None:
                class_name = type(obj).__name__
                if (isinstance(obj, (nn.ModuleList, nn.Sequential))
                    or "ModuleList" in class_name
                    or "Sequential" in class_name
                    or "LayerList" in class_name
                    or "BlockList" in class_name):
                    try:
                        return obj, len(obj)
                    except Exception:
                        pass

    # 2. Dynamic fallback: Search named_modules for matching nn.ModuleList / nn.Sequential
    num_layers = None
    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
        getattr(getattr(model, "model", None), "config", None),
    ]
    cfg = None
    for c in configs:
        if c:
            cfg = c
            break

    if cfg:
        for attr in ("num_hidden_layers", "num_layers", "n_layer", "num_blocks", "n_blocks", "n_layers"):
            if hasattr(cfg, attr):
                num_layers = getattr(cfg, attr)
                break

    candidates = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        is_list_like = (
            isinstance(module, (nn.ModuleList, nn.Sequential))
            or "ModuleList" in class_name
            or "Sequential" in class_name
            or "LayerList" in class_name
            or "BlockList" in class_name
        )
        if is_list_like:
            try:
                l = len(module)
                if l > 0:
                    candidates.append((name, module, l))
            except Exception:
                pass

    if num_layers is not None:
        for name, module, l in candidates:
            if l == num_layers:
                logger.info("Found decoder layer list via config num_layers matching at '%s' (length %d)", name, l)
                return module, l

    for name, module, l in candidates:
        if l >= 8:
            lower_name = name.lower()
            if any(k in lower_name for k in ("layer", "h", "block", "dec")):
                logger.info("Found decoder layer list via naming pattern matching at '%s' (length %d)", name, l)
                return module, l

    for name, module, l in candidates:
        if l >= 8:
            logger.info("Found decoder layer list via fallback length check at '%s' (length %d)", name, l)
            return module, l

    raise RuntimeError(f"Cannot locate decoder in {type(model).__name__}.")



def build_shortcut_manager(
    model: nn.Module, config: ShortcutHeadConfig
) -> ShortcutManager | None:
    """Build ShortcutManager. Returns None if disabled or unsupported."""
    if not config.enabled:
        return None
    try:
        decoder, nl = _find_decoder(model)
    except RuntimeError as e:
        logger.warning("Shortcut heads disabled: %s", e)
        return None
    resolved = []
    for idx in config.layer_indices:
        r = nl + idx if idx < 0 else idx
        if not (0 <= r < nl):
            raise ValueError(f"Shortcut layer {idx} (resolved={r}) out of [0,{nl}).")
        resolved.append(r)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate resolved indices: {resolved}")
    logger.info(
        "Shortcut Heads: %d head(s), layers=%s, offsets=%s, lambda=%.2f",
        len(resolved),
        resolved,
        config.offsets,
        config.loss_weight,
    )
    return ShortcutManager(model, config, decoder, resolved)


class ShortcutLossWrapper(nn.Module):
    """Wraps a causal LM to auto-add shortcut loss. Injects mgr.shortcut_loss() into output.loss."""

    def __init__(self, model: nn.Module, manager: ShortcutManager):
        super().__init__()
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "manager", manager)

    # ── Proxy model attributes that Unsloth's SFTTrainer expects ────────────
    # Any attribute not found on the wrapper (config, torch_dtype, methods,
    # etc.) is forwarded to self.model, making the wrapper transparent.

    def __getattr__(self, name):
        obj = self.model  # self.model is always on the instance
        return getattr(obj, name)

    def __setattr__(self, name, value):
        if (
            name in ("model", "manager")
            or name in ShortcutLossWrapper.__dict__
            or name in self.__dict__
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(self.model, name, value)

    def __dir__(self):
        return list(super().__dir__() + dir(self.model))

    def children(self):
        return self.model.children()

    def named_children(self):
        return self.model.named_children()

    def modules(self):
        return self.model.modules()

    def named_modules(self, memo=None, prefix=""):
        return self.model.named_modules(memo, prefix)

    def named_parameters(self, prefix="", recurse=True):
        return self.model.named_parameters(prefix, recurse)

    def parameters(self, recurse=True):
        return self.model.parameters(recurse)

    def _modules(self):
        return self.model._modules

    def _parameters(self):
        return self.model._parameters

    def _apply(self, fn, recurse=True):
        return self.model._apply(fn, recurse)

    def train(self, mode=True):
        return self.model.train(mode)

    def eval(self):
        return self.model.eval()

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def type(self, dst_type):
        return self.model.type(dst_type)

    def half(self):
        return self.model.half()

    def float(self):
        return self.model.float()

    def double(self):
        return self.model.double()

    def cpu(self):
        return self.model.cpu()

    def cuda(self, device=None):
        return self.model.cuda(device)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        labels = kwargs.get("labels", None)
        self.manager.set_labels(labels)
        out = self.model(*args, **kwargs)
        if self.training:
            sc = self.manager.shortcut_loss()
            if hasattr(out, "loss") and out.loss is not None and sc.item() > 0:
                out.loss = out.loss + sc
        return out

    def generate(self, *args, **kwargs):
        """Passthrough to model.generate() — shortcuts are not used during generation."""
        return self.model.generate(*args, **kwargs)


def wrap_model(model: nn.Module, mgr: ShortcutManager) -> nn.Module:
    """Wrap model to inject shortcut loss automatically by monkey-patching forward. Idempotent."""
    # Move and cast manager to match model's device and dtype
    try:
        p = next(model.parameters())
        device = p.device
        dtype = p.dtype
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

    if isinstance(mgr, nn.Module):
        mgr.to(device=device, dtype=dtype)

    # If it is already monkey-patched, just update the manager and return
    if hasattr(model, "_hml_is_shortcut_wrapped") and model._hml_is_shortcut_wrapped:
        if isinstance(mgr, nn.Module):
            model.add_module("_hml_shortcut_manager", mgr)
        model._hml_shortcut_manager = mgr
        return model

    # If it is an old ShortcutLossWrapper, handle it gracefully
    if isinstance(model, ShortcutLossWrapper):
        model.manager = mgr
        return model

    # Store the original forward if we haven't already
    if not hasattr(model, "_hml_shortcut_original_forward"):
        model._hml_shortcut_original_forward = model.forward

    # Define the custom forward
    def custom_forward(self, *args, **kwargs):
        labels = kwargs.get("labels", None)
        self._hml_shortcut_manager.set_labels(labels)
        
        # Call the original forward
        out = self._hml_shortcut_original_forward(*args, **kwargs)
        
        if self.training:
            sc = self._hml_shortcut_manager.shortcut_loss()
            if hasattr(out, "loss") and out.loss is not None and sc.item() > 0:
                out.loss = out.loss + sc
        return out

    import types
    model.forward = types.MethodType(custom_forward, model)
    if isinstance(mgr, nn.Module):
        model.add_module("_hml_shortcut_manager", mgr)
    model._hml_shortcut_manager = mgr
    model._hml_is_shortcut_wrapped = True
    return model


def unwrap_model(wrapped: nn.Module) -> nn.Module:
    """Unwrap/restore the monkey-patched forward method or get the underlying model of a ShortcutLossWrapper."""
    if isinstance(wrapped, ShortcutLossWrapper):
        if wrapped.manager is not None:
            wrapped.manager.detach()
        return wrapped.model

    if hasattr(wrapped, "_hml_is_shortcut_wrapped") and wrapped._hml_is_shortcut_wrapped:
        if hasattr(wrapped, "_hml_shortcut_manager") and wrapped._hml_shortcut_manager is not None:
            wrapped._hml_shortcut_manager.detach()
            if "_hml_shortcut_manager" in wrapped._modules:
                del wrapped._modules["_hml_shortcut_manager"]
            if hasattr(wrapped, "_hml_shortcut_manager"):
                delattr(wrapped, "_hml_shortcut_manager")
        if hasattr(wrapped, "_hml_shortcut_original_forward"):
            wrapped.forward = wrapped._hml_shortcut_original_forward
            del wrapped._hml_shortcut_original_forward
        wrapped._hml_is_shortcut_wrapped = False
    return wrapped

