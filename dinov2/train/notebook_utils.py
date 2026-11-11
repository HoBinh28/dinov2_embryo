# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Helpers to run DINOv2 SSL training from a Jupyter notebook on a single GPU
(no torchrun / submitit / command line), and to continue from official DINOv2 weights.
"""

import logging
import math
import os
import re
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

import dinov2.distributed as distributed
from dinov2.configs import dinov2_default_config
from dinov2.hub.utils import _DINOV2_BASE_URL, _make_dinov2_model_name, _safe_load_state_dict_from_url
from dinov2.logging import setup_logging
from dinov2.utils import utils
from dinov2.utils.config import apply_scaling_rules_to_cfg, write_config


logger = logging.getLogger("dinov2")

_LOGGING_READY = False


def make_cfg(overrides: Dict[str, Any], output_dir: str) -> DictConfig:
    """Default SSL config merged with `overrides` (nested dict); sets train.output_dir."""
    output_dir = os.path.abspath(output_dir)
    cfg = OmegaConf.merge(OmegaConf.create(dinov2_default_config), OmegaConf.create(overrides))
    cfg.train.output_dir = output_dir
    return cfg


def setup_single_gpu(cfg: DictConfig, seed: int = 0) -> DictConfig:
    """
    Notebook equivalent of `dinov2.utils.config.setup`: creates the output dir, sets up
    logging/seed, applies the lr scaling rule and saves config.yaml.
    With train.use_fsdp=true it also starts a 1-process NCCL group (FSDP needs one);
    with train.use_fsdp=false (recommended on 1 GPU / recent torch) no process group is created.
    Safe to call again after a cell re-run.
    """
    global _LOGGING_READY
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    if cfg.train.get("use_fsdp", True):
        if not distributed.is_enabled():
            distributed.enable(overwrite=True)
        if distributed.get_global_size() != 1:
            logger.warning(f"expected 1 process, got world size {distributed.get_global_size()}")
    else:
        torch.cuda.set_device(0)

    if not _LOGGING_READY:
        setup_logging(output=cfg.train.output_dir, level=logging.INFO)
        _LOGGING_READY = True

    utils.fix_random_seeds(seed)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, cfg.train.output_dir)
    return cfg


def official_weights_url(cfg: DictConfig) -> str:
    """URL of the official LVD-142M checkpoint matching cfg.student (arch, patch size, registers)."""
    base = _make_dinov2_model_name(cfg.student.arch, cfg.student.patch_size)
    full = _make_dinov2_model_name(cfg.student.arch, cfg.student.patch_size, cfg.student.num_register_tokens)
    return f"{_DINOV2_BASE_URL}/{base}/{full}_pretrain.pth"


def _extract_backbone_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Accepts an official checkpoint (plain state dict), {"model": ...}, or our {"teacher": ...} files."""
    for key in ("teacher", "model", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    out = {}
    for k, v in ckpt.items():
        k = k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        if k.startswith("backbone."):
            k = k[len("backbone.") :]
        elif k.startswith(("dino_head.", "ibot_head.")):
            continue
        out[k] = v
    return out


def _unchunk_block_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """blocks.{chunk}.{i}.xxx -> blocks.{i}.xxx"""
    pattern = re.compile(r"^blocks\.\d+\.(\d+)\.")
    return {pattern.sub(r"blocks.\1.", k): v for k, v in state_dict.items()}


def _chunk_block_keys(state_dict: Dict[str, torch.Tensor], depth: int, block_chunks: int) -> Dict[str, torch.Tensor]:
    """blocks.{i}.xxx -> blocks.{chunk}.{i}.xxx (layout used by DinoVisionTransformer with block_chunks > 0)."""
    chunksize = depth // block_chunks
    pattern = re.compile(r"^blocks\.(\d+)\.")
    return {pattern.sub(lambda m: f"blocks.{int(m.group(1)) // chunksize}.{m.group(1)}.", k): v for k, v in state_dict.items()}


def _resize_pos_embed(pos_embed: torch.Tensor, n_target_tokens: int) -> torch.Tensor:
    """Bicubic resize of a (1, 1 + M*M, D) pos embedding to (1, n_target_tokens, D)."""
    n_src = pos_embed.shape[1] - 1
    n_dst = n_target_tokens - 1
    m_src, m_dst = int(math.sqrt(n_src)), int(math.sqrt(n_dst))
    assert m_src * m_src == n_src and m_dst * m_dst == n_dst
    cls_pos, patch_pos = pos_embed[:, :1], pos_embed[:, 1:]
    dim = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, m_src, m_src, dim).permute(0, 3, 1, 2).float()
    patch_pos = torch.nn.functional.interpolate(patch_pos, size=(m_dst, m_dst), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, m_dst * m_dst, dim).to(pos_embed.dtype)
    return torch.cat([cls_pos, patch_pos], dim=1)


def load_backbone_weights(backbone: torch.nn.Module, source: str, strict_core: bool = True) -> None:
    """
    Loads backbone weights into a (non FSDP-wrapped) DinoVisionTransformer.

    Args:
        source: local path or https URL of a checkpoint (official DINOv2 .pth, or a teacher_checkpoint.pth).
        strict_core: raise if any backbone parameter is missing or unexpected (recommended: True).
    """
    if source.startswith(("http://", "https://", "file://")):
        ckpt = _safe_load_state_dict_from_url(source, map_location="cpu")
    else:
        ckpt = torch.load(source, map_location="cpu")
    state_dict = _unchunk_block_keys(_extract_backbone_state_dict(ckpt))

    if getattr(backbone, "chunked_blocks", False):
        depth = backbone.n_blocks
        block_chunks = len(backbone.blocks)
        state_dict = _chunk_block_keys(state_dict, depth, block_chunks)

    target_pos = backbone.pos_embed
    if "pos_embed" in state_dict and state_dict["pos_embed"].shape != target_pos.shape:
        logger.info(f"resizing pos_embed {tuple(state_dict['pos_embed'].shape)} -> {tuple(target_pos.shape)}")
        state_dict["pos_embed"] = _resize_pos_embed(state_dict["pos_embed"], target_pos.shape[1])

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    logger.info(f"loaded backbone weights from {source}")
    if missing:
        logger.warning(f"missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        logger.warning(f"unexpected keys ({len(unexpected)}): {unexpected[:10]}")
    if strict_core and (missing or unexpected):
        raise RuntimeError(
            "pretrained weights do not match the backbone (check student.arch / patch_size / "
            "num_register_tokens / ffn_layer). Missing: %s Unexpected: %s" % (missing[:10], unexpected[:10])
        )


def find_teacher_checkpoints(output_dir: str):
    """Sorted list of (iteration, path) for the teacher checkpoints saved during training."""
    eval_dir = os.path.join(output_dir, "eval")
    if not os.path.isdir(eval_dir):
        return []
    found = []
    for name in os.listdir(eval_dir):
        path = os.path.join(eval_dir, name, "teacher_checkpoint.pth")
        m = re.search(r"(\d+)$", name)
        if m and os.path.isfile(path):
            found.append((int(m.group(1)), path))
    return sorted(found)


def export_teacher_backbone(teacher_checkpoint: str, out_path: str) -> str:
    """
    Converts a training `teacher_checkpoint.pth` into a plain backbone state dict with the same
    key layout as the official DINOv2 weights (no chunking, no 'backbone.' prefix), loadable with
    e.g. `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False).load_state_dict(...)`
    when student.img_size == 518, or with `build_backbone_from_cfg` otherwise.
    """
    ckpt = torch.load(teacher_checkpoint, map_location="cpu")
    state_dict = _unchunk_block_keys(_extract_backbone_state_dict(ckpt))
    state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(state_dict, out_path)
    logger.info(f"exported teacher backbone ({len(state_dict)} tensors) to {out_path}")
    return out_path


def build_backbone_from_cfg(cfg: DictConfig, weights: Optional[str] = None) -> torch.nn.Module:
    """Builds the teacher backbone (eval mode, block_chunks=0) and optionally loads weights."""
    from dinov2.models import build_model

    student_cfg = OmegaConf.create(OmegaConf.to_container(cfg.student, resolve=True))
    student_cfg.block_chunks = 0
    img_size = cfg.student.get("img_size", None) or cfg.crops.global_crops_size
    backbone, _ = build_model(student_cfg, only_teacher=True, img_size=img_size)
    if weights:
        load_backbone_weights(backbone, weights)
    return backbone.eval()
