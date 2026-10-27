"""Versioned, actor-only fusion of the five existing observation modalities."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from supermimic import networks as robot_networks
from .observations import MATCHING

MLP_ACTOR = "mlp"
FUSION_ACTOR = "five_token_fusion_v1"
ACTOR_ARCHITECTURES = (MLP_ACTOR, FUSION_ACTOR)
TOKEN_INPUTS = (("robot_history", 930), ("human_future", 448),
                ("object_history", 90), ("object_future", 63), ("shape", 64))


def validate_actor_architecture(architecture, observation_profile):
    if architecture not in ACTOR_ARCHITECTURES:
        raise ValueError(f"unknown actor architecture: {architecture}")
    if architecture == FUSION_ACTOR and observation_profile != MATCHING:
        raise ValueError("five-token fusion requires the v1.1 observation profile")
    return architecture


def actor_architecture_contract(architecture, observation_profile):
    validate_actor_architecture(architecture, observation_profile)
    if architecture == MLP_ACTOR:
        return {"name": MLP_ACTOR}
    return {
        "name": FUSION_ACTOR,
        "version": "supermimic_hoi_v1_2_actor_fusion_1",
        "observation_profile": observation_profile,
        "tokens": [{"name": name, "input_dim": dim} for name, dim in TOKEN_INPUTS],
        "tokenizer": "independent_linear_projections_of_normalized_raw_inputs;shape64_unnormalized",
        "token_dim": 256, "layers": 2, "heads": 4, "feedforward_dim": 1024,
        "activation": "GELU", "pre_layer_norm": True, "final_layer_norm": True,
        "dropout": 0., "modality_embeddings": "learned_5x256",
        "position_encoding": "none;time_slots_fixed_inside_each_flattened_modality",
        "attention": "joint_bidirectional_self_attention;no_causal_or_padding_mask",
        "readout": ["robot_history", "object_history"], "readout_dim": 512,
        "decoder_input": ["unchanged_v1_1_prefix1148", "fusion512"], "decoder_input_dim": 1660,
        "initialization": "retain_original_mlp_initialization;zero_only_new_decoder_columns",
        "added_modules_rng": "fork_restore_cpu_rng;preserve_subsequent_WM_critic_initialization",
        "geometry_disabled": "zero_shape64_before_both_paths",
        "precision": "cuda_bf16_autocast_when_enabled;otherwise_input_dtype;output_input_dtype",
        "objective": "unchanged_differentiable_imagined_return",
    }


class FiveTokenFusion(nn.Module):
    """Grouped modality attention; time order remains inside each input vector.

    Inputs have already passed through the existing actor RMS/clipping. No
    running statistics or persistent temporal/attention caches live here.
    """

    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleDict({name: nn.Linear(dim, 256) for name, dim in TOKEN_INPUTS})
        self.modality_embedding = nn.Parameter(torch.empty(5, 256))
        nn.init.normal_(self.modality_embedding, std=.02)
        # Construct each layer independently rather than cloning one initial
        # set of TransformerEncoderLayer parameters into both layers.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(256, 4, dim_feedforward=1024, dropout=0.,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(2)
        ])
        self.output_norm = nn.LayerNorm(256)

    def forward(self, robot_history, human_future, object_history, object_future, shape):
        values = (robot_history, human_future, object_history, object_future, shape)
        leading = robot_history.shape[:-1]
        for (name, dim), value in zip(TOKEN_INPUTS, values):
            if value.shape != leading + (dim,):
                raise ValueError(f"fusion {name} must share the batch shape and have {dim} features")
        amp = robot_networks.AMP_DTYPE
        context = (torch.autocast("cuda", dtype=amp)
                   if robot_history.is_cuda and amp is not None else nullcontext())
        with context:
            tokens = torch.stack([self.projections[name](value)
                                  for (name, _), value in zip(TOKEN_INPUTS, values)], -2)
            tokens = tokens + self.modality_embedding
            tokens = tokens.reshape(-1, 5, 256)
            for layer in self.layers:
                tokens = layer(tokens)
            tokens = self.output_norm(tokens)
            fused = torch.cat((tokens[:, 0], tokens[:, 2]), -1).reshape(leading + (512,))
        return fused.to(dtype=robot_history.dtype)
