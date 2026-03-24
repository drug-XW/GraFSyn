"""模型定义：GraphCrossSynergy 及其子模块。"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def drug_feat(tensor: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """将输入整理成 [batch, seq_len, feature_dim] 并构造 attention mask。"""
    if device is None:
        device = tensor.device

    tensor = tensor.float().to(device)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)

    mask = torch.ones(tensor.size(0), 1, device=device)
    expanded_mask = (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -10000.0
    return tensor, expanded_mask


class DynamicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_list: Sequence[int] = (3, 5, 7)):
        super().__init__()
        self.kernel_list = list(kernel_list)
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2)
                for k in self.kernel_list
            ]
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, len(self.kernel_list)),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) != self.in_channels:
            x = x.permute(0, 2, 1)

        attn_weights = self.attention(x).unsqueeze(-1).unsqueeze(-1)

        features = [conv(x).unsqueeze(1) for conv in self.conv_layers]
        features = torch.cat(features, dim=1)
        output = torch.sum(features * attn_weights, dim=1)
        return output


class CNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dynamic_conv = DynamicConv1d(1, 16, kernel_list=(3, 5, 7))
        self.conv_fuse = nn.Conv1d(16, 64, kernel_size=1)

        self.shortcut = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
            nn.BatchNorm1d(64),
        )

        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(hidden_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if x.dim() == 3 and x.size(1) == 1 else x.unsqueeze(1)

        x = self.relu(self.dynamic_conv(x))
        x = self.relu(self.conv_fuse(x))

        shortcut = self.shortcut(identity)
        x = self.relu(x + shortcut)

        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, variance_epsilon: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        return self.gamma * x + self.beta


class SelfAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        self.output_projection = nn.Linear(self.all_head_size, hidden_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        context_layer = self.output_projection(context_layer)
        return context_layer, attention_probs_0


class CrossAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)
        self.output_projection = nn.Linear(self.all_head_size, hidden_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_input: torch.Tensor,
        key_value_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed_query_layer = self.query(query_input)
        mixed_key_layer = self.key(key_value_input)
        mixed_value_layer = self.value(key_value_input)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        context_layer = self.output_projection(context_layer)
        return context_layer, attention_probs_0


class SelfOutput(nn.Module):
    def __init__(self, hidden_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class CrossOutput(nn.Module):
    def __init__(self, hidden_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size: int, hidden_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class AttentionD(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
    ):
        super().__init__()
        self.cnn = CNNLayer(input_size, hidden_size)
        self.self_attention = SelfAttention(
            hidden_size, hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(
        self, input_tensor: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cnn_output = self.cnn(input_tensor)
        self_output, attention_probs_0 = self.self_attention(cnn_output, attention_mask)
        attention_output = self.output(self_output, cnn_output)
        return attention_output, attention_probs_0


class AttentionCD(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
    ):
        super().__init__()
        self.cross_attention = CrossAttention(
            input_size, hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.output = CrossOutput(hidden_size, hidden_dropout_prob)

    def forward(
        self,
        drug: torch.Tensor,
        cell: torch.Tensor,
        drug_attention_mask: torch.Tensor,
        cell_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        drug_self_output, drug_attention_probs_0 = self.cross_attention(drug, cell, drug_attention_mask)
        cell_self_output, cell_attention_probs_0 = self.cross_attention(cell, drug, cell_attention_mask)
        drug_attention_output = self.output(drug_self_output, drug)
        cell_attention_output = self.output(cell_self_output, cell)
        return (
            drug_attention_output,
            cell_attention_output,
            drug_attention_probs_0,
            cell_attention_probs_0,
        )


class Intermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.relu(self.dense(hidden_states))


class SelfEncoderD(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
    ):
        super().__init__()
        self.attention = AttentionD(
            input_size,
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
        )
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_probs_0 = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs_0


class EncoderCD(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
    ):
        super().__init__()
        self.attention_dc = AttentionCD(
            hidden_size,
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
        )
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(
        self,
        drug: torch.Tensor,
        cell: torch.Tensor,
        drug_attention_mask: torch.Tensor,
        cell_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        drug_attention_output, cell_attention_output, drug_attention_probs_0, cell_attention_probs_0 = self.attention_dc(
            drug, cell, drug_attention_mask, cell_attention_mask
        )
        drug_intermediate_output = self.intermediate(drug_attention_output)
        drug_layer_output = self.output(drug_intermediate_output, drug_attention_output)
        cell_intermediate_output = self.intermediate(cell_attention_output)
        cell_layer_output = self.output(cell_intermediate_output, cell_attention_output)
        return (
            drug_layer_output,
            cell_layer_output,
            drug_attention_probs_0,
            cell_attention_probs_0,
        )


class DynamicWeighting(nn.Module):
    """保留自 notebook 的动态权重模块，当前主干模型未直接使用。"""

    def __init__(self, input_size: int, num_features: int = 4):
        super().__init__()
        self.num_features = num_features
        self.input_size = input_size
        self.weight_calculator = nn.Sequential(
            nn.Linear(input_size * num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_features),
            nn.Softmax(dim=1),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        combined_for_weights = torch.cat([f1, f2, f3, f4], dim=-1)
        weights = self.weight_calculator(combined_for_weights).unsqueeze(2)
        features_stack = torch.stack([f1, f2, f3, f4], dim=1)
        weighted_sum_features = torch.sum(features_stack * weights, dim=1)
        return weighted_sum_features


class PreNN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class GraphCrossSynergy(nn.Module):
    def __init__(
        self,
        input_size_drug: int = 4937,
        input_size_cell: int = 977,
        hidden_self_size: int = 256,
        hidden_cross_size: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        intermediate_size = hidden_self_size * 2

        self.self_attention_drug_a = SelfEncoderD(
            input_size_drug, hidden_self_size, intermediate_size, num_heads, dropout, dropout
        )
        self.self_attention_drug_b = SelfEncoderD(
            input_size_drug, hidden_self_size, intermediate_size, num_heads, dropout, dropout
        )
        self.self_attention_cell = SelfEncoderD(
            input_size_cell, hidden_self_size, intermediate_size, num_heads, dropout, dropout
        )

        self.cross_attention_drug_a_cell = AttentionCD(
            hidden_self_size, hidden_cross_size, num_heads, dropout, dropout
        )
        self.cross_attention_drug_b_cell = AttentionCD(
            hidden_self_size, hidden_cross_size, num_heads, dropout, dropout
        )

        self.pre = PreNN(hidden_cross_size * 4)

    def forward(self, drug_a: torch.Tensor, drug_b: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        device = drug_a.device
        drug_a, drug_a_mask = drug_feat(drug_a.unsqueeze(1), device)
        drug_b, drug_b_mask = drug_feat(drug_b.unsqueeze(1), device)
        cell, cell_mask = drug_feat(cell.unsqueeze(1), device)

        drug_a_features, _ = self.self_attention_drug_a(drug_a, drug_a_mask)
        drug_b_features, _ = self.self_attention_drug_b(drug_b, drug_b_mask)
        cell_features, _ = self.self_attention_cell(cell, cell_mask)

        drug_a_cell_features, cell_a_attention_output, _, _ = self.cross_attention_drug_a_cell(
            drug_a_features, cell_features, drug_a_mask, cell_mask
        )
        drug_b_cell_features, cell_b_attention_output, _, _ = self.cross_attention_drug_b_cell(
            drug_b_features, cell_features, drug_b_mask, cell_mask
        )

        combined_features = torch.cat(
            [
                drug_a_cell_features.squeeze(1),
                cell_a_attention_output.squeeze(1),
                drug_b_cell_features.squeeze(1),
                cell_b_attention_output.squeeze(1),
            ],
            dim=-1,
        )
        return self.pre(combined_features)
