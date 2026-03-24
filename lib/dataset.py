"""数据加载：DrugDataset 与 CSV -> TensorDataset 构建逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DrugDataset(Dataset):
    def __init__(self, drug_a: torch.Tensor, drug_b: torch.Tensor, cell: torch.Tensor, target: torch.Tensor):
        self.drug_a = drug_a
        self.drug_b = drug_b
        self.cell = cell
        self.target = target

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int):
        return self.drug_a[idx], self.drug_b[idx], self.cell[idx], self.target[idx]


@dataclass
class DatasetBuildInfo:
    total_samples: int
    valid_samples: int
    missing_samples: int
    drug_feature_dim: int
    cell_feature_dim: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "missing_samples": self.missing_samples,
            "drug_feature_dim": self.drug_feature_dim,
            "cell_feature_dim": self.cell_feature_dim,
        }


def _load_feature_tables(
    main_csv: str,
    graphlet_csv: str,
    cell_csv: str,
    graphlet_id_col: str = "PubChem_CID",
    graphlet_feature_start_col: int = 5,
    cell_id_col: str = "Cell_Line",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_data = pd.read_csv(main_csv)
    graphlet_features_data = pd.read_csv(graphlet_csv)
    cell_features_data = pd.read_csv(cell_csv)

    graphlet_features = graphlet_features_data.set_index(graphlet_id_col).iloc[:, graphlet_feature_start_col:]
    cell_features = cell_features_data.set_index(cell_id_col)
    return main_data, graphlet_features, cell_features


def build_dataset_from_csvs(
    main_csv: str,
    graphlet_csv: str,
    cell_csv: str,
    drug_a_col: str = "Drug_A",
    drug_b_col: str = "Drug_B",
    cell_col: str = "Cell_Line",
    label_col: str = "Label",
    graphlet_id_col: str = "PubChem_CID",
    graphlet_feature_start_col: int = 5,
    cell_id_col: str = "Cell_Line",
) -> Tuple[DrugDataset, DatasetBuildInfo]:
    """从三个 CSV 构造 DrugDataset。"""
    main_data, graphlet_features, cell_features = _load_feature_tables(
        main_csv=main_csv,
        graphlet_csv=graphlet_csv,
        cell_csv=cell_csv,
        graphlet_id_col=graphlet_id_col,
        graphlet_feature_start_col=graphlet_feature_start_col,
        cell_id_col=cell_id_col,
    )

    drug_a_graphlet_list = []
    drug_b_graphlet_list = []
    cell_feature_list = []
    labels = []

    missing_samples_count = 0
    total_samples_count = len(main_data)

    for _, row in main_data.iterrows():
        try:
            drug_a_cid = row[drug_a_col]
            drug_b_cid = row[drug_b_col]
            cell_line_name = row[cell_col]

            drug_a_graphlet = graphlet_features.loc[drug_a_cid].values
            drug_b_graphlet = graphlet_features.loc[drug_b_cid].values
            cell_line_features = cell_features.loc[cell_line_name].values

            drug_a_graphlet_list.append(drug_a_graphlet)
            drug_b_graphlet_list.append(drug_b_graphlet)
            cell_feature_list.append(cell_line_features)
            labels.append(row[label_col])
        except KeyError:
            missing_samples_count += 1
            continue

    if len(labels) == 0:
        raise ValueError(
            "所有样本都因特征缺失被跳过。请检查主表中的 Drug_A/Drug_B/Cell_Line 是否都能在特征表中找到。"
        )

    drug_a_graphlet_tensor = torch.tensor(np.asarray(drug_a_graphlet_list), dtype=torch.float32)
    drug_b_graphlet_tensor = torch.tensor(np.asarray(drug_b_graphlet_list), dtype=torch.float32)
    cell_features_tensor = torch.tensor(np.asarray(cell_feature_list), dtype=torch.float32)
    targets_tensor = torch.tensor(np.asarray(labels), dtype=torch.float32)

    dataset = DrugDataset(
        drug_a_graphlet_tensor,
        drug_b_graphlet_tensor,
        cell_features_tensor,
        targets_tensor,
    )

    info = DatasetBuildInfo(
        total_samples=total_samples_count,
        valid_samples=len(labels),
        missing_samples=missing_samples_count,
        drug_feature_dim=drug_a_graphlet_tensor.shape[1],
        cell_feature_dim=cell_features_tensor.shape[1],
    )
    return dataset, info
