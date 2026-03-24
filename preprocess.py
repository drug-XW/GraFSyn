"""基础预处理脚本。

说明：你当前上传的 notebook 里没有原始 Graphlet 生成过程，
所以这里提供的是“训练前数据整理/校验版 preprocess.py”：
1. 检查主表中的 Drug_A / Drug_B / Cell_Line 是否都能在特征表里找到；
2. 过滤掉缺失样本；
3. 导出一份可直接用于训练的 clean CSV。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def preprocess_data(
    main_csv: str,
    graphlet_csv: str,
    cell_csv: str,
    output_csv: str,
    drug_a_col: str = "Drug_A",
    drug_b_col: str = "Drug_B",
    cell_col: str = "Cell_Line",
    graphlet_id_col: str = "PubChem_CID",
    cell_id_col: str = "Cell_Line",
) -> None:
    main_df = pd.read_csv(main_csv)
    graphlet_df = pd.read_csv(graphlet_csv)
    cell_df = pd.read_csv(cell_csv)

    valid_drug_ids = set(graphlet_df[graphlet_id_col].tolist())
    valid_cell_ids = set(cell_df[cell_id_col].tolist())

    mask = (
        main_df[drug_a_col].isin(valid_drug_ids)
        & main_df[drug_b_col].isin(valid_drug_ids)
        & main_df[cell_col].isin(valid_cell_ids)
    )

    clean_df = main_df.loc[mask].copy()
    dropped = len(main_df) - len(clean_df)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_csv, index=False)

    print(f"原始样本数: {len(main_df)}")
    print(f"保留样本数: {len(clean_df)}")
    print(f"过滤样本数: {dropped}")
    print(f"清洗后文件已保存到: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练前数据校验与清洗")
    parser.add_argument("--main_csv", default="data/Merk/ONEIL_SCORE_Processed.csv")
    parser.add_argument("--graphlet_csv", default="data/Graphlet_features_6.csv")
    parser.add_argument("--cell_csv", default="data/Merk/ONEIL_CELL_LINE_EXPRESSION.csv")
    parser.add_argument("--output_csv", default="data/Merk/ONEIL_SCORE_Processed_clean.csv")
    args = parser.parse_args()

    preprocess_data(
        main_csv=args.main_csv,
        graphlet_csv=args.graphlet_csv,
        cell_csv=args.cell_csv,
        output_csv=args.output_csv,
    )
