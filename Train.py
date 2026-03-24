"""训练脚本：K-Fold 交叉验证、标准化、模型保存。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from lib.dataset import DrugDataset, build_dataset_from_csvs
from lib.models import GraphCrossSynergy
from utils.metrics import calculateScore


def plot_metrics(metric_history: List[List[float]], title: str, save_path: Path | None = None) -> None:
    plt.figure(figsize=(10, 6))
    for i, fold_values in enumerate(metric_history):
        plt.plot(range(1, len(fold_values) + 1), fold_values, label=f"Fold {i + 1}")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_scaled_subsets(
    dataset: DrugDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[DrugDataset, DrugDataset]:
    train_subset = Subset(dataset, train_idx)

    drug_a_train = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
    drug_b_train = torch.stack([train_subset[i][1] for i in range(len(train_subset))])
    cell_train = torch.stack([train_subset[i][2] for i in range(len(train_subset))])

    scaler_drug_a = StandardScaler()
    scaler_drug_b = StandardScaler()
    scaler_cell = StandardScaler()

    drug_a_train_scaled = scaler_drug_a.fit_transform(drug_a_train.numpy())
    drug_b_train_scaled = scaler_drug_b.fit_transform(drug_b_train.numpy())
    cell_train_scaled = scaler_cell.fit_transform(cell_train.numpy())

    drug_a_val_scaled = scaler_drug_a.transform(dataset.drug_a[val_idx].numpy())
    drug_b_val_scaled = scaler_drug_b.transform(dataset.drug_b[val_idx].numpy())
    cell_val_scaled = scaler_cell.transform(dataset.cell[val_idx].numpy())

    train_dataset_scaled = DrugDataset(
        torch.tensor(drug_a_train_scaled, dtype=torch.float32),
        torch.tensor(drug_b_train_scaled, dtype=torch.float32),
        torch.tensor(cell_train_scaled, dtype=torch.float32),
        dataset.target[train_idx],
    )
    val_dataset_scaled = DrugDataset(
        torch.tensor(drug_a_val_scaled, dtype=torch.float32),
        torch.tensor(drug_b_val_scaled, dtype=torch.float32),
        torch.tensor(cell_val_scaled, dtype=torch.float32),
        dataset.target[val_idx],
    )
    return train_dataset_scaled, val_dataset_scaled


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, Dict]:
    model.eval()
    val_loss = 0.0
    val_targets: List[float] = []
    val_probs: List[float] = []

    with torch.no_grad():
        for drug_a, drug_b, cell, target in loader:
            drug_a = drug_a.to(device)
            drug_b = drug_b.to(device)
            cell = cell.to(device)
            target = target.to(device)

            logits = model(drug_a, drug_b, cell).squeeze()
            loss = criterion(logits, target)
            probs = torch.sigmoid(logits)

            val_loss += loss.item()
            val_targets.extend(target.cpu().numpy().tolist())
            val_probs.extend(probs.cpu().numpy().tolist())

    avg_val_loss = val_loss / max(len(loader), 1)
    score = calculateScore(np.array(val_targets), np.array(val_probs))
    return avg_val_loss, score


def train_one_fold(
    fold: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    pos_weight_value: float,
    l2_lambda: float,
    early_stop_patience: int,
    min_delta: float,
    checkpoint_dir: Path,
) -> Tuple[Dict, List[float], List[float], List[float]]:
    pos_weight = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    epoch_train_losses: List[float] = []
    epoch_val_losses: List[float] = []
    epoch_val_accs: List[float] = []

    best_val_loss = float("inf")
    best_state = None
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for drug_a, drug_b, cell, target in train_loader:
            drug_a = drug_a.to(device)
            drug_b = drug_b.to(device)
            cell = cell.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(drug_a, drug_b, cell).squeeze()
            loss = criterion(logits, target)

            l2_reg = torch.tensor(0.0, device=device)
            for param in model.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param, 2)
            loss += l2_lambda * l2_reg

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_loss, epoch_score = evaluate_model(model, val_loader, device, criterion)

        epoch_train_losses.append(avg_train_loss)
        epoch_val_losses.append(avg_val_loss)
        epoch_val_accs.append(epoch_score["acc"])

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"ACC: {epoch_score['acc']:.4f} | "
            f"AUC: {epoch_score['AUC']:.4f}"
        )
        scheduler.step(avg_val_loss)

        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_dir / f"fold_{fold + 1}_best.pt")
        else:
            early_stop_counter += 1
            print(f"No significant improvement for {early_stop_counter} epoch(s).")
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_score = evaluate_model(model, val_loader, device, criterion)
    final_score["val_loss"] = final_val_loss
    return final_score, epoch_train_losses, epoch_val_losses, epoch_val_accs


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, dataset_info = build_dataset_from_csvs(
        main_csv=args.main_csv,
        graphlet_csv=args.graphlet_csv,
        cell_csv=args.cell_csv,
        graphlet_feature_start_col=args.graphlet_feature_start_col,
    )
    print("Dataset info:", dataset_info.as_dict())

    labels = dataset.target.numpy() if isinstance(dataset.target, torch.Tensor) else dataset.target
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    train_losses: List[List[float]] = []
    val_losses: List[List[float]] = []
    val_accs: List[List[float]] = []
    final_scores: List[Dict] = []

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    figure_dir = Path(args.output_dir) / "figures"
    report_dir = Path(args.output_dir) / "reports"

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print("=" * 80)
        print(f"Fold {fold + 1}/{args.n_splits}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        print(f"Train class distribution: {np.bincount(labels[train_idx])}")
        print(f"Val class distribution: {np.bincount(labels[val_idx])}")

        train_dataset_scaled, val_dataset_scaled = create_scaled_subsets(dataset, train_idx, val_idx)
        train_loader = DataLoader(train_dataset_scaled, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_scaled, batch_size=args.batch_size, shuffle=False)

        model = GraphCrossSynergy(
            input_size_drug=dataset_info.drug_feature_dim,
            input_size_cell=dataset_info.cell_feature_dim,
            hidden_self_size=args.hidden_self_size,
            hidden_cross_size=args.hidden_cross_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)

        final_score, epoch_train_losses, epoch_val_losses, epoch_val_accs = train_one_fold(
            fold=fold,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            pos_weight_value=args.pos_weight,
            l2_lambda=args.l2_lambda,
            early_stop_patience=args.early_stop_patience,
            min_delta=args.min_delta,
            checkpoint_dir=checkpoint_dir,
        )

        final_scores.append(final_score)
        train_losses.append(epoch_train_losses)
        val_losses.append(epoch_val_losses)
        val_accs.append(epoch_val_accs)
        print(f"Fold {fold + 1} Final Score: {final_score}")

    summary_metrics = ["acc", "AUC", "AUC_prec_rec", "F1", "bacc", "precision", "sn", "sp", "kappa"]
    summary = {}
    for metric_name in summary_metrics:
        scores = [score[metric_name] for score in final_scores]
        summary[metric_name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
        print(f"Final Average {metric_name.upper()}: {summary[metric_name]['mean']:.3f} ± {summary[metric_name]['std']:.3f}")

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_info.as_dict(), "summary": summary, "folds": final_scores}, f, ensure_ascii=False, indent=2)

    plot_metrics(train_losses, "Training Loss", figure_dir / "training_loss.png")
    plot_metrics(val_losses, "Validation Loss", figure_dir / "validation_loss.png")
    plot_metrics(val_accs, "Validation ACC", figure_dir / "validation_acc.png")
    print(f"All outputs saved under: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCrossSynergy K-Fold training")
    parser.add_argument("--main_csv", default="data/Merk/ONEIL_SCORE_Processed.csv")
    parser.add_argument("--graphlet_csv", default="data/Graphlet_features_6.csv")
    parser.add_argument("--cell_csv", default="data/Merk/ONEIL_CELL_LINE_EXPRESSION.csv")
    parser.add_argument("--graphlet_feature_start_col", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--hidden_self_size", type=int, default=256)
    parser.add_argument("--hidden_cross_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pos_weight", type=float, default=4.5)
    parser.add_argument("--l2_lambda", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
