from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.model import EnhancedSSAPeriocularNet
from train_eval import SEED, UBIPrMultiTaskDataset, build_ubipr_manifest, evaluate_model


def compute_far_frr_at_eer(y_true_binary: np.ndarray, y_scores: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    fnr = 1.0 - tpr
    valid = np.isfinite(thresholds)
    fpr_v, fnr_v, thr_v = fpr[valid], fnr[valid], thresholds[valid]

    eer_idx = int(np.argmin(np.abs(fpr_v - fnr_v)))
    far = float(fpr_v[eer_idx])
    frr = float(fnr_v[eer_idx])
    opt_thresh = float(thr_v[eer_idx])
    auc_val = float(roc_auc_score(y_true_binary, y_scores))
    return far, frr, opt_thresh, auc_val, fpr_v, fnr_v, tpr[valid], thr_v


def main():
    base_dir = Path(__file__).resolve().parent
    ckpt_path = base_dir / "checkpoints" / "ssa_periocular_best.pt"
    gal_path = base_dir / "gallery" / "ubipr_gallery_templates.npz"
    res_dir = base_dir / "results"
    res_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Computing MCC, FAR, FRR & ROC-AUC on {device}...")

    df, _ = build_ubipr_manifest(base_dir / "data" / "ubipr")
    _, test_df = train_test_split(
        df, test_size=0.20, stratify=df["identity_label"], random_state=SEED
    )

    test_loader = DataLoader(
        UBIPrMultiTaskDataset(test_df, is_train=False),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EnhancedSSAPeriocularNet(
        num_identities=ckpt["num_identities"],
        backbone_name=ckpt.get("backbone", "resnet34"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    m = evaluate_model(model, test_loader, device)

    # 1. MCC for Gender & Closed-Set Person Identification
    gender_mcc = float(matthews_corrcoef(m["g_trues"], m["g_preds"]))
    id_mcc = float(matthews_corrcoef(m["id_trues"], m["id_preds"]))

    # 2. Gender FAR (False Positive Rate) & FRR (False Negative Rate)
    g_far, g_frr, g_thr, g_auc, _, _, _, _ = compute_far_frr_at_eer(
        m["g_trues"], m["g_probs"][:, 1]
    )

    # 3. Person Identification Open-Set Verification (Genuine vs. Impostor Pairs)
    gal_data = np.load(gal_path, allow_pickle=True)
    templates = gal_data["templates"]  # [N_gallery, 512]
    num_classes = templates.shape[0]

    # Cosine similarity matrix: [N_test_probes, N_gallery]
    sim_scores = m["embeddings"] @ templates.T
    pair_labels = (m["id_trues"][:, None] == np.arange(num_classes)[None, :]).astype(int)

    id_far, id_frr, id_thr, id_ver_auc, fpr_id, fnr_id, tpr_id, thr_id = compute_far_frr_at_eer(
        pair_labels.ravel(), sim_scores.ravel()
    )


    objective_df = pd.DataFrame([
        {
            "Task": "Person Identification (Proposed Hybrid SSA-ArcFace)",
            "Accuracy (%)": round(m["id_acc"] * 100, 2),
            "Precision (%)": round(m["id_precision"] * 100, 2),
            "Recall (%)": round(m["id_recall"] * 100, 2),
            "F1-Score (%)": round(m["id_f1"] * 100, 2),
            "MCC": round(id_mcc, 4),
            "FAR (%)": round(id_far * 100, 3),
            "FRR (%)": round(id_frr * 100, 3),
            "ROC-AUC": round(id_ver_auc, 4),
            "Optimal Threshold": round(id_thr, 4),
        },
        {
            "Task": "Gender Classification (Proposed Hybrid SSA-ArcFace)",
            "Accuracy (%)": round(m["gender_acc"] * 100, 2),
            "Precision (%)": round(m["gender_precision"] * 100, 2),
            "Recall (%)": round(m["gender_recall"] * 100, 2),
            "F1-Score (%)": round(m["gender_f1"] * 100, 2),
            "MCC": round(gender_mcc, 4),
            "FAR (%)": round(g_far * 100, 3),
            "FRR (%)": round(g_frr * 100, 3),
            "ROC-AUC": round(g_auc, 4),
            "Optimal Threshold": round(g_thr, 4),
        },
    ])
    objective_df.to_csv(res_dir / "complete_objective_metrics.csv", index=False)
    print("\n Complete Objective Evaluation Table:")
    print(objective_df.to_string(index=False))

    # Plot 1: Person Identification Verification ROC Curve
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr_id, tpr_id, color="#1f77b4", lw=2, label=f"Identity Verification (AUC = {id_ver_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        xlabel="False Acceptance Rate (FAR)",
        ylabel="Genuine Acceptance Rate (1 - FRR)",
        title="Person Identification ROC Curve",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res_dir / "identity_verification_roc.png", dpi=150)
    plt.close(fig)

    # Plot 2: Biometric FAR vs FRR Crossover (Equal Error Rate) Curve
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thr_id, fpr_id * 100, label="FAR (%)", color="crimson", lw=2)
    ax.plot(thr_id, fnr_id * 100, label="FRR (%)", color="royalblue", lw=2)
    ax.axvline(id_thr, color="darkgreen", linestyle="--", label=f"EER Threshold = {id_thr:.2f} ({id_far*100:.2f}%)")
    ax.set(
        xlabel="Cosine Similarity Threshold",
        ylabel="Error Rate (%)",
        title="Biometric FAR vs. FRR Trade-off Curve",
        xlim=(0.0, 1.0),
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res_dir / "far_frr_eer_curve.png", dpi=150)
    plt.close(fig)
    print(f"\n📁 Saved complete_objective_metrics.csv, identity_verification_roc.png, and far_frr_eer_curve.png to {res_dir}")


# Overwrite base_paper_comparison.csv with 100% genuine, unmanipulated test metrics
    bp_clean = pd.DataFrame([
        {"Model": "SNN (Base Paper Ref)", "Task": "Person Identification", "Accuracy (%)": 98.41, "Precision (%)": 97.22, "Recall (%)": 96.59, "F1-Score (%)": 96.91},
        {"Model": "DAA-CNN (Base Paper)", "Task": "Person Identification", "Accuracy (%)": 99.20, "Precision (%)": 98.05, "Recall (%)": 97.36, "F1-Score (%)": 97.71},
        {"Model": "PHCSN (Base Paper)", "Task": "Person Identification", "Accuracy (%)": 99.24, "Precision (%)": 99.24, "Recall (%)": 99.12, "F1-Score (%)": 99.18},
        {"Model": "SSA-RTNet (Base Paper)", "Task": "Person Identification", "Accuracy (%)": 99.80, "Precision (%)": 99.61, "Recall (%)": 99.56, "F1-Score (%)": 99.58},
        {
            "Model": "Proposed Hybrid SSA-ArcFace (Ours)",
            "Task": "Person Identification",
            "Accuracy (%)": round(m["id_acc"] * 100, 2),
            "Precision (%)": round(m["id_precision"] * 100, 2),
            "Recall (%)": round(m["id_recall"] * 100, 2),
            "F1-Score (%)": round(m["id_f1"] * 100, 2),
        },
        {"Model": "DAA-CNN (Base Paper)", "Task": "Gender Classification", "Accuracy (%)": 98.80, "Precision (%)": 98.05, "Recall (%)": 97.36, "F1-Score (%)": 97.71},
        {"Model": "PHCSN (Base Paper)", "Task": "Gender Classification", "Accuracy (%)": 99.28, "Precision (%)": 98.55, "Recall (%)": 98.27, "F1-Score (%)": 98.41},
        {"Model": "SSA-RTNet (Base Paper)", "Task": "Gender Classification", "Accuracy (%)": 99.40, "Precision (%)": 99.64, "Recall (%)": 99.59, "F1-Score (%)": 99.59},
        {
            "Model": "Proposed Hybrid SSA-ArcFace (Ours)",
            "Task": "Gender Classification",
            "Accuracy (%)": round(m["gender_acc"] * 100, 2),
            "Precision (%)": round(m["gender_precision"] * 100, 2),
            "Recall (%)": round(m["gender_recall"] * 100, 2),
            "F1-Score (%)": round(m["gender_f1"] * 100, 2),
        },
    ])
    bp_clean.to_csv(res_dir / "base_paper_comparison.csv", index=False)

if __name__ == "__main__":
    main()