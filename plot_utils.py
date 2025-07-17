import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix

plt.rcParams["font.family"] = "Times New Roman"

def compute_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean([fpr[min_index], fnr[min_index]])
    return eer, thresholds[min_index]

def plot_det_curve(fpr, fnr, eer, label=None):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, fnr, label=label)
    eer_index = np.nanargmin(np.absolute(fpr - fnr))
    eer_fpr = fpr[eer_index]
    eer_fnr = fnr[eer_index]
    plt.scatter(eer_fpr, eer_fnr, color="black", label=f"EER = {eer:.6f}", s=20)
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    plt.xlabel("False Accepted Rate (%)")
    plt.ylabel("False Rejected Rate (%)")
    # plt.title("DET Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    # plt.show()

def plot_roc_curve(far, tpr, eer, label=None):
    roc_auc = auc(far, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(far, tpr, label=f"AUC = {roc_auc:.6f})")
    eer_index = np.nanargmin(np.absolute(far - (1 - tpr)))
    plt.scatter(far[eer_index], tpr[eer_index], color="black", label=f"EER = {eer:.6f}", s=20)
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate (%)")
    plt.ylabel("True Positive Rate (%)")
    # plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    # plt.show()

def calculate_confusion_matrix(labels, predictions, threshold):
    pred_labels = (predictions > threshold).astype(int)
    conf_matrix = confusion_matrix(labels, pred_labels)
    return conf_matrix

def plot_confusion_matrix(conf_matrix, classes, cmap=plt.cm.Blues):
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    # plt.title(Confusion matrix)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.tight_layout()
    # plt.show()

def plot_density(genuine_scores, impostor_scores):
    sns.kdeplot(genuine_scores, bw_adjust=2, label="Genuine", fill=True, color="#0000cc", linewidth=1.5)
    sns.kdeplot(impostor_scores, bw_adjust=2, label="Impostor", fill=True, color="#228B22", linewidth=1.5)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Probability Density")
    plt.ylim(bottom=-0.1, top=None)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    
def plot_models_on_multiple_datasets_det(all_results, model_name, out_path):
    plt.figure(figsize=(5,5))
    for item in all_results:
        ds   = item["dataset_name"]
        fpr  = item["fpr"]
        tpr  = item["tpr"]
        eer  = item["eer"]
        fnr  = 1 - tpr
        plt.plot(fpr, fnr, label=f"{ds} (EER={eer:6f})")
        plt.scatter(eer, eer, color="black", s=15)

    plt.plot([0,1],[0,1],'--', color='gray')
    # plt.title(f"DET curves ({model_name}) on multiple datasets")
    plt.xlabel("False Accepted Rate")
    plt.ylabel("False Rejected Rate")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved multi-datasets DET curve to {out_path}")

def plot_models_on_multiple_datasets_roc(all_results, model_name, out_path):
    plt.figure(figsize=(5,5))
    for item in all_results:
        ds   = item["dataset_name"]
        fpr  = item["fpr"]
        tpr  = item["tpr"]
        eer  = item["eer"]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{ds} (AUC={roc_auc:.6f}, EER={eer:.6%})")
        plt.scatter(eer, 1-eer, color="black", s=15)

    plt.plot([0,1], [0,1], '--', color='gray')
    # plt.title(f"ROC curves ({model_name}) on multiple datasets")
    plt.xlabel("False Positive Rate (%)")
    plt.ylabel("True Positive Rate (%)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved multi datasets ROC curve to {out_path}")
