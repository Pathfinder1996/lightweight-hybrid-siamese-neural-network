import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams["font.family"] = "Times New Roman"

# network_names = ["Ours",
#                  "Ours Without SE Block"]

network_names = ["Ours",
                 "ResNet18",
                 "MobileNetV1", 
                 "MobileNetV2", 
                 "MobileNetV3",
                 "EfficientNetB0"
                 ]

dataset_list = ["Ours", 
                "FYO", 
                "PUT"
                ]

results_base_dir = "results"

def parse_best_fold_index(k_fold_result_path):
    """
    Parse Best fold index (int) from k_fold_results.txt.
    Returns None if not found.
    """
    if not os.path.exists(k_fold_result_path):
        print(f"[Warning] {k_fold_result_path} not found.")
        return None

    best_fold_idx = None
    with open(k_fold_result_path, "r", encoding="cp950") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith("Best fold index"):
                parts = line.strip().split(":")
                if len(parts) == 2:
                    best_fold_str = parts[1].strip()
                    try:
                        best_fold_idx = int(best_fold_str)
                    except ValueError:
                        pass
                break
    return best_fold_idx

def load_roc_det_eer_data(model_name, dataset_name, fold_idx):
    """
    Load ROC (fpr, tpr) / DET (fpr, fnr) / EER data from specified fold file.
    Returns: (fpr, tpr, fnr, eer); Returns None if file or values are missing.
    """
    base_dir = os.path.join(results_base_dir, model_name, "curve", dataset_name)

    roc_file_path = os.path.join(base_dir, f"{model_name}_fold_{fold_idx}_roc_curve_data.txt")
    det_file_path = os.path.join(base_dir, f"{model_name}_fold_{fold_idx}_det_curve_data.txt")
    eer_file_path = os.path.join(base_dir, f"{model_name}_fold_{fold_idx}_det_eer_data.txt")

    if not (os.path.exists(roc_file_path) and os.path.exists(det_file_path) and os.path.exists(eer_file_path)):
        print(f"[Warning] Missing file(s) for {model_name} fold={fold_idx}, dataset={dataset_name}")
        return None

    # Read ROC file (fpr, tpr)
    fpr_list = []
    tpr_list = []
    with open(roc_file_path, "r", encoding="cp950") as rf:
        lines = rf.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    fpr_val = float(parts[0])
                    tpr_val = float(parts[1])
                    fpr_list.append(fpr_val)
                    tpr_list.append(tpr_val)
                except ValueError:
                    pass
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)

    # Read DET file (fpr, fnr)
    fnr_list = []
    fpr_list2 = []
    with open(det_file_path, "r", encoding="cp950") as df:
        lines = df.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    fpr2_val = float(parts[0])
                    fnr_val  = float(parts[1])
                    fpr_list2.append(fpr2_val)
                    fnr_list.append(fnr_val)
                except ValueError:
                    pass
    fpr_det = np.array(fpr_list2)
    fnr_det = np.array(fnr_list)

    # Read EER
    eer_val = None
    with open(eer_file_path, "r", encoding="cp950") as ef:
        lines = ef.readlines()
        for line in lines:
            if line.strip().startswith("EER:"):
                try:
                    parts = line.split(":")
                    if len(parts) == 2:
                        eer_str = parts[1].strip()
                        eer_val = float(eer_str)
                except ValueError:
                    pass

    if eer_val is None:
        print(f"[Warning] EER not found in {eer_file_path}, set to None.")
        return None

    # Multiply by 100
    fpr_array *= 100
    tpr_array *= 100
    fpr_det *= 100
    fnr_det *= 100

    return (fpr_array, tpr_array, fpr_det, fnr_det, eer_val)

# Main program: Plot multi-model comparison for each dataset
for dataset_name in dataset_list:
    print(f"\nPlotting for dataset: {dataset_name}")

    comparison_results = []

    for model_name in network_names:
        k_fold_result_path = os.path.join(results_base_dir, model_name, f"{model_name}_k_fold_results.txt")
        best_fold_idx = parse_best_fold_index(k_fold_result_path)
        if best_fold_idx is None:
            print(f"[Warning] {model_name}: best fold not found, skip.")
            continue

        data_tuple = load_roc_det_eer_data(model_name, dataset_name, best_fold_idx)
        if data_tuple is None:
            print(f"[Warning] {model_name}: no data for best fold={best_fold_idx}, skip.")
            continue

        fpr, tpr, fpr_det, fnr_det, eer_val = data_tuple

        comparison_results.append({
            "model_name": model_name,
            "fpr": fpr,
            "tpr": tpr,
            "fpr_det": fpr_det,
            "fnr_det": fnr_det,
            "eer": eer_val
        })

    if len(comparison_results) == 0:
        print(f"[Warning] No valid model data for {dataset_name}, skip plotting.")
        continue

    # Plot ROC based on comparison_results
    print("Plotting ROC Curve...")
    plt.figure(figsize=(5, 5))
    for res in comparison_results:
        model_name = res["model_name"]
        fpr_ = res["fpr"]
        tpr_ = res["tpr"]
        eer_ = res["eer"]
        # Data already multiplied by 100, so AUC needs to consider scale (divide by 10000 = 100 * 100)
        roc_auc = np.trapz(tpr_, fpr_) / 10000
        plt.plot(fpr_, tpr_, label=f"{model_name} (AUC={roc_auc:.2%})")

    plt.plot([0, 100], [0, 100], linestyle="--", color="black")
    plt.xlabel("False Positive Rate (%)")
    plt.ylabel("True Positive Rate (%)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_roc_path = os.path.join(results_base_dir, f"All_Model_{dataset_name}_ROC_Comparison.svg")
    plt.savefig(out_roc_path, format="svg")
    plt.close()
    print(f"ROC Comparison saved {out_roc_path}")

    # Plot DET Curve
    print("Plotting DET Curve...")
    plt.figure(figsize=(5, 5))
    for res in comparison_results:
        model_name = res["model_name"]
        fpr_ = res["fpr_det"]
        fnr_ = res["fnr_det"]
        eer_ = res["eer"]
        # Plot DET
        plt.plot(fpr_, fnr_, label=f"{model_name} (EER={eer_:.2%})")
        plt.scatter(eer_*100, eer_*100, s=15, color="black", zorder=10)

    plt.plot([0, 100], [0, 100], linestyle="--", color="black")
    plt.xlabel("False Accepted Rate (%)")
    plt.ylabel("False Rejected Rate (%)")

    eer_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            markersize=5, label='EER')

    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [eer_marker],
            loc="upper right")

    plt.tight_layout()
    out_det_path = os.path.join(results_base_dir, f"All_Model_{dataset_name}_DET_Comparison.svg")
    plt.savefig(out_det_path, format="svg")
    plt.close()
    print(f"DET Comparison saved {out_det_path}")
