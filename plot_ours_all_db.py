import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams["font.family"] = "Times New Roman"

model_name = "Ours"
dataset_list = ["Ours", "FYO", "PUT"]
dataset_display_names = {
    "Ours": "NTUST-IB811",
    "FYO": "FYO",
    "PUT": "PUT"
}

results_base_dir = "results"
K_FOLDS = 6

def load_roc_det_eer_data(model_name, dataset_name, fold_idx):
    base_dir = os.path.join(results_base_dir, model_name, "curve", dataset_name)
    roc_file_path = os.path.join(base_dir, f"{model_name}_fold_{fold_idx}_roc_curve_data.txt")
    det_file_path = os.path.join(base_dir, f"{model_name}_fold_{fold_idx}_det_curve_data.txt")
    eer_file_path = os.path.join(base_dir, f"{model_name}_fold_{fold_idx}_det_eer_data.txt")

    if not (os.path.exists(roc_file_path) and os.path.exists(det_file_path) and os.path.exists(eer_file_path)):
        print(f"[Warning] Missing file(s) for {model_name} fold={fold_idx}, dataset={dataset_name}")
        return None

    # 讀取 ROC 檔案 (fpr, tpr)
    fpr_list, tpr_list = [], []
    with open(roc_file_path, "r", encoding="cp950") as rf:
        lines = rf.readlines()
        for line in lines[1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        fpr_val = float(parts[0])
                        tpr_val = float(parts[1])
                        fpr_list.append(fpr_val)
                        tpr_list.append(tpr_val)
                    except ValueError:
                        continue
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)

    # 讀取 DET 檔案 (fpr, fnr)
    fpr_list2, fnr_list = [], []
    with open(det_file_path, "r", encoding="cp950") as df:
        lines = df.readlines()
        for line in lines[1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        fpr2_val = float(parts[0])
                        fnr_val = float(parts[1])
                        fpr_list2.append(fpr2_val)
                        fnr_list.append(fnr_val)
                    except ValueError:
                        continue
    fpr_det = np.array(fpr_list2)
    fnr_det = np.array(fnr_list)

    eer_val = None
    with open(eer_file_path, "r", encoding="cp950") as ef:
        for line in ef:
            if line.strip().startswith("EER:"):
                try:
                    eer_val = float(line.split(":")[1].strip())
                except ValueError:
                    pass

    if eer_val is None:
        print(f"[Warning] EER not found in {eer_file_path}, set to None.")
        return None

    fpr_array *= 100
    tpr_array *= 100
    fpr_det *= 100
    fnr_det *= 100

    return (fpr_array, tpr_array, fpr_det, fnr_det, eer_val)

for fold_idx in range(K_FOLDS):
    print(f"\nPlotting fold {fold_idx} for all datasets")

    dataset_results = []

    for dataset_name in dataset_list:
        data_tuple = load_roc_det_eer_data(model_name, dataset_name, fold_idx)
        if data_tuple is None:
            print(f"[Warning] {model_name}: no data for dataset={dataset_name}, fold={fold_idx}, skip.")
            continue

        fpr, tpr, fpr_det, fnr_det, eer_val = data_tuple
        dataset_results.append({
            "dataset_name": dataset_name,
            "fpr": fpr,
            "tpr": tpr,
            "fpr_det": fpr_det,
            "fnr_det": fnr_det,
            "eer": eer_val
        })

    if len(dataset_results) == 0:
        print(f"[Warning] No valid data for fold={fold_idx}, skip plotting.")
        continue

    print(f"Plotting DET Curve for fold {fold_idx}...")
    plt.figure(figsize=(5, 5))

    for res in dataset_results:
        dataset_name = res["dataset_name"]
        display_name = dataset_display_names[dataset_name]
        fpr_ = res["fpr_det"]
        fnr_ = res["fnr_det"]
        eer_ = res["eer"]

        plt.plot(fpr_, fnr_, label=f"{display_name} (EER={eer_:.2%})")
        plt.scatter(eer_ * 100, eer_ * 100, s=15, color="black", zorder=10)

    plt.plot([0, 100], [0, 100], linestyle="--", color="black")
    plt.xlabel("False Accepted Rate (%)")
    plt.ylabel("False Rejected Rate (%)")

    eer_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                               markersize=5, label='EER')
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [eer_marker],
               loc="upper right")

    plt.tight_layout()
    out_det_path = os.path.join(results_base_dir, f"Ours_Model_All_Datasets_DET_Comparison_fold_{fold_idx}.svg")
    plt.savefig(out_det_path, format="svg")
    plt.close()
    print(f"DET Comparison for fold {fold_idx} saved {out_det_path}")
