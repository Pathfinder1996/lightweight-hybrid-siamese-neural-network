import os
import pickle
import matplotlib.pyplot as plt

# Use a consistent font for all plots
plt.rcParams["font.family"] = "Times New Roman"

# Define your model names
# network_names = ["Ours", 
#                  "ResNet18",
#                  "ResNet34",
#                  "ResNet50", 
#                  "MobileNetV1", 
#                  "MobileNetV2", 
#                  "MobileNetV3_Small", 
#                  "EfficientNetB0",
#                  "EfficientNetB1",
#                  "VGG16",
#                  "VGG19"
#                  ] 

network_names = ["Ours",
                "SE PRE Block",
                "SE POST Block",
                "SE Identity Block",
                "Without SE Block",
                ]

# Total number of K-Folds used during training
n_folds = 5

# Directory where training results are stored
result_root = "results"

# Output directory for plots
output_dir = os.path.join(result_root, "summary_plots")
os.makedirs(output_dir, exist_ok=True)

# Loop over each fold
for fold in range(1, n_folds + 1):
    print(f"Processing Fold {fold}")

    # ==== Training Accuracy ====
    plt.figure(figsize=(5, 5))
    for net in network_names:
        print(f"Processing network: {net}")
        if net == "Ours":
            continue
        pkl_path = os.path.join(result_root, net, f"{net}_history_fold_{fold}.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            hist = pickle.load(f)
        acc_scaled = [v * 100 for v in hist["accuracy"]]
        plt.plot(acc_scaled, label=net, zorder=1)
    # Ours
    ours_path = os.path.join(result_root, "Ours", f"Ours_history_fold_{fold}.pkl")
    if os.path.exists(ours_path):
        with open(ours_path, "rb") as f:
            hist = pickle.load(f)
        acc_scaled = [v * 100 for v in hist["accuracy"]]
        plt.plot(acc_scaled, label="Ours", color="black", linewidth=3, zorder=10)
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy (%)")
    plt.title(f"Training Accuracy - Fold {fold}")
    plt.ylim(70, 101)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fold_{fold}_training_accuracy.svg"), format="svg")
    plt.close()

    # ==== Validation Accuracy ====
    plt.figure(figsize=(5, 5))
    for net in network_names:
        if net == "Ours":
            continue
        pkl_path = os.path.join(result_root, net, f"{net}_history_fold_{fold}.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            hist = pickle.load(f)
        if "val_accuracy" in hist:
            val_acc_scaled = [v * 100 for v in hist["val_accuracy"]]
            plt.plot(val_acc_scaled, label=net, zorder=1)
    if os.path.exists(ours_path):
        with open(ours_path, "rb") as f:
            hist = pickle.load(f)
        if "val_accuracy" in hist:
            val_acc_scaled = [v * 100 for v in hist["val_accuracy"]]
            plt.plot(val_acc_scaled, label="Ours", color="black", linewidth=3, zorder=10)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(f"Validation Accuracy - Fold {fold}")
    plt.ylim(70, 101)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fold_{fold}_validation_accuracy.svg"), format="svg")
    plt.close()

    # ==== Training Loss ====
    plt.figure(figsize=(5, 5))
    for net in network_names:
        if net == "Ours":
            continue
        pkl_path = os.path.join(result_root, net, f"{net}_history_fold_{fold}.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            hist = pickle.load(f)
        plt.plot(hist["loss"], label=net, zorder=1)
    if os.path.exists(ours_path):
        with open(ours_path, "rb") as f:
            hist = pickle.load(f)
        plt.plot(hist["loss"], label="Ours", color="black", linewidth=3, zorder=10)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss - Fold {fold}")
    plt.ylim(-0.1, 5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fold_{fold}_training_loss.svg"), format="svg")
    plt.close()

    # ==== Validation Loss ====
    plt.figure(figsize=(5, 5))
    for net in network_names:
        if net == "Ours":
            continue
        pkl_path = os.path.join(result_root, net, f"{net}_history_fold_{fold}.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            hist = pickle.load(f)
        if "val_loss" in hist:
            plt.plot(hist["val_loss"], label=net, zorder=1)
    if os.path.exists(ours_path):
        with open(ours_path, "rb") as f:
            hist = pickle.load(f)
        if "val_loss" in hist:
            plt.plot(hist["val_loss"], label="Ours", color="black", linewidth=3, zorder=10)
    plt.xlabel("Epochs")
    plt.ylabel("Contrastive Loss")
    plt.title(f"Validation Loss - Fold {fold}")
    plt.ylim(0, 0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fold_{fold}_validation_loss.svg"), format="svg")
    plt.close()

print("\nAll plots generated successfully! Check the folder:", output_dir)
