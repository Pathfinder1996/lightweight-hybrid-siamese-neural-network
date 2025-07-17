import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, recall_score, f1_score, auc

from my_metrics import contrastive_loss, EuclideanDistance
from models import get_network
from data_loader import load_image, create_labels
from plot_utils import plot_det_curve, plot_roc_curve, plot_density, plot_confusion_matrix, compute_eer

plt.rcParams["font.family"] = "Times New Roman"

# Data directory
image_dir1 = r"D:\ib811_wrist_vein_database\all\s1"
image_dir2 = r"D:\ib811_wrist_vein_database\all\s2"

images1 = load_image(image_dir1)
images2 = load_image(image_dir2)

# Make sure images1 and images2 have the same length
pairs_1, pairs_2, labels = create_labels(images1, images2)
labels = np.array(labels)

# K-Fold Cross Validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create results directory
if not os.path.exists("results"):
    os.makedirs("results")

# Define the datasets to be used
dataset_list = ["Ours", 
                "FYO", 
                "PUT"
                ]

network_names = ["Ours", 
                 "Ours Without SE Block", 
                 "ResNet18", 
                 "MobileNetV1", 
                 "MobileNetV2", 
                 "MobileNetV3", 
                 "EfficientNetB0"
                 ] 

for network_name in network_names:
    print(f"Training for network: {network_name}")

    result_dir = os.path.join("results", network_name)
    os.makedirs(result_dir, exist_ok=True)

    curve_dir = os.path.join(result_dir, "curve")
    os.makedirs(curve_dir, exist_ok=True)

    # Create a dictionary to store results for each dataset
    fold_results = { }
    for ds in dataset_list:
        fold_results[ds] = {
            "EER": [],
            "Precision": [],
            "Recall": [],
            "F1": []
        }

    # Initialize a dictionary to store the best fold results
    best_fold = {
        "index": -1,
        "avg_EER": np.inf,  
        "avg_precision": 0.0,
        "avg_recall": 0.0,
        "avg_f1": 0.0
    }

    # Store fold metrics
    fold_accuracy_list = []
    fold_val_accuracy_list = []
    fold_loss_list = []
    fold_val_loss_list = []
    
    # Start K-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(pairs_1, labels)):
        
        print(f"Training fold {fold + 1}/{n_splits} for {network_name}")

        # Divide data into training and validation sets
        train_pairs_1, val_pairs_1 = np.array(pairs_1)[train_idx], np.array(pairs_1)[val_idx]
        train_pairs_2, val_pairs_2 = np.array(pairs_2)[train_idx], np.array(pairs_2)[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        print("train_pairs_1:", train_pairs_1.shape)
        print("train_pairs_2:", train_pairs_2.shape)
        print("train_labels:", train_labels.shape)
        print("val_pairs_1:", val_pairs_1.shape)
        print("val_pairs_2:", val_pairs_2.shape)
        print("val_labels:", val_labels.shape)

        # Reshape pairs to match input shape
        trainPairs_0 = train_pairs_1.reshape(-1, 128, 128, 1)
        trainPairs_1 = train_pairs_2.reshape(-1, 128, 128, 1)
        valPairs_0   = val_pairs_1.reshape(-1, 128, 128, 1)
        valPairs_1   = val_pairs_2.reshape(-1, 128, 128, 1)

        # Build the Siamese network
        inputShape = (128, 128, 1)
        
        inputA = Input(inputShape)
        inputB = Input(inputShape)
        
        featureExtractor = get_network(network_name, inputShape)
        
        featsA = featureExtractor(inputA)
        featsB = featureExtractor(inputB)
        
        distance = EuclideanDistance()([featsA, featsB])
        
        model = Model(inputs=[inputA, inputB], outputs=distance)
        
        # Check if this is the first fold
        if fold == 0:
            # Calculate FLOPs for the model
            def get_flops(model, input_shape):
                # Create a concrete function for the model
                inputA = tf.TensorSpec([1] + list(input_shape), tf.float32)
                inputB = tf.TensorSpec([1] + list(input_shape), tf.float32)
                
                concrete_func = tf.function(lambda inputsA, inputsB: model([inputsA, inputsB]))
                concrete_func = concrete_func.get_concrete_function(inputA, inputB)

                # Use the profiler to calculate FLOPs
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

                flops = tf.compat.v1.profiler.profile(
                    graph=concrete_func.graph,
                    run_meta=run_meta,
                    options=opts
                )

                if flops is not None:
                    gflops = flops.total_float_ops / 1e9  # Transform to GFLOPs
                    return gflops
                else:
                    return None
            
            gflops = get_flops(model, inputShape)
            print(f"\nSubnet: {network_name} GFLOPs: {gflops:.6f}")

        # Compile the model with contrastive loss
        model.compile(
            loss=contrastive_loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )

        # Set up model checkpoint to save the best model based on validation loss
        model_checkpoint = ModelCheckpoint(
            os.path.join(result_dir, f"{network_name}_model_fold_{fold + 1}.hdf5"), 
            monitor="val_loss", 
            verbose=1, 
            save_best_only=True
        )

        # Train the model with the training and validation data
        history = model.fit(
            [trainPairs_0, trainPairs_1], train_labels,
            validation_data=([valPairs_0, valPairs_1], val_labels),
            batch_size=24,
            epochs=100,
            callbacks=[model_checkpoint]
        )

        # Save the training history to a pickle file
        with open(os.path.join(result_dir, f"{network_name}_history_fold_{fold + 1}.pkl"), "wb") as file_pi:
            pickle.dump(history.history, file_pi)

        # Get the final training and validation accuracy and loss
        final_train_acc  = history.history["accuracy"][-1]
        final_train_loss = history.history["loss"][-1]
        final_val_acc    = history.history["val_accuracy"][-1]
        final_val_loss   = history.history["val_loss"][-1]
        fold_accuracy_list.append(final_train_acc)
        fold_loss_list.append(final_train_loss)
        fold_val_accuracy_list.append(final_val_acc)
        fold_val_loss_list.append(final_val_loss)

        # Print final training and validation metrics
        if "accuracy" in history.history:
            plt.figure(figsize=(5, 5))
            plt.plot(history.history["accuracy"], label="Train")
            if "val_accuracy" in history.history:
                plt.plot(history.history["val_accuracy"], label="Validation")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            output_path = os.path.join(result_dir, f"{network_name}_accuracy_fold_{fold + 1}.svg")
            plt.savefig(output_path, format="svg")
            plt.close()

        # Plot training and validation loss
        plt.figure(figsize=(5, 5))
        plt.plot(history.history["loss"], label="Train")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Contrastive Loss")
        plt.legend()
        plt.tight_layout()
        output_path2 = os.path.join(result_dir, f"{network_name}_loss_fold_{fold + 1}.svg")
        plt.savefig(output_path2, format="svg")
        plt.close()

        # Load the best model weights for this fold
        best_ckpt_path = os.path.join(result_dir, f"{network_name}_model_fold_{fold + 1}.hdf5")
        model.load_weights(best_ckpt_path)
        
        # Convert the model to TFLite format
        tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = tflite_converter.convert()

        # Save the TFLite model
        tflite_model_path = os.path.join(result_dir, f"{network_name}_model_fold_{fold+1}.tflite")
        with open(tflite_model_path, "wb") as tflite_file:
            tflite_file.write(tflite_model)

        print(f"[TFLite] fold {fold + 1} model saved at {tflite_model_path}")

        # Calculate EER, Precision, Recall, F1 for each dataset
        fold_dataset_eer_list       = []
        fold_dataset_precision_list = []
        fold_dataset_recall_list    = []
        fold_dataset_f1_list        = []

        all_results = []

        for dataset_name in dataset_list:
            print(f"\nTesting dataset: {dataset_name}")

            # Test data directory
            test_image_dir1 = fr"D:\ib811_wrist_vein_database\all\test\{dataset_name}\s1"
            test_image_dir2 = fr"D:\ib811_wrist_vein_database\all\test\{dataset_name}\s2"

            test_images1 = load_image(test_image_dir1)
            test_images2 = load_image(test_image_dir2)
            test_pairs_1, test_pairs_2, test_labels = create_labels(test_images1, test_images2)
            test_labels = np.array(test_labels)

            testPairs_0 = test_pairs_1.reshape(-1, 128, 128, 1)
            testPairs_1r= test_pairs_2.reshape(-1, 128, 128, 1)
            
            dataset_curve_dir = os.path.join(curve_dir, dataset_name)
            os.makedirs(dataset_curve_dir, exist_ok=True)
            
            y_pred = model.predict([testPairs_0, testPairs_1r]).ravel()
            
            fpr, tpr, thresholds = roc_curve(test_labels, y_pred)
            eer, eer_threshold = compute_eer(fpr, tpr, thresholds)

            fold_dataset_eer_list.append(eer)

            # EER Threshold Binary Classification
            y_pred_binary = (y_pred >= eer_threshold).astype(int)
            precision = precision_score(test_labels, y_pred_binary)
            recall    = recall_score(test_labels, y_pred_binary)
            f1        = f1_score(test_labels, y_pred_binary)

            fold_dataset_precision_list.append(precision)
            fold_dataset_recall_list.append(recall)
            fold_dataset_f1_list.append(f1)

            # Store results for this dataset
            fold_results[dataset_name]["EER"].append(eer)
            fold_results[dataset_name]["Precision"].append(precision)
            fold_results[dataset_name]["Recall"].append(recall)
            fold_results[dataset_name]["F1"].append(f1)

            # Record results for plotting later
            all_results.append({
                "dataset": dataset_name,
                "fpr": fpr,
                "tpr": tpr,
                "eer": eer
            })

            # Plot and save results for this dataset
            with open(os.path.join(dataset_curve_dir, f"{network_name}_fold_{fold + 1}_det_eer_data.txt"), "w") as f:
                f.write(f"EER: {eer}\n")
                f.write(f"Threshold at EER: {eer_threshold}\n")
            
            # DET curve    
            with open(os.path.join(dataset_curve_dir, f"{network_name}_fold_{fold + 1}_det_curve_data.txt"), "w") as f:
                f.write("False Positive Rate(%), False Negative Rate(%), Thresholds\n")
                for i in range(len(fpr)):
                    f.write(f"{fpr[i]}, {1 - tpr[i]}, {thresholds[i]}\n")
            
            plot_det_curve(fpr, 1 - tpr, eer, label=dataset_name)
            plt.savefig(os.path.join(dataset_curve_dir, f"{network_name}_fold_{fold + 1}_det_curve.svg"), format="svg")
            plt.close()
            
            # ROC curve
            with open(os.path.join(dataset_curve_dir, f"{network_name}_fold_{fold + 1}_roc_curve_data.txt"), "w") as f:
                f.write("False Positive Rate(%), True Positive Rate(%), Thresholds\n")
                for i in range(len(fpr)):
                    f.write(f"{fpr[i]}, {tpr[i]}, {thresholds[i]}\n")    
            
            plot_roc_curve(fpr, tpr, eer)
            plt.savefig(os.path.join(dataset_curve_dir, f"{network_name}_fold_{fold + 1}_roc_curve.svg"), format="svg")
            plt.close()
            
            # Scores density
            genuine_scores = y_pred[test_labels == 0]
            impostor_scores = y_pred[test_labels == 1]
            
            with open(os.path.join(dataset_curve_dir, f"{network_name}_fold_{fold + 1}_scores_density_data.txt"), "w") as f:
                f.write("Score Type,Score\n")
                for score in genuine_scores:
                    f.write(f"Genuine,{score}\n")
                for score in impostor_scores:
                    f.write(f"Impostor,{score}\n")
            
            plot_density(genuine_scores, impostor_scores)
            plt.savefig(os.path.join(dataset_curve_dir, f"{network_name}_fold_{fold + 1}_scores_density.svg"), format="svg")
            plt.close()
            
            # Confusion matrix
            conf_matrix = confusion_matrix(test_labels, y_pred_binary)
            tn, fp, fn, tp = conf_matrix.ravel()
            
            with open(os.path.join(dataset_curve_dir, f"{network_name}_fold_{fold + 1}_confusion_matrix.txt"), "w") as f:
                f.write("Confusion Matrix\n")
                f.write(f"\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
                f.write("\nPrecision: {:.6f}\n".format(precision))
                f.write("Recall: {:.6f}\n".format(recall))
                f.write("F1: {:.6f}\n".format(f1))
            
            classes = ["Genuine", "Impostor"]
            plot_confusion_matrix(conf_matrix, classes)
            plt.savefig(os.path.join(dataset_curve_dir, f"{network_name}_fold_{fold + 1}_confusion_matrix.svg"), format="svg")
            plt.close()
            
            print(f"Done testing on dataset: {dataset_name}")

        # All datasets metrics results average for this fold
        fold_avg_eer       = np.mean(fold_dataset_eer_list)
        fold_avg_precision = np.mean(fold_dataset_precision_list)
        fold_avg_recall    = np.mean(fold_dataset_recall_list)
        fold_avg_f1        = np.mean(fold_dataset_f1_list)

        # if fold average EER is lower than the best fold, update best fold
        if fold_avg_eer < best_fold["avg_EER"]:
            best_fold["index"]         = fold + 1
            best_fold["avg_EER"]       = fold_avg_eer
            best_fold["avg_precision"] = fold_avg_precision
            best_fold["avg_recall"]    = fold_avg_recall
            best_fold["avg_f1"]        = fold_avg_f1

        # Plot and save fold results
        print("\nPlot all datasets ROC in a single figure")
        plt.figure(figsize=(5, 5))
        for result in all_results:
            ds   = result["dataset"]
            fpr_ = result["fpr"]
            tpr_ = result["tpr"]
            eer_ = result["eer"]
            roc_auc = auc(fpr_, tpr_)
            plt.plot(fpr_, tpr_, label=f"{ds} (AUC={roc_auc:.6%})")
            plt.scatter(eer_, 1 - eer_, color="black", s=15)
        plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
        plt.xlabel("False Positive Rate (%)")
        plt.ylabel("True Positive Rate (%)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        out_roc = os.path.join(curve_dir, f"{network_name}_fold_{fold+1}_allDatasets_ROC.svg")
        plt.savefig(out_roc, format="svg")
        plt.close()

        print("Plot all datasets DET in a single figure")
        plt.figure(figsize=(5, 5))
        for result in all_results:
            ds   = result["dataset"]
            fpr_ = result["fpr"]
            tpr_ = result["tpr"]
            eer_ = result["eer"]
            fnr_ = 1 - tpr_

            plt.plot(fpr_, fnr_, label=f"{ds} (EER={eer_:.6%})")
            plt.scatter(eer_, eer_, color="black", s=15)
        plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
        plt.xlabel("False Accepted Rate (%)")
        plt.ylabel("False Rejected Rate (%)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        out_det = os.path.join(curve_dir, f"{network_name}_fold_{fold+1}_allDatasets_DET.svg")
        plt.savefig(out_det, format="svg")
        plt.close()

        print("All datasets comparison plots saved for this fold\n")
              
    # Calculate the best fold based on average EER across all datasets
    print(f"\nCross-validation results for {network_name}")
    per_dataset_stats = {}

    for ds in dataset_list:
        ds_eer       = np.array(fold_results[ds]["EER"])
        ds_precision = np.array(fold_results[ds]["Precision"])
        ds_recall    = np.array(fold_results[ds]["Recall"])
        ds_f1        = np.array(fold_results[ds]["F1"])

        eer_mean, eer_std = ds_eer.mean(), ds_eer.std()
        pre_mean, pre_std = ds_precision.mean(), ds_precision.std()
        rec_mean, rec_std = ds_recall.mean(), ds_recall.std()
        f1_mean,  f1_std  = ds_f1.mean(), ds_f1.std()

        per_dataset_stats[ds] = {
            "eer_mean": eer_mean,
            "eer_std": eer_std,
            "pre_mean": pre_mean,
            "pre_std": pre_std,
            "rec_mean": rec_mean,
            "rec_std": rec_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
        }

        print(f"\nDataset: {ds}")
        print(f"  EER       : {eer_mean:.6f} ± {eer_std:.6f}")
        print(f"  Precision : {pre_mean:.6f} ± {pre_std:.6f}")
        print(f"  Recall    : {rec_mean:.6f} ± {rec_std:.6f}")
        print(f"  F1        : {f1_mean:.6f} ± {f1_std:.6f}")
    
    acc_mean = np.mean(fold_accuracy_list)
    acc_std  = np.std(fold_accuracy_list)
    loss_mean = np.mean(fold_loss_list)
    loss_std  = np.std(fold_loss_list)
    val_acc_mean = np.mean(fold_val_accuracy_list)
    val_acc_std  = np.std(fold_val_accuracy_list)
    val_loss_mean = np.mean(fold_val_loss_list)
    val_loss_std  = np.std(fold_val_loss_list)

    # Print the best fold results
    print(f"\nBest fold (by lowest average EER across {dataset_list}): {best_fold['index']}")
    print(f"Avg EER       : {best_fold['avg_EER']:.6f}")
    print(f"Avg Precision : {best_fold['avg_precision']:.6f}")
    print(f"Avg Recall    : {best_fold['avg_recall']:.6f}")
    print(f"Avg F1        : {best_fold['avg_f1']:.6f}")

    print(f"\nTraining final accuracy and loss (5-Fold)")
    print(f"Avg Training Accuracy   : {acc_mean:.6f} ± {acc_std:.6f}")
    print(f"Avg Validation Accuracy : {val_acc_mean:.6f} ± {val_acc_std:.6f}")
    print(f"Avg Training Loss       : {loss_mean:.6f} ± {loss_std:.6f}")
    print(f"Avg Validation Loss     : {val_loss_mean:.6f} ± {val_loss_std:.6f}")

    # Save the results to a text file
    out_txt = os.path.join(result_dir, f"{network_name}_k_fold_results.txt")
    with open(out_txt, "w") as f:
        f.write(f"5-Fold Results for {network_name}\n\n")
        for ds in dataset_list:
            stats = per_dataset_stats[ds]
            f.write(f"Dataset: {ds}\n")
            f.write(f"EER       : {stats['eer_mean']:.6f} ± {stats['eer_std']:.6f}\n")
            f.write(f"Precision : {stats['pre_mean']:.6f} ± {stats['pre_std']:.6f}\n")
            f.write(f"Recall    : {stats['rec_mean']:.6f} ± {stats['rec_std']:.6f}\n")
            f.write(f"F1        : {stats['f1_mean']:.6f} ± {stats['f1_std']:.6f}\n\n")

        f.write(f"Best fold by lowest average EER across all datasets\n")
        f.write(f"Best fold index      : {best_fold['index']}\n")
        f.write(f"Average EER          : {best_fold['avg_EER']:.6f}\n")
        f.write(f"Average Precision    : {best_fold['avg_precision']:.6f}\n")
        f.write(f"Average Recall       : {best_fold['avg_recall']:.6f}\n")
        f.write(f"Average F1           : {best_fold['avg_f1']:.6f}\n\n")
        
        # Record the training metrics averages and standard deviations
        f.write(f"5-Fold final training metrics\n")
        f.write(f"Avg Training Accuracy   : {acc_mean:.6f} ± {acc_std:.6f}\n")
        f.write(f"Avg Validation Accuracy : {val_acc_mean:.6f} ± {val_acc_std:.6f}\n")
        f.write(f"Avg Training Loss       : {loss_mean:.6f} ± {loss_std:.6f}\n")
        f.write(f"Avg Validation Loss     : {val_loss_mean:.6f} ± {val_loss_std:.6f}\n")

    print(f"\n5-Fold results saved to: {out_txt}\n")
