## Lightweight Hybrid Siamese Neural Network
This project contains the Python implementation used to train, validate, and test the proposed lightweight hybrid Siamese neural network on a personal computer.

The trained models can be exported to TFLite format for deployment on resource-constrained edge devices.

## Contents
- `benchmark_energy.py` - Measures GPU power consumption and energy per inference for all models.
- `best_model_score.py` - The scoring method follows the weighted normalization strategy proposed in Romano *et al*. [1], enabling fair comparison of models under energy-constrained edge-AI environments.
- `blocks.py` - Building blocks used in `models.py` (Conv, Depthwise Conv, Residual Block, Inverted Residual Block, SE block, etc.).
- `data_loader.py` - Dataset loader and label-pair generator for Siamese training.
- `labels_vis.py` - Visualizes positive and negative training pairs.
- `main.py` - Main training script.
- `models.py` - Network architectures (Ours, ResNet18, MobileNet family, and several custom experimental models).
- `multiobjective_scores_result.csv` - Multi-objective score of all models, computed using a combined Z-score and min-max normalization.
- `my_metrics.py` -  Custom evaluation metrics (contrastive loss, Euclidean distance, ROC/EER computation, etc.).
- `plot_all_model_db.py` - Visualization of ROC/DET curves for all models across all datasets.
- `plot_ours_all_db.py` - Visualization of the proposed Ours model over all folds and datasets.
- `plot_all_model_history.py` - Plots training and validation accuracy/loss curves for every model.
- `plot_utils.py` - Utility functions for metric computation and plotting (confusion matrix, DET curve, EER point, etc.).
- `requirements.txt` - Python 3.9.2 dependency list.

## Datasets
Three wrist-vein datasets were used for model training:
- NTUST-IB811 [2]: Collected using our imaging device [https://ieee-dataport.org/documents/ntust-ib811-wrist-vein-dataset](https://ieee-dataport.org/documents/ntust-ib811-wrist-vein-dataset)
- FYO [3]: Available upon request
- PUT [4]: Available upon request

## Model Architecture (click images to enlarge)
-  Hybrid Siamese Neural Network:

<td><img src="image/fig13.png" width="600"/></td>

## Training Workflow (click to enlarge)

<td><img src="image/fig16.png" width="600"/></td>

## 🔧 GPU and CUDA Environment
- NVIDIA RTX3060 (12 GB VRAM)
- CUDA 11.2
- cuDNN 8.1.1

## How to Use
Install dependencies:
```
pip install -r .\requirements.txt
```
Run training:
```
python .\main.py
```

## Reference
[1] D. Romano, C. Mennella, and M. Lapegna, ‘‘Adeep learning-based method
for efficient floating garbage debris recognition on high-performance edge
computing platform,’’ *Future Gener. Comput. Syst.*, vol. 174, p. 108000,
2026, doi: 10.1016/j.future.2025.108000.

[2] Sheng-Yan Dai, "NTUST-IB811 Wrist Vein Dataset", *IEEE Dataport*, November 13, 2025, doi:10.21227/w3ec-br30

[3] Ö. Toygar, F. O. Babalola and Y. Bitirim, ‘‘FYO: A Novel Multimodal Vein
Database With Palmar, Dorsal and Wrist Biometrics,’’ *IEEE Access*, vol.
8, pp. 82461-82470, 2020, doi: 10.1109/ACCESS.2020.2991475.

[4] R. Kabaciński and M. Kowalski, ‘‘Vein pattern database and benchmark
results,’’ *Electron. Lett.*, vol. 47, no. 20, pp. 1127–1128, Oct. 2011, doi:
10.1049/el.2011.1441.
