## 📝 輕量化混合孿生神經網路
此專題為我的碩士論文中靜脈辨識系統的靜脈特徵匹配程式碼與實驗數據。

使用 Python 在個人電腦上完成模型訓練、驗證與測試，並將建立的模型轉為 TFLite 格式到資源受限之嵌入式裝置上運行。

模型架構、損失函數選用、超參數設置與訓練流程皆在我的論文第 50–57 頁。[請點此到我的論文連結並到電子全文下載論文](https://etheses.lib.ntust.edu.tw/thesis/detail/2b733280676d7c87e0445313c40a9b74/?seq=2#)

### 📁 壓縮檔內容
- `blocks.py` - 定義 models.py(模型主架構)中所用到的一些 blocks(像是卷積、深度分離卷積、殘差塊、反向殘差塊等等...)。
- `data_loader.py` - 訓練資料載入與訓練標籤製作。
- `labels_vis.py` - 將訓練標籤製作視覺化。
- `main.py` - 主程式(建模)。
- `models.py` - 各模型主架構(包含本論文(Ours)、ResNet、MobileNet 自定義的一些測試架構等等...)。
- `my_metrics.py` -  參考 stack overflow 或其他教學網站自定義的模型評估指標(像是對比損失函數、歐幾里得距離等等...)。
- `plot_all_model_db.py` - 自定義函式部分，將實驗數據資料視覺化。將所有模型的訓練結果(DET 與 ROC Curve)繪製在同一張圖上。
- `plot_ours_all_db.py` - 自定義函式部分，將實驗數據資料視覺化。將 Ours 自建的模型在三個資料庫上每一折交叉驗證的訓練結果繪製在同一張圖上。
- `plot_utils.py` - 自定義函式部分，評估指標的計算與圖表繪製(包含混淆矩陣、DET曲線上的EER等等...)。

## 📁 資料庫
用來訓練模型的三個手腕靜脈資料庫分別為 NTUST-IB811、FYO 與 PUT
- NTUST-IB811: 本論文所收集之手腕靜脈影像。
- FYO: 經申請後下載。 [點此連結到資料庫申請網址或聯繫期刊作者](https://fyo.emu.edu.tr/en/download)
- PUT: 經申請後下載。 [點此連結到資料庫申請網址或聯繫期刊作者](https://digital-library.theiet.org/doi/abs/10.1049/el.2011.1441)

## 📦 模型架構
- 主架構:

![main](image/1.svg)
  
- 特徵提取子網路(Subnetwork):

![subnet](image/2.svg)

## 📊 實驗結果


## 🚀 如何使用
請輸入以下指令建置 Python3.9.2 環境用到的函式庫及其版本:
```
pip install -r .\requirements.txt
```
輸入以下指令執行程式進行模型訓練:
```
python .\main.py
```
