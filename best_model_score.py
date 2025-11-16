import pandas as pd
import numpy as np

def compute_multiobjective_scores(data):
    """
    data: list of tuples
          (Model, F1-Score, FPS, Watts, Joules)
    """
    df = pd.DataFrame(data, columns=["Model", "F1-Score", "FPS", "Watts", "Joules", "EER"])

    # Step 1: Z-score
    def zscore(col):
        col_mean = col.mean()
        col_std = col.std(ddof=0)
        print(f"{col} Z-score mean: {col_mean}, std: {col_std}")
        return (col - col.mean()) / col.std(ddof=0)

    Z = df[["F1-Score", "FPS", "Watts", "Joules", "EER"]].apply(zscore)

    # Step 2: Min-Max normalization to [0,1]
    def minmax_to_01(series):
        print(f"{series} Min-Max min: {series.min()}, max: {series.max()}")
        return (series - series.min()) / (series.max() - series.min())

    N = Z.apply(minmax_to_01)

    # Step 3: Invert Watts and Joules (since lower is better)
    N["Watts"] = 1 - N["Watts"]
    N["Joules"] = 1 - N["Joules"]
    N["EER"] = 1 - N["EER"]

    # Step 4: Weighted sum
    weights = {"F1-Score": 0.1, "FPS": 0.2, "Watts": 0.2, "Joules": 0.2, "EER": 0.3}
    df["F1_Score"] = N["F1-Score"]
    df["FPS_Score"] = N["FPS"]
    df["Watts_Score"] = N["Watts"]
    df["Joules_Score"] = N["Joules"]
    df["EER_Score"] = N["EER"]

    df["Total"] = (
        weights["F1-Score"] * df["F1_Score"] +
        weights["FPS"] * df["FPS_Score"] +
        weights["Watts"] * df["Watts_Score"] +
        weights["Joules"] * df["Joules_Score"] +
        weights["EER"] * df["EER_Score"]
    )

    # Step 5: Sort by Total score
    df_sorted = df[["Model", "F1_Score", "FPS_Score", "Watts_Score", "Joules_Score", "EER_Score", "Total"]].sort_values(by="Total", ascending=False).reset_index(drop=True)
    
    df_sorted.to_csv("multiobjective_scores_result.csv", index=False)
    
    return df_sorted

data = [
    ("Ours", 98.88, 50.4, 42.78, 0.00506, 1.09),
    ("ResNet18", 98.06, 45.72, 62.98, 0.01183, 1.93),
    ("ResNet34", 95.90, 38.3, 66.89, 0.019829, 3.56),
    ("ResNet50", 95.21, 38.71, 68.23, 0.021975, 4.55),
    ("MobileNetV1", 90.85, 28.68, 44.71, 0.005416, 9.60),
    ("MobileNetV2", 91.25, 39.7, 45.8, 0.008032, 8.80),
    ("MobileNetV3(Small)", 91.32, 38.72, 10.9, 0.002362, 8.59),
    ("EfficientNetB0", 86.04, 32.86, 47.81, 0.014097, 13.30),
    ("EfficientNetB1", 88.40, 18.54, 51.5, 0.021579, 11.65),
    ("VGG16", 98.61, 42.03, 93.14, 0.034442, 1.93),
    ("VGG19", 98.61, 45.75, 101.49, 0.043929, 1.79)
]

result = compute_multiobjective_scores(data)
print(result)
