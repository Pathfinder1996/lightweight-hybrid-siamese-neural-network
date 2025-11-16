import os
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input
from keras import backend as K

from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
)

from models import get_network
from my_metrics import EuclideanDistance

plt.rcParams["font.family"] = "Times New Roman"

# ---------------------- GPU NVML INIT ----------------------
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)

# ---------------------- CONFIG ----------------------
IDLE_POWER = 19.0  # W
WARMUP = 5
REPEAT = 21
TEST_IMG = 1000


def get_gpu_power():
    return nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0


def get_best_fold_index(result_txt_path):
    with open(result_txt_path, "r") as f:
        for line in f:
            if "Best fold index" in line:
                return int(line.strip().split(":")[1])
    return None


def build_and_load_model(model_name, input_shape, weight_path):
    inputA = Input(input_shape)
    inputB = Input(input_shape)
    feature_extractor = get_network(model_name, input_shape)
    featsA = feature_extractor(inputA)
    featsB = feature_extractor(inputB)
    distance = EuclideanDistance()([featsA, featsB])
    model = Model(inputs=[inputA, inputB], outputs=distance)
    model.load_weights(weight_path)
    return model


def wait_for_stable_idle_gpu(target_power=IDLE_POWER, required_stable_secs=5, timeout=120):
    print(f"\n⏳ Waiting for GPU to idle below {target_power}W for {required_stable_secs} seconds ...")
    stable_time = 0
    start_time = time.time()

    while True:
        current_power = get_gpu_power()
        if current_power <= target_power:
            stable_time += 1
        else:
            stable_time = 0  # reset if spike

        print(f"   → GPU Power: {current_power:.2f}W, Stable for {stable_time}s", end='\r')

        if stable_time >= required_stable_secs:
            print(f"\n✅ GPU stable at {current_power:.2f}W. Proceeding...")
            return

        if time.time() - start_time > timeout:
            print(f"\n⚠ Timeout reached. Proceeding anyway.")
            return

        time.sleep(1.0)


def measure_true_fps(model, input_shape, test_img=100):
    total_time = 0.0
    for _ in range(test_img):
        input1 = np.random.rand(1, *input_shape).astype(np.float32)
        input2 = np.random.rand(1, *input_shape).astype(np.float32)
        start = time.perf_counter()
        _ = model.predict([input1, input2], verbose=0)
        end = time.perf_counter()
        total_time += (end - start)
    return test_img * 2 / total_time


def measure_energy_and_trace(model, input_shape, repeat=REPEAT, warmup=WARMUP, cooldown_secs=5):
    total_energy, total_time = 0.0, 0.0
    power_trace = []

    for i in range(repeat):
        input1 = np.random.rand(TEST_IMG, *input_shape).astype(np.float32)
        input2 = np.random.rand(TEST_IMG, *input_shape).astype(np.float32)

        time.sleep(0.1)
        power_before = get_gpu_power()
        start = time.perf_counter()
        _ = model.predict([input1, input2], verbose=0)
        end = time.perf_counter()
        time.sleep(0.1)
        power_after = get_gpu_power()

        duration = end - start
        avg_power_measured = (power_before + power_after) / 2.0
        net_power = max(0.0, avg_power_measured - IDLE_POWER)
        energy = net_power * duration
        timestamp = (start + end) / 2.0
        power_trace.append((timestamp, avg_power_measured))

        if i >= warmup:
            total_energy += energy
            total_time += duration

    # cooldown recording
    cooldown_start = time.perf_counter()
    while time.perf_counter() - cooldown_start < cooldown_secs:
        timestamp = time.perf_counter()
        power = get_gpu_power()
        power_trace.append((timestamp, power))
        time.sleep(0.5)

    valid = repeat - warmup
    avg_time = total_time / valid
    avg_energy = total_energy / valid
    avg_power = avg_energy / avg_time if avg_time > 0 else 0
    batch_fps = TEST_IMG / avg_time if avg_time > 0 else 0
    joules_per_pair = avg_energy / TEST_IMG if TEST_IMG > 0 else 0
    joules_per_image = avg_energy / (2 * TEST_IMG) if TEST_IMG > 0 else 0

    return avg_energy, avg_time, avg_power, batch_fps, joules_per_pair, joules_per_image, power_trace


def main():
    input_shape = (64, 64, 1)
    
    network_names = ["Ours", 
                    "ResNet18",
                    "ResNet34",
                    "ResNet50", 
                    "MobileNetV1", 
                    "MobileNetV2", 
                    "MobileNetV3_Small", 
                    "EfficientNetB0",
                    "EfficientNetB1",
                    "VGG16",
                    "VGG19"
                    ] 

    output_csv = "energy_benchmark_results.csv"
    traces = {}
    with open(output_csv, "w") as f:
        f.write("Model,AvgEnergy(J),AvgTime(s),AvgPower(W),BatchFPS,Joules_per_Pair,Joules_per_Image,TrueFPS(perImage)\n")

        for name in network_names:
            try:
                print(f"\n\U0001F680 Testing model: {name}")
                result_txt = os.path.join("results", name, f"{name}_k_fold_results.txt")
                best_fold = get_best_fold_index(result_txt)
                if best_fold is None:
                    raise ValueError("No fold index found")

                weight_path = os.path.join("results", name, f"{name}_model_fold_{best_fold}.keras")
                if not os.path.exists(weight_path):
                    raise FileNotFoundError(weight_path)

                model = build_and_load_model(name, input_shape, weight_path)
                _ = model.predict([np.random.rand(1, *input_shape), np.random.rand(1, *input_shape)], verbose=0)

                wait_for_stable_idle_gpu()
                time.sleep(5)

                e, t, p, batch_fps, jpp, jpi, trace = measure_energy_and_trace(model, input_shape)
                true_fps = measure_true_fps(model, input_shape)

                traces[name] = trace
                f.write(f"{name},{e:.6f},{t:.6f},{p:.2f},{batch_fps:.2f},{jpp:.6f},{jpi:.6f},{true_fps:.2f}\n")

                K.clear_session()
                gc.collect()

            except Exception as e:
                print(f"❌ {name} failed: {e}")
                f.write(f"{name},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR\n")

    # Plot power trace
    plt.figure(figsize=(14, 6))
    for model_name, trace in traces.items():
        if not trace:
            continue
        base_time = trace[0][0]
        times = [t - base_time for t, _ in trace]
        powers = [p for _, p in trace]

        if model_name == "Ours":
            plt.plot(times, powers, label=model_name, color="black", linewidth=3)
        else:
            plt.plot(times, powers, label=model_name)

    plt.axhline(IDLE_POWER, color='red', linestyle='--', label='Idle Baseline')
    plt.title("GPU Power Trace During Inference for Each Model")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Power (Watts)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("power_trace_comparison.svg", format="svg")
    plt.close()

    # Plot Joules per Image
    try:
        df = pd.read_csv(output_csv)
        df = df[df["Joules_per_Image"] != "ERROR"]
        df["Joules_per_Image"] = pd.to_numeric(df["Joules_per_Image"], errors="coerce")
        df = df.dropna()
        df = df.sort_values("Joules_per_Image")

        plt.figure(figsize=(14, 6))
        bars = plt.bar(df["Model"], df["Joules_per_Image"], color="skyblue", edgecolor="black")
        plt.title("Average Energy Consumption per Inference (Joules/Image)")
        plt.ylabel("Energy (Joules)")
        plt.xticks(rotation=45)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.3f}", va='bottom', ha='center')
        plt.tight_layout()
        plt.savefig("joules_per_image_barplot.svg", format="svg")
        plt.close()
    except Exception as e:
        print(f"⚠ Failed to plot bar chart: {e}")

    nvmlShutdown()
    print("\n✅ Benchmarking complete. Results saved.")


if __name__ == "__main__":
    main()
