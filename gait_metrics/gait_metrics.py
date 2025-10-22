# Modified gait_metrics.py with chunked processing by day

# !/usr/bin/env python3

"""
Gait Metrics Computation Tool

This script processes multiple accelerometer data files to compute gait metrics
such as walking time, step count, gait speed, cadence, stride length, and
regularity using pre-trained models, loaded once for efficiency from .zip files
on GitHub Releases.
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.amp import autocast
from scipy.stats import kurtosis, skew

import utils
from model_utils import setup_model

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000000
mpl.use('TkAgg')
import matplotlib.pyplot as plt


# Constants
N_BINS = 10

GLOBAL_RANGES = {
    "bout_duration_all_values": {"min": 10, "max": 3200},
    "gait_speed_all_values": {"min": 0, "max": 1.8},
    "bout_gait_speed_all_values": {"min": 0, "max": 1.8},
    "cadence_all_values": {"min": 40, "max": 160},
    "bout_cadence_all_values": {"min": 40, "max": 160},
    "gait_length_all_values": {"min": 0, "max": 2},
    "bout_gait_length_all_values": {"min": 0, "max": 2},
    "gait_length_indirect_all_values": {"min": 0, "max": 2},
    "bout_gait_length_indirect_all_values": {"min": 0, "max": 2},
    "regularity_eldernet_all_values": {"min": 0, "max": 1},
    "bout_regularity_eldernet_all_values": {"min": 0, "max": 1},
    "regularity_sp_all_values": {"min": 0, "max": 1},
    "bout_regularity_sp_all_values": {"min": 0, "max": 1},
}

MODEL_URLS = {
    "gait_detection_model.pt": (
        "https://github.com/yonbrand/gait-metrics/releases/download/v0.1.0/gait_detection_model.pt"
    ),
    "step_count_model.pt": (
        "https://github.com/yonbrand/gait-metrics/releases/download/v0.1.0/step_count_model.pt"
    ),
    "gait_speed_model.pt": (
        "https://github.com/yonbrand/gait-metrics/releases/download/v0.1.0/gait_speed_model.pt"
    ),
    "cadence_model.pt": (
        "https://github.com/yonbrand/gait-metrics/releases/download/v0.1.0/cadence_model.pt"
    ),
    "gait_length_model.pt": (
        "https://github.com/yonbrand/gait-metrics/releases/download/v0.1.0/gait_length_model.pt"
    ),
    "regularity_model.pt": (
        "https://github.com/yonbrand/gait-metrics/releases/download/v0.1.0/regularity_model.pt"
    ),
}

# Set device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model cache, initialized to None
_models_cache = {
    "gait_detection": None,
    "step_count": None,
    "gait_speed": None,
    "cadence": None,
    "stride_length": None,
    "regularity": None,
}


def get_model(model_name: str) -> torch.nn.Module:
    """
    Lazily loads and returns a model from the cache.
    If not in the cache, it loads the model (downloading from URL if needed)
    and stores it in the cache before returning.
    """

    # 1. Check if model is already loaded
    if _models_cache[model_name] is not None:
        return _models_cache[model_name]

    # 2. If not loaded, load it now
    print(f"Loading {model_name} model...")
    model = None

    if model_name == "gait_detection":
        model = setup_model(
            net="ElderNet",
            output_size=2,
            is_classification=True,
            model_url=MODEL_URLS["gait_detection_model.pt"],
            device=device,
        )

    elif model_name == "step_count":
        model = setup_model(
            net="ElderNet",
            output_size=1,
            is_regression=True,
            num_layers_regressor=1,
            max_mu=25.0,
            batch_norm=True,
            model_url=MODEL_URLS["step_count_model.pt"],
            device=device,
        )

    elif model_name == "gait_speed":
        model = setup_model(
            net="ElderNet",
            output_size=1,
            is_regression=True,
            num_layers_regressor=0,
            max_mu=2.0,
            model_url=MODEL_URLS["gait_speed_model.pt"],
            device=device,
        )

    elif model_name == "cadence":
        model = setup_model(
            net="ElderNet",
            output_size=1,
            is_regression=True,
            num_layers_regressor=1,
            max_mu=160.0,
            batch_norm=True,
            model_url=MODEL_URLS["cadence_model.pt"],
            device=device,
        )

    elif model_name == "stride_length":
        model = setup_model(
            net="ElderNet",
            output_size=1,
            is_regression=True,
            num_layers_regressor=1,
            max_mu=2.0,
            batch_norm=True,
            model_url=MODEL_URLS["gait_length_model.pt"],
            device=device,
        )

    elif model_name == "regularity":
        model = setup_model(
            net="ElderNet",
            output_size=1,
            is_regression=True,
            num_layers_regressor=1,
            max_mu=1.0,
            model_url=MODEL_URLS["regularity_model.pt"],
            device=device,
        )

    else:
        raise ValueError(f"Unknown model name requested: {model_name}")

    # 3. Set to evaluation mode and store in cache
    if model is not None:
        model.eval()
        _models_cache[model_name] = model

    return model


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def histogram_features(column_data, global_ranges, bins: int = 5, feature_name: str = None) -> dict:
    """
    Computes histogram-based features from all_values columns, with normalization
    and additional stats.

    Args:
        column_data: Input data array.
        global_ranges (dict): Dictionary of min/max ranges for normalization.
        bins (int, optional): Number of histogram bins. Defaults to 5.
        feature_name (str, optional): Name of the feature for range lookup.

    Returns:
        dict: Dictionary containing histogram probabilities.
    """
    if len(column_data) == 0:
        return {"prob": [np.nan] * bins}

    if feature_name in global_ranges:
        min_val, max_val = global_ranges[feature_name]["min"], global_ranges[feature_name]["max"]
    else:
        min_val, max_val = np.min(column_data), np.max(column_data)

    hist, _ = np.histogram(column_data, bins=bins, range=(min_val, max_val))
    total = hist.sum()
    prob = (hist / total).tolist() if total > 0 else [0.0] * bins
    return {"freqs": hist.tolist(), "prob": prob}


def process_signal_regularity(walking_batch, fs: float) -> np.ndarray:
    """
    Process walking batch to compute regularity scores.

    Args:
        walking_batch: Array of walking segments.
        fs (float): Sampling frequency.

    Returns:
        np.ndarray: Regularity scores reshaped.
    """
    regularity_scores = np.array(
        [utils.calc_regularity(np.linalg.norm(segment, axis=1), fs) for segment in walking_batch]
    )
    return regularity_scores.reshape(-1, 1)


def process_batch(batch, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """
    Process a batch of data through a model with mixed precision.

    Args:
        batch: Input data batch (numpy array or tensor-like).
        model (torch.nn.Module): PyTorch model instance.
        device (torch.device): Torch device (CPU or CUDA).

    Returns:
        np.ndarray: Model predictions.
    """

    # Convert batch to tensor and move to device
    X = torch.as_tensor(batch, dtype=torch.float32).to(device, non_blocking=True)

    # Transpose if needed (check can be optimized)
    if X.shape[1] != 3:
        X = X.transpose(1, 2).contiguous()  # Ensure contiguous memory for efficiency

    # Use autocast for mixed precision inference
    with torch.no_grad(), autocast(device_type=device.type):
        predictions = model(X)
        if predictions.shape[1] == 2:
            predictions = torch.argmax(predictions, dim=1)

    # Move to CPU and convert to numpy efficiently
    return predictions.cpu().to(torch.float32).numpy()


def calculate_statistics(
        result: dict, window_sec: int, window_len: int, analyze_bouts: bool
) -> dict:
    """
    Calculate statistical metrics from gait data.

    Args:
        result (dict): Dictionary containing prediction results.
        window_sec (int): Window size in seconds.
        window_len (int): Window length in samples.
        analyze_bouts (bool): Whether to analyze bout-specific stats.

    Returns:
        dict: Statistical metrics for walking and bouts.
    """

    windows_per_day = int(24 * 60 * 60 / window_sec)
    days = len(np.unique(result["window_days"]))

    if len(result["pred_walk"]) % windows_per_day == 0:
        reshaped = result["pred_walk"].reshape(-1, windows_per_day)
        daily_walking_amounts = np.sum(reshaped, axis=1) * window_sec / 60
    else:
        indices = np.arange(0, len(result["pred_walk"]), windows_per_day)
        daily_walking_amounts = np.array(
            [np.sum(result["pred_walk"][i: i + windows_per_day]) for i in indices]
        ) * window_sec / 60

    window_days = np.array(result["window_days"], dtype=np.int64).flatten()
    pred_window_steps = np.array(result["pred_window_steps"], dtype=np.float64).flatten()

    if len(window_days) > 0 and len(window_days) == len(pred_window_steps):
        if np.all(window_days >= 0):
            daily_step_count = np.bincount(
                window_days, weights=pred_window_steps, minlength=days
            )
        else:
            print("Warning: Negative indices in window_days, setting daily_step_count to zeros")
            daily_step_count = np.zeros(days, dtype=np.float64)
    else:
        print(
            "Warning: Empty or mismatched window_days/pred_window_steps, "
            "setting daily_step_count to zeros"
        )
        daily_step_count = np.zeros(days, dtype=np.float64)

    def calc_stats(data, prefix: str = "", compute_advanced: bool = False, n_bins: int = 5) -> dict:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data = data.flatten()
        if len(data) == 0:
            stats_dict = {
                f"{prefix}{stat}": np.nan
                for stat in [
                    "median",
                    "mean",
                    "std",
                    "p5",
                    "p10",
                    "p90",
                    "p95",
                    "kurtosis",
                    "skewness",
                    "range",
                ]
            }
            if compute_advanced:
                stats_dict.update({f"{prefix}prob_bin{i}": np.nan for i in range(n_bins)})
            return stats_dict

        stats_dict = {
            f"{prefix}median": np.median(data),
            f"{prefix}mean": np.mean(data),
            f"{prefix}std": np.std(data),
            f"{prefix}p10": np.percentile(data, 10),
            f"{prefix}p25": np.percentile(data, 25),
            f"{prefix}p75": np.percentile(data, 75),
            f"{prefix}p90": np.percentile(data, 90),
            f"{prefix}kurtosis": kurtosis(data),
            f"{prefix}skewness": skew(data),
            f"{prefix}range": np.max(data) - np.min(data),
        }

        if compute_advanced:
            hist_features = histogram_features(
                data, GLOBAL_RANGES, bins=n_bins, feature_name=prefix + "all_values"
            )

            hist_freqs = hist_features["freqs"]
            hist_probs = hist_features["prob"]
            for i, (freq, prob) in enumerate(zip(hist_freqs, hist_probs)):
                stats_dict[f"{prefix}freq_bin{i}"] = freq
                stats_dict[f"{prefix}prob_bin{i}"] = prob

        return stats_dict

    walking_time = calc_stats(daily_walking_amounts, prefix="walking_time_")
    step_count = calc_stats(daily_step_count, prefix="step_count_")

    all_values = {
        "gait_speed_all_values": result["pred_speed"],
        "cadence_all_values": result["pred_cadence"],
        "gait_length_all_values": result["pred_gait_length"],
        "gait_length_indirect_all_values": result["pred_gait_length_indirect"],
        "regularity_eldernet_all_values": result["pred_regularity_eldernet"],
        "regularity_sp_all_values": result["pred_regularity_sp"],
    }

    all_values_stats = {}
    for name, data in all_values.items():
        all_values_stats.update(
            calc_stats(data, prefix=name[:-10], compute_advanced=True, n_bins=N_BINS)
        )

    bout_values = {}
    if analyze_bouts:
        bout_days = np.array(result["bout_days"], dtype=np.int64).flatten()
        bout_durations = np.array(result["bouts_durations"], dtype=np.float64).flatten()

        # Compute per-bout medians for each metric
        bouts_id = np.array(result["bouts_id"])
        unique_bouts = np.unique(bouts_id)
        bout_gait_speed = []
        bout_cadence = []
        bout_gait_length = []
        bout_gait_length_indirect = []
        bout_regularity_eldernet = []
        bout_regularity_sp = []

        for bout in unique_bouts:
            mask = bouts_id == bout
            if np.any(mask):
                bout_gait_speed.append(np.median(result["pred_speed"][mask]))
                bout_cadence.append(np.median(result["pred_cadence"][mask]))
                bout_gait_length.append(np.median(result["pred_gait_length"][mask]))
                bout_gait_length_indirect.append(np.median(result["pred_gait_length_indirect"][mask]))
                bout_regularity_eldernet.append(np.median(result["pred_regularity_eldernet"][mask]))
                bout_regularity_sp.append(np.median(result["pred_regularity_sp"][mask]))

        bout_values = {
            "bout_duration_all_values": bout_durations,
            "bout_gait_speed_all_values": np.array(bout_gait_speed),
            "bout_cadence_all_values": np.array(bout_cadence),
            "bout_gait_length_all_values": np.array(bout_gait_length),
            "bout_gait_length_indirect_all_values": np.array(bout_gait_length_indirect),
            "bout_regularity_eldernet_all_values": np.array(bout_regularity_eldernet),
            "bout_regularity_sp_all_values": np.array(bout_regularity_sp),
        }

        bout_values_stats = {}
        for name, data in bout_values.items():
            bout_values_stats.update(
                calc_stats(data, prefix=name[:-10], compute_advanced=True, n_bins=N_BINS)
            )

    else:
        bout_values_stats = {}

    stats = {
        "subject_id": result["subject_id"],
        "wear_days": result["wear_days"],
        **walking_time,
        **step_count,
        **all_values_stats,
        **bout_values_stats
    }

    return stats


def set_model_to_eval(*models):
    """
    Set multiple models to evaluation mode.

    Args:
        *models: Variable number of model instances.
    """
    for model in models:
        model.eval()


def main():
    # Hardcoded arguments for debugging in PyCharm
    file_path = r'N:\Gait-Neurodynamics by Names\Yonatan\gait-metrics\gait_metrics\rush-sample.cwa.gz'  # Replace with your file path
    output_dir = None  # Or set to a directory like "output"
    txyz = "time,x,y,z"
    exclude_first_last = 'None'
    verbose = True
    analyze_bouts = True

    # Wrap in a class to mimic args
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    args = Args(
        file_path=file_path,
        output_dir=output_dir,
        txyz=txyz,
        exclude_first_last=exclude_first_last,
        exclude_wear_below=None,
        verbose=verbose,
        analyze_bouts=analyze_bouts
    )

    # Process each file
    file_path = args.file_path
    basename = utils.resolve_path(file_path)[1]
    info = {"GaitArgs": vars(args)}

    # Data processing
    try:
        data, info_read = utils.read(
            file_path,
            usecols=args.txyz,
            start_time=None,
            end_time=None,
            sample_rate=None,
            resample_hz=30,
            verbose=args.verbose,
        )
        info.update(info_read)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


    if args.exclude_first_last is not None:
        data = utils.drop_first_last_days(data, args.exclude_first_last)

    if args.exclude_wear_below is not None:
        data = utils.flag_wear_below_days(data, args.exclude_wear_below)

    info.update(utils.calculate_wear_stats(data))

    fs = info["ResampleRate"]
    window_sec = 10
    window_len = int(window_sec * fs)
    window_step_len = window_len

    processed_acc = data[["x", "y", "z"]].to_numpy()
    samples_per_day_resampled = int(24 * 60 * 60 * fs)
    num_days = len(processed_acc) // samples_per_day_resampled + (
        1 if len(processed_acc) % samples_per_day_resampled > 0 else 0)

    batch_size = 1024

    # Load the gait detection model
    gait_detection_model = get_model("gait_detection")

    # Initialize accumulators
    pred_walk_all = []
    pred_steps_all = []
    pred_speed_all = []
    pred_cadence_all = []
    pred_gait_length_all = []
    pred_gait_length_indirect_all = []
    pred_regularity_eldernet_all = []
    pred_regularity_sp_all = []
    window_days_all = []
    bouts_durations_all = []
    bout_days_all = []
    walking_bouts_id_all = []
    current_bout_id = 1

    for day in range(num_days):
        start_idx = day * samples_per_day_resampled
        end_idx = min((day + 1) * samples_per_day_resampled, len(processed_acc))
        acc_day = processed_acc[start_idx:end_idx]
        day_length = end_idx - start_idx
        if day_length < window_len:
            continue  # Skip days with insufficient data

        # Create non-overlapping windows for the day
        acc_win_day = np.array(
            [acc_day[j: j + window_len] for j in range(0, day_length - window_len + 1, window_step_len)]
        )

        # Process gait detection in batches
        pred_walk_day = []
        for i in range(0, len(acc_win_day), batch_size):
            batch = acc_win_day[i: i + batch_size]
            with torch.inference_mode():
                batch_pred_walk = process_batch(batch, gait_detection_model, device)
            pred_walk_day.extend(batch_pred_walk)
        pred_walk_day = np.array(pred_walk_day)

        # Bout detection
        diff_preds = np.diff(pred_walk_day, prepend=0, append=0)
        where_bouts_start = np.where(diff_preds == 1)[0]
        where_bouts_end = np.where(diff_preds == -1)[0]
        bouts_durations_day = (where_bouts_end - where_bouts_start) * window_sec

        bouts_id_windows = np.zeros_like(pred_walk_day)
        num_bouts_day = len(where_bouts_start)
        if args.analyze_bouts and num_bouts_day > 0:
            for local_bout_idx, (start, end) in enumerate(zip(where_bouts_start, where_bouts_end)):
                global_bout_id = current_bout_id + local_bout_idx
                bouts_id_windows[start:end] = global_bout_id
            current_bout_id += num_bouts_day
            bouts_durations_all.extend(bouts_durations_day)
            bout_days_all.extend([day] * num_bouts_day)

        # Walking segments
        walk_mask = pred_walk_day == 1
        walking_window_indices = np.where(walk_mask)[0]
        walking_batch = acc_win_day[walk_mask]

        if args.analyze_bouts:
            walking_bouts_id_day = bouts_id_windows[walk_mask]
            walking_bouts_id_all.extend(walking_bouts_id_day)

        if walking_batch.size > 0:
            # Lazily load the other models only if walking is detected in any day.
            step_count_model = get_model("step_count")
            gait_speed_model = get_model("gait_speed")
            cadence_model = get_model("cadence")
            stride_length_model = get_model("stride_length")
            regularity_model = get_model("regularity")

            pred_steps_day = []
            pred_speed_day = []
            pred_cadence_day = []
            pred_gait_length_day = []
            pred_regularity_eldernet_day = []

            for i in range(0, len(walking_batch), batch_size):
                batch = walking_batch[i: i + batch_size]
                with torch.inference_mode():
                    batch_pred_steps = process_batch(batch, step_count_model, device)
                    batch_pred_speed = process_batch(batch, gait_speed_model, device)
                    batch_pred_cadence = process_batch(batch, cadence_model, device)
                    batch_pred_gait_length = process_batch(batch, stride_length_model, device)
                    batch_pred_regularity_eldernet = process_batch(batch, regularity_model, device)
                pred_steps_day.extend(batch_pred_steps)
                pred_speed_day.extend(batch_pred_speed)
                pred_cadence_day.extend(batch_pred_cadence)
                pred_gait_length_day.extend(batch_pred_gait_length)
                pred_regularity_eldernet_day.extend(batch_pred_regularity_eldernet)

            pred_steps_day = np.array(np.round(pred_steps_day))
            pred_speed_day = np.array(pred_speed_day)
            pred_cadence_day = np.array(pred_cadence_day)
            pred_gait_length_day = np.array(pred_gait_length_day)
            pred_gait_length_indirect_day = np.array(120 * pred_speed_day / pred_cadence_day) if len(
                pred_cadence_day) > 0 else np.array([])
            pred_regularity_eldernet_day = np.array(pred_regularity_eldernet_day)
            pred_regularity_sp_day = process_signal_regularity(walking_batch, fs)

            pred_walk_all.extend(pred_walk_day)
            pred_steps_all.extend(pred_steps_day)
            pred_speed_all.extend(pred_speed_day)
            pred_cadence_all.extend(pred_cadence_day)
            pred_gait_length_all.extend(pred_gait_length_day)
            pred_gait_length_indirect_all.extend(pred_gait_length_indirect_day)
            pred_regularity_eldernet_all.extend(pred_regularity_eldernet_day)
            pred_regularity_sp_all.extend(pred_regularity_sp_day)

        else:
            pred_walk_all = np.array([])
            pred_steps_all = np.array([])
            pred_speed_all = np.array([])
            pred_cadence_all = np.array([])
            pred_gait_length_all = np.array([])
            pred_gait_length_indirect_all = np.array([])
            pred_regularity_eldernet_all = np.array([])
            pred_regularity_sp_all = np.array([])

        window_days_day = np.full(len(walking_window_indices), day, dtype=np.int64)
        window_days_all.extend(window_days_day)

        # Clean up day-specific memory
        del acc_win_day, walking_batch

    # Convert accumulators to arrays
    pred_walk = np.array(pred_walk_all)
    pred_steps = np.array(pred_steps_all)
    pred_speed = np.array(pred_speed_all)
    pred_cadence = np.array(pred_cadence_all)
    pred_gait_length = np.array(pred_gait_length_all)
    pred_gait_length_indirect = np.array(pred_gait_length_indirect_all)
    pred_regularity_eldernet = np.array(pred_regularity_eldernet_all)
    pred_regularity_sp = np.array(pred_regularity_sp_all)
    window_days = np.array(window_days_all)
    bouts_durations = np.array(bouts_durations_all)
    bout_days = np.array(bout_days_all)
    walking_bouts_id = np.array(walking_bouts_id_all)

    result = {
        "subject_id": basename,
        "wear_days": num_days,
        "pred_window_steps": pred_steps,
        "window_days": window_days,
        "pred_walk": pred_walk,
        "pred_speed": pred_speed,
        "pred_cadence": pred_cadence,
        "pred_gait_length": pred_gait_length,
        "pred_gait_length_indirect": pred_gait_length_indirect,
        "pred_regularity_eldernet": pred_regularity_eldernet,
        "pred_regularity_sp": pred_regularity_sp,
        "bouts_id": walking_bouts_id,
        "bouts_durations": bouts_durations,
        "bout_days": bout_days,
    }

    stats = calculate_statistics(result, window_sec, window_len, args.analyze_bouts)

    output_path = os.path.join(args.output_dir, f"{basename}.json") if args.output_dir else f"{basename}.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4, cls=NumpyEncoder)


    return None  # No return value needed for CLI

if __name__ == "__main__":
    main()
