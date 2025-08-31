#!/usr/bin/env python3

import argparse
from importlib_resources import files
import numpy as np
import torch
from scipy.stats import kurtosis, skew
import json  # Added for JSON output

from . import utils
from .model_utils import setup_model

N_BINS = 10

GLOBAL_RANGES = {'bout_duration_all_values': {'min': 10, 'max': 3200}, 'gait_speed_all_values': {'min': 0, 'max': 1.8},
                 'bout_gait_speed_all_values': {'min': 0, 'max': 1.8}, 'cadence_all_values': {'min': 40, 'max': 160},
                 'bout_cadence_all_values': {'min': 40, 'max': 160}, 'gait_length_all_values': {'min': 0, 'max': 2},
                 'bout_gait_length_all_values': {'min': 0, 'max': 2},
                 'gait_length_indirect_all_values': {'min': 0, 'max': 2},
                 'bout_gait_length_indirect_all_values': {'min': 0, 'max': 2},
                 'regularity_eldernet_all_values': {'min': 0, 'max': 1},
                 'bout_regularity_eldernet_all_values': {'min': 0, 'max': 1},
                 'regularity_sp_all_values': {'min': 0, 'max': 1},
                 'bout_regularity_sp_all_values': {'min': 0, 'max': 1}}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def histogram_features(column_data, global_ranges, bins=5, feature_name=None):
    """Computes histogram-based features from all_values columns, with normalization and additional stats."""
    if len(column_data) == 0:
        return {'prob': [np.nan] * bins}

    if feature_name in global_ranges:
        min_val, max_val = global_ranges[feature_name]["min"], global_ranges[feature_name]["max"]
    else:
        min_val, max_val = np.min(column_data), np.max(column_data)

    hist, bin_edges = np.histogram(column_data, bins=bins, range=(min_val, max_val))
    total = hist.sum()
    prob = (hist / total).tolist() if total > 0 else [0.0] * bins
    return {'prob': prob}


def process_signal_regularity(walking_batch, fs):
    regularity_scores = np.array(
        [utils.calc_regularity(np.linalg.norm(segment, axis=1), fs) for segment in walking_batch])
    return regularity_scores.reshape(-1, 1)


def process_batch(batch, model, device):
    X = torch.tensor(batch, dtype=torch.float32).to(device)
    if X.shape[1] != 3:
        X = X.transpose(1, 2)
    with torch.no_grad():
        predictions = model(X)
        if predictions.shape[1] == 2:
            predictions = torch.argmax(predictions, dim=1)
    return predictions.cpu().numpy()


def calc_second_prediction(pred_walk, window_sec, second_predictions):
    for idx, window_prediction in enumerate(pred_walk):
        center_second = idx + window_sec // 2
        if center_second < len(second_predictions):
            second_predictions[center_second] = window_prediction
    return second_predictions


def detect_bouts(pred_walk):
    diff_preds = np.diff(pred_walk, prepend=0, append=0)
    where_bouts_start = np.where(diff_preds == 1)[0]
    where_bouts_end = np.where(diff_preds == -1)[0]
    bouts_id = np.zeros_like(pred_walk)
    for bout_idx, (start, end) in enumerate(zip(where_bouts_start, where_bouts_end), 1):
        bouts_id[start:end] = bout_idx
    return bouts_id


def calculate_statistics(result, window_sec, window_len, analyze_bouts):
    seconds_per_sample = window_sec / window_len
    samples_per_day = int(24 * 60 * 60 / seconds_per_sample)
    days = len(np.unique(result['window_days']))

    if len(result['pred_walk']) % samples_per_day == 0:
        reshaped = result['pred_walk'].reshape(-1, samples_per_day)
        daily_walking_amounts = np.sum(reshaped, axis=1) * seconds_per_sample / 60
    else:
        indices = np.arange(0, len(result['pred_walk']), samples_per_day)
        daily_walking_amounts = np.array(
            [np.sum(result['pred_walk'][i:i + samples_per_day]) for i in indices]) * seconds_per_sample / 60

    window_days = np.array(result['window_days'], dtype=np.int64).flatten()
    pred_window_steps = np.array(result['pred_window_steps'], dtype=np.float64).flatten()

    if len(window_days) > 0 and len(window_days) == len(pred_window_steps):
        if np.all(window_days >= 0):
            daily_step_count = np.bincount(
                window_days,
                weights=pred_window_steps,
                minlength=days
            )
        else:
            print("Warning: Negative indices in window_days, setting daily_step_count to zeros")
            daily_step_count = np.zeros(days, dtype=np.float64)
    else:
        print("Warning: Empty or mismatched window_days/pred_window_steps, setting daily_step_count to zeros")
        daily_step_count = np.zeros(days, dtype=np.float64)

    def calc_stats(data, prefix='', compute_advanced=False, n_bins=5):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data = data.flatten()
        if len(data) == 0:
            stats_dict = {f'{prefix}{stat}': np.nan for stat in
                          ['median', 'mean', 'std', 'p5', 'p10', 'p90', 'p95',
                           'kurtosis', 'skewness', 'range']}
            if compute_advanced:
                stats_dict.update({f'{prefix}prob_bin{i}': np.nan for i in range(n_bins)})
            return stats_dict

        stats_dict = {
            f'{prefix}median': np.median(data),
            f'{prefix}mean': np.mean(data),
            f'{prefix}std': np.std(data),
            f'{prefix}p5': np.percentile(data, 5),
            f'{prefix}p10': np.percentile(data, 10),
            f'{prefix}p90': np.percentile(data, 90),
            f'{prefix}p95': np.percentile(data, 95),
            f'{prefix}kurtosis': kurtosis(data),
            f'{prefix}skewness': skew(data),
            f'{prefix}range': np.max(data) - np.min(data)
        }

        if compute_advanced:
            hist_features = histogram_features(data, GLOBAL_RANGES, bins=n_bins, feature_name=prefix + 'all_values')
            stats_dict.update({f'{prefix}prob_bin{i}': prob for i, prob in enumerate(hist_features['prob'])})

        return stats_dict

    walking_time = calc_stats(daily_walking_amounts, prefix='walking_time_')
    step_count = calc_stats(daily_step_count, prefix='step_count_')

    all_values = {
        'gait_speed_all_values': result['pred_speed'],
        'cadence_all_values': result['pred_cadence'],
        'gait_length_all_values': result['pred_gait_length'],
        'gait_length_indirect_all_values': result['pred_gait_length_indirect'],
        'regularity_eldernet_all_values': result['pred_regularity_eldernet'],
        'regularity_sp_all_values': result['pred_regularity_sp']
    }

    all_values_stats = {}
    for name, data in all_values.items():
        all_values_stats.update(calc_stats(data, prefix=name[:-10] + '_', compute_advanced=True, n_bins=N_BINS))

    bout_values = {}
    if analyze_bouts:
        bout_days = np.array(result['bout_days'], dtype=np.int64).flatten()
        bout_durations = np.array(result['bouts_durations'], dtype=np.float64).flatten()
        if len(bout_days) > 0 and len(bout_days) == len(bout_durations):
            if np.all(bout_days >= 0):
                daily_bout_durations = np.bincount(
                    bout_days,
                    weights=bout_durations,
                    minlength=days
                )
            else:
                print("Warning: Negative indices in bout_days, setting daily_bout_durations to zeros")
                daily_bout_durations = np.zeros(days, dtype=np.float64)
        else:
            print("Warning: Empty or mismatched bout_days/bout_durations, setting daily_bout_durations to zeros")
            daily_bout_durations = np.zeros(days, dtype=np.float64)

        bout_values = {
            'bout_duration_all_values': bout_durations,
            'bout_gait_speed_all_values': result['pred_speed'],
            'bout_cadence_all_values': result['pred_cadence'],
            'bout_gait_length_all_values': result['pred_gait_length'],
            'bout_gait_length_indirect_all_values': result['pred_gait_length_indirect'],
            'bout_regularity_eldernet_all_values': result['pred_regularity_eldernet'],
            'bout_regularity_sp_all_values': result['pred_regularity_sp']
        }

        bout_values_stats = {}
        for name, data in bout_values.items():
            bout_values_stats.update(calc_stats(data, prefix=name[:-10] + '_', compute_advanced=True, n_bins=N_BINS))

        bout_durations_stats = calc_stats(daily_bout_durations, prefix='bout_duration_')
    else:
        bout_values_stats = {}
        bout_durations_stats = {}

    stats = {
        'subject_id': result['subject_id'],
        'wear_days': result['wear_days'],
        **walking_time,
        **step_count,
        **all_values_stats,
        **bout_durations_stats,
        **bout_values_stats
    }

    return stats

def set_model_to_eval(*models):
    for model in models:
        model.eval()

def main():
    parser = argparse.ArgumentParser(description='Gait Metrics Computation Tool')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the accelerometer file')
    parser.add_argument('--txyz', type=str, default='time,x,y,z', help='Column names for time and acceleration axes')
    parser.add_argument('--start', type=str, default=None, help='Start time for processing (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end', type=str, default=None, help='End time for processing (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--sample_rate', type=int, default=None, help='Sample rate of the accelerometer data')
    parser.add_argument('--exclude_first_last', type=int, default=None, help='Exclude first and last N days')
    parser.add_argument('--exclude_wear_below', type=int, default=None, help='Exclude days with wear below N hours')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--analyze_bouts', action='store_true', help='Analyze walking bouts')
    parser.add_argument("--output", type=str, required=False, help="Output JSON file path")  # Added

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dir = files('gait_metrics.resources.models')
    # gait_detection_model = joblib.load(models_dir / 'gait_detection_model.joblib')
    # step_count_model = joblib.load(models_dir / 'step_count_model.joblib')
    # gait_speed_model = joblib.load(models_dir / 'gait_speed_model.joblib')
    # cadence_model = joblib.load(models_dir / 'cadence_model.joblib')
    # stride_length_model = joblib.load(models_dir / 'stride_length_model.joblib')
    # regularity_model = joblib.load(models_dir / 'regularity_model.joblib')

    gait_detection_model = setup_model(
        net='ElderNet',  # Or 'Resnet' if applicable
        output_size=2,  # For classification (e.g., walk detection)
        is_classification=True,
        trained_model_path=str(models_dir / 'gait_detection_model.pt'),
        device=device
    )

    step_count_model = setup_model(
        net='ElderNet',
        output_size=1,
        is_regression=True,
        num_layers_regressor=1,
        batch_norm=True,
        max_mu=25,
        trained_model_path=str(models_dir / 'step_count_model.pt'),
        device=device
    )

    gait_speed_model = setup_model(
        net='ElderNet',
        output_size=1,
        is_regression=True,
        num_layers_regressor=0,
        max_mu=2.0,
        trained_model_path=str(models_dir / 'gait_speed_model.pt'),
        device=device
    )

    cadence_model = setup_model(
        net='ElderNet',
        output_size=1,
        is_regression=True,
        num_layers_regressor=1,
        max_mu=160.0,
        batch_norm=True,
        trained_model_path=str(models_dir / 'cadence_model.pt'),
        device=device
    )

    stride_length_model = setup_model(
        net='ElderNet',
        output_size=1,
        is_regression=True,
        num_layers_regressor=1,
        max_mu=2.0,
        batch_norm=True,
        trained_model_path=str(models_dir / 'stride_length_model.pt'),
        device=device
    )

    regularity_model = setup_model(
        net='ElderNet',
        output_size=1,
        is_regression=True,
        num_layers_regressor=1,
        max_mu=1.0,
        trained_model_path=str(models_dir / 'regularity_model.pt'),
        device=device
    )

    models = ([
        gait_detection_model,
        step_count_model,
        gait_speed_model,
        cadence_model,
        stride_length_model,
        regularity_model
    ])

    set_model_to_eval(*models)

    filepath = args.file_path

    # Use filepath (loop var) instead of args.filepath
    basename = utils.resolve_path(filepath)[1]

    info = {}
    info['GaitArgs'] = vars(args)

    # Use filepath for reading
    data, info_read = utils.read(
        filepath,
        usecols=args.txyz,
        start_time=args.start,
        end_time=args.end,
        sample_rate=args.sample_rate,
        resample_hz=30,
        verbose=args.verbose
    )
    info.update(info_read)

    if args.exclude_first_last is not None:
        data = utils.drop_first_last_days(data, args.exclude_first_last)

    if args.exclude_wear_below is not None:
        data = utils.flag_wear_below_days(data, args.exclude_wear_below)

    info.update(utils.calculate_wear_stats(data))

    fs = info['ResampleRate']
    window_sec = 10
    window_len = int(window_sec * fs)
    window_step_len = window_len

    processed_acc = data[['x', 'y', 'z']].to_numpy()
    samples_per_day_resampled = int(24 * 60 * 60 * fs)
    num_days = processed_acc.shape[0] // samples_per_day_resampled
    acc_win_all = np.array(
        [processed_acc[i:i + window_len] for i in range(0, len(processed_acc) - window_len + 1, window_step_len)]
    )

    batch_size = 512
    pred_walk = []
    for i in range(0, len(acc_win_all), batch_size):
        batch = acc_win_all[i:i + batch_size]
        with torch.inference_mode():
            batch_pred_walk = process_batch(batch, gait_detection_model, device)
        pred_walk.extend(batch_pred_walk)
    pred_walk = np.array(pred_walk)
    flat_predictions = np.repeat(pred_walk, window_step_len)
    days_array = np.array(np.arange(len(flat_predictions)) // samples_per_day_resampled, dtype=np.int64)

    diff_preds = np.diff(pred_walk, prepend=0, append=0)
    where_bouts_start = np.where(diff_preds == 1)[0]
    where_bouts_end = np.where(diff_preds == -1)[0]
    bouts_durations = (where_bouts_end - where_bouts_start) * window_sec
    if args.analyze_bouts:
        bouts_id_windows = np.zeros_like(pred_walk)
        for bout_idx, (start, end) in enumerate(zip(where_bouts_start, where_bouts_end), 1):
            bouts_id_windows[start:end] = bout_idx
        bout_starts = where_bouts_start * window_len
        bout_starts = np.minimum(bout_starts, len(days_array) - 1)
        bout_days = days_array[bout_starts]
    else:
        bouts_id_windows = np.array([])
        bout_days = np.array([])

    walk_mask = pred_walk == 1
    walking_window_indices = np.where(walk_mask)[0]
    walking_bouts_id = bouts_id_windows[walk_mask] if args.analyze_bouts else np.array([])
    walking_batch = acc_win_all[walk_mask]
    del acc_win_all
    window_starts = walking_window_indices * window_len
    window_starts = np.minimum(window_starts, len(days_array) - 1)
    window_days = days_array[window_starts]

    if walking_batch.size > 0:
        pred_steps = []
        pred_speed = []
        pred_cadence = []
        pred_gait_length = []
        pred_regularity_eldernet = []
        for i in range(0, len(walking_batch), batch_size):
            batch = walking_batch[i:i + batch_size]
            with torch.inference_mode():
                batch_pred_steps = process_batch(batch, step_count_model, device)
                batch_pred_speed = process_batch(batch, gait_speed_model, device)
                batch_pred_cadence = process_batch(batch, cadence_model, device)
                batch_pred_gait_length = process_batch(batch, stride_length_model, device)
                batch_pred_regularity_eldernet = process_batch(batch, regularity_model, device)
            pred_steps.extend(batch_pred_steps)
            pred_speed.extend(batch_pred_speed)
            pred_cadence.extend(batch_pred_cadence)
            pred_gait_length.extend(batch_pred_gait_length)
            pred_regularity_eldernet.extend(batch_pred_regularity_eldernet)

        pred_steps = np.array(np.round(pred_steps))
        pred_speed = np.array(pred_speed)
        pred_cadence = np.array(pred_cadence)
        pred_gait_length = np.array(pred_gait_length)
        pred_gait_length_indirect = np.array(120 * pred_speed / pred_cadence)
        pred_regularity_eldernet = np.array(pred_regularity_eldernet)
        pred_regularity_sp = process_signal_regularity(walking_batch, fs)
        del walking_batch, batch
    else:
        pred_steps = np.array([])
        pred_speed = np.array([])
        pred_cadence = np.array([])
        pred_gait_length = np.array([])
        pred_gait_length_indirect = np.array([])
        pred_regularity_eldernet = np.array([])
        pred_regularity_sp = np.array([])
        window_days = np.array([])

    result = {
        'subject_id': basename,
        'wear_days': num_days,
        'pred_walk': flat_predictions,
        'pred_window_steps': pred_steps,
        'window_days': np.array(window_days),
        'pred_speed': pred_speed,
        'pred_cadence': pred_cadence,
        'pred_gait_length': pred_gait_length,
        'pred_gait_length_indirect': pred_gait_length_indirect,
        'pred_regularity_eldernet': pred_regularity_eldernet,
        'pred_regularity_sp': pred_regularity_sp,
        'bouts_id': walking_bouts_id,
        'bouts_durations': bouts_durations,
        'bout_days': bout_days
    }

    stats = calculate_statistics(result, window_sec, window_len, args.analyze_bouts)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=4, cls=NumpyEncoder)

    return stats

if __name__ == '__main__':
    main()