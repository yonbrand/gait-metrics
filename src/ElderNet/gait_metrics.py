#!/usr/bin/env python3

import argparse
import pathlib
import time
import os
import joblib
import numpy as np
import torch
import json
from scipy.stats import kurtosis, skew
import urllib.request


# import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 1000000000
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt

from . import utils

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
            f'{prefix}range': np.ptp(data)
        }

        if compute_advanced:
            feature_name = prefix[:-1] + '_all_values'
            add_feats = histogram_features(data, GLOBAL_RANGES, bins=n_bins, feature_name=feature_name)
            for i in range(n_bins):
                stats_dict[f'{prefix}prob_bin{i}'] = add_feats['prob'][i]
        return stats_dict

    stats_dict = {
        'sub_id': result['subject_id'],
        'wear_days': result['wear_days'],
        **calc_stats(daily_walking_amounts, 'daily_walking_'),
        **calc_stats(daily_step_count, 'daily_step_count_'),
        **calc_stats(result['pred_speed'], 'gait_speed_', compute_advanced=True),
        **calc_stats(result['pred_cadence'], 'cadence_', compute_advanced=True),
        **calc_stats(result['pred_gait_length'], 'gait_length_', compute_advanced=True),
        **calc_stats(result['pred_gait_length_indirect'], 'gait_length_indirect_', compute_advanced=True),
        **calc_stats(result['pred_regularity_eldernet'], 'regularity_eldernet_', compute_advanced=True),
        **calc_stats(result['pred_regularity_sp'], 'regularity_sp_', compute_advanced=True)
    }

    if analyze_bouts:
        bout_stats = {
            'speeds': [], 'cadences': [], 'gait_lengths': [], 'gait_lengths_indirect': [],
            'regularity_eldernet': [], 'regularity_sp': []
        }
        if len(result['bouts_id']) > 0:
            changes = np.where(np.diff(result['bouts_id']) != 0)[0] + 1
            splits = np.concatenate(([0], changes, [len(result['bouts_id'])]))
            for i in range(len(splits) - 1):
                sl = slice(splits[i], splits[i + 1])
                bout_stats['speeds'].append(np.median(result['pred_speed'][sl]) if sl.stop > sl.start else np.nan)
                bout_stats['cadences'].append(np.median(result['pred_cadence'][sl]) if sl.stop > sl.start else np.nan)
                bout_stats['gait_lengths'].append(
                    np.median(result['pred_gait_length'][sl]) if sl.stop > sl.start else np.nan)
                bout_stats['gait_lengths_indirect'].append(
                    np.median(result['pred_gait_length_indirect'][sl]) if sl.stop > sl.start else np.nan)
                bout_stats['regularity_eldernet'].append(
                    np.median(result['pred_regularity_eldernet'][sl]) if sl.stop > sl.start else np.nan)
                bout_stats['regularity_sp'].append(
                    np.median(result['pred_regularity_sp'][sl]) if sl.stop > sl.start else np.nan)

        stats_dict.update({
            **calc_stats(result['bouts_durations'], 'bout_duration_', compute_advanced=True),
            **calc_stats(bout_stats['speeds'], 'bout_gait_speed_', compute_advanced=True),
            **calc_stats(bout_stats['cadences'], 'bout_cadence_', compute_advanced=True),
            **calc_stats(bout_stats['gait_lengths'], 'bout_gait_length_', compute_advanced=True),
            **calc_stats(bout_stats['gait_lengths_indirect'], 'bout_gait_length_indirect_', compute_advanced=True),
            **calc_stats(bout_stats['regularity_eldernet'], 'bout_regularity_eldernet_', compute_advanced=True),
            **calc_stats(bout_stats['regularity_sp'], 'bout_regularity_sp_', compute_advanced=True)
        })

    return stats_dict

def set_models_to_eval(models):
    for model in models:
        model.eval()

def load_model(filename, device, verbose=True):
    model_dir = pathlib.Path(__file__).parent / 'models'
    model_path = model_dir / filename

    if verbose:
        print(f"Loading model {filename}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model {filename} not found in {model_dir}. Please ensure it is bundled with the applet.")
    model = joblib.load(model_path).to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description='Process gait data.')
    parser.add_argument('--config', default="conf/main_config.yaml", help='Config file')
    parser.add_argument('--filepath', default=None, help='Single input file path (for backward compatibility)')
    parser.add_argument("--filepaths", nargs='+', help="List of input file paths (for batch processing)")
    parser.add_argument('--outdir', default="outputs/", help='Output directory')
    parser.add_argument('--model_path', default=None, help='Custom model path (not used for multiple models)')
    parser.add_argument('--pytorch_device', default="cuda:0", help='PyTorch device')
    parser.add_argument('--sample_rate', default=None, type=int, help='Sample rate')
    parser.add_argument('--txyz', default="time,x,y,z", help='Columns')
    parser.add_argument('--exclude_wear_below', default=None, type=float, help='Exclude wear below')
    parser.add_argument('--exclude_first_last', default='both', help='Exclude first last')
    parser.add_argument('--min_wear_per_day', default=21 * 60, type=int, help='Min wear per day')
    parser.add_argument('--min_wear_per_hour', default=50, type=int, help='Min wear per hour')
    parser.add_argument('--min_wear_per_minute', default=0.5, type=float, help='Min wear per minute')
    parser.add_argument('--min_walk_per_day', default=5, type=int, help='Min walk per day')
    parser.add_argument('--bouts_min_walk', default=0.8, type=float, help='Bouts min walk')
    parser.add_argument('--bouts_max_idle', default=3, type=int, help='Bouts max idle')
    parser.add_argument('--start', default=None, help='Start time')
    parser.add_argument('--end', default=None, help='End time')
    parser.add_argument('--analyze_bouts', default=False, help='Skip bout-level analyses')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    args = parser.parse_args()

    analyze_bouts = args.analyze_bouts
    verbose = not args.quiet
    device = args.pytorch_device

    # Load models once, outside the loop (this is efficient and should not change)
    gait_detection_model = load_model('gait_detection_model_mobD_cap24.joblib', device, verbose)
    step_count_model = load_model('step_count_model.joblib', device, verbose)
    gait_speed_model = load_model('gait_speed_model.joblib', device, verbose)
    cadence_model = load_model('cadence_model.joblib', device, verbose)
    stride_length_model = load_model('stride_length_model.joblib', device, verbose)
    regularity_model = load_model('regularity_model.joblib', device, verbose)

    set_models_to_eval([
        gait_detection_model,
        step_count_model,
        gait_speed_model,
        cadence_model,
        stride_length_model,
        regularity_model
    ])

    # Handle single filepath for backward compatibility
    if args.filepath and not args.filepaths:
        args.filepaths = [args.filepath]
    elif not args.filepaths:
        raise ValueError("No input files provided. Use --filepath or --filepaths.")

    os.makedirs(args.outdir, exist_ok=True)
    all_stats = []

    for filepath in args.filepaths:
        # Use filepath (loop var) instead of args.filepath
        basename = utils.resolve_path(filepath)[1]
        file_outdir = os.path.join(args.outdir, basename)
        os.makedirs(file_outdir, exist_ok=True)

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
            verbose=verbose
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
        if analyze_bouts:
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
        walking_bouts_id = bouts_id_windows[walk_mask] if analyze_bouts else np.array([])
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

        stats = calculate_statistics(result, window_sec, window_len, analyze_bouts)
        all_stats.append(stats)

        # Optional: Save per-file stats (useful for debugging or if DNAnexus needs separate outputs)
        per_file_output = os.path.join(file_outdir, f"{basename}_stats.json")
        with open(per_file_output, 'w') as f:
            json.dump(stats, f, indent=4, cls=utils.NpEncoder)
        if verbose:
            print(f"Stats for {basename} saved to {per_file_output}")

    # Save combined stats (all files in one JSON)
    combined_output_file = os.path.join(args.outdir, "all_stats.json")
    with open(combined_output_file, 'w') as f:
        json.dump(all_stats, f, indent=4, cls=utils.NpEncoder)
    if verbose:
        print(f"Combined stats for all files saved to {combined_output_file}")



if __name__ == '__main__':
    main()