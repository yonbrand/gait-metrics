import argparse
import yaml
import sys
import time
import os
import json
import re
from pathlib import Path
import numpy as np
import torch
from scipy.stats import kurtosis, skew

import utils
from model_utils import get_config, setup_model


# Function to process gait regularity using signal processing
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
        # Gait detection output is 2 columns of logits
        if predictions.shape[1] == 2:
            predictions = torch.argmax(predictions, dim=1)

    return predictions.cpu().numpy()


def calc_second_prediction(pred_walk, window_sec, second_predictions):
    '''
    Map window predictions to seconds based on the middle-second approach
    :param pred_walk: numpy array with the prediction for each window
    :param window_sec: number of seconds per window
    :param second_predictions: zeros nump array for the second-level predictions
    :return: second-level predictions
    '''
    for idx, window_prediction in enumerate(pred_walk):
        center_second = idx + window_sec // 2  # Calculate center second of the window
        if center_second < len(second_predictions):
            second_predictions[center_second] = window_prediction
    return second_predictions


def majority_vote(predictions, window_len, step_len, signal_len):
    """
    Assign predictions to each second using majority vote from overlapping windows.

    :param predictions: List or array of window predictions (0 or 1 for gait detection).
    :param window_len: Length of each window in samples.
    :param step_len: Step length between consecutive windows in samples.
    :param signal_len: Total length of the signal in samples.
    :return: Array of second-wise predictions.
    """
    seconds = signal_len // step_len  # Estimate number of seconds based on step length
    second_predictions = np.zeros(seconds)

    for sec in range(seconds):
        overlapping_window_indices = [
            i for i in range(len(predictions))
            if (sec * step_len) < (i * step_len + window_len) and (sec * step_len) >= (i * step_len)
        ]

        if overlapping_window_indices:
            # Majority vote among overlapping window predictions
            overlapping_preds = [predictions[i] for i in overlapping_window_indices]
            second_predictions[sec] = np.round(np.mean(overlapping_preds))
        else:
            second_predictions[sec] = 0  # Default to 0 if no overlapping windows

    return second_predictions


def merge_gait_bouts(pred_walk, sec_per_sample, min_bout_duration=5, merge_interval=3):
    """
    Post-process gait predictions to merge close bouts and filter short bouts.

    Args:
        pred_walk (np.ndarray): Binary array of walking predictions (1 = walking, 0 = not walking).
        fs (int): Sampling frequency of the predictions (Hz).
        min_bout_duration (int): Minimum duration for a gait bout (in seconds).
        merge_interval (int): Maximum gap duration to merge bouts (in seconds).

    Returns:
        np.ndarray: Post-processed binary array of walking predictions.
    """
    min_bout_samples = int(min_bout_duration * sec_per_sample)
    merge_gap_samples = int(merge_interval * sec_per_sample)

    # Detect start and end indices of gait bouts
    diff_preds = np.diff(np.concatenate([[0], pred_walk, [0]]))  # Add edges to detect transitions
    bout_starts = np.where(diff_preds == 1)[0]
    bout_ends = np.where(diff_preds == -1)[0]

    merged_pred_walk = pred_walk.copy()

    # Iterate through bouts and process gaps
    new_bouts = []
    for i in range(len(bout_starts) - 1):
        start, end = bout_starts[i], bout_ends[i]
        next_start = bout_starts[i + 1]

        # Check if the gap between this bout and the next is within the merge threshold
        if next_start - end <= merge_gap_samples:
            # Merge current bout with the next
            bout_ends[i] = bout_ends[i + 1]
            bout_starts[i + 1] = bout_starts[i]
        else:
            new_bouts.append((start, end))

    # Add the last bout
    if bout_starts[-1] != bout_ends[-1]:
        new_bouts.append((bout_starts[-1], bout_ends[-1]))

    # Apply the new bouts to the prediction array
    merged_pred_walk[:] = 0
    for start, end in new_bouts:
        # Filter out short bouts
        if end - start >= min_bout_samples:
            merged_pred_walk[start:end] = 1

    return merged_pred_walk


def detect_bouts(pred_walk):
    """Detect walking bouts from predicted walk data."""
    diff_preds = np.diff(pred_walk, prepend=0, append=0)
    where_bouts_start = np.where(diff_preds == 1)[0]
    where_bouts_end = np.where(diff_preds == -1)[0]

    bouts_id = np.zeros_like(pred_walk)
    for bout_idx, (start, end) in enumerate(zip(where_bouts_start, where_bouts_end), 1):
        bouts_id[start:end] = bout_idx

    return bouts_id


def calculate_statistics(result, fs, win_step_len):
    # Calculate time metrics once
    seconds_per_sample = win_step_len / fs
    samples_per_day = int(24 * 60 * 60 / seconds_per_sample)
    days = len(result['pred_walk']) // samples_per_day

    # Calculate daily walking statistics
    daily_walking = np.array_split(result['pred_walk'], days)
    daily_walk_amounts = [sum(day) * seconds_per_sample / 60 for day in daily_walking]

    # Calculate step count
    bout_days = result['bout_days']
    bout_steps = result['pred_bout_steps']
    daily_step_count = []
    for day in range(days):
        mask = bout_days == day
        day_bout_steps = bout_steps[mask]
        daily_step_count.append(np.sum(day_bout_steps))

    # Pre-calculate bout masks for all unique bouts
    unique_bouts = np.unique(result['bouts_id'])
    bout_stats = {
        'speeds': [],
        'cadences': [],
        'gait_lengths': [],
        'gait_lengths_indirect': [],
        'regularity_eldernet': [],
        'regularity_sp': []
    }

    # Calculate bout statistics in a single loop
    for bout_id in unique_bouts:
        bout_mask = result['bouts_id'] == bout_id
        for metric, pred_key in [
            ('speeds', 'pred_speed'),
            ('cadences', 'pred_cadence'),
            ('gait_lengths', 'pred_gait_length'),
            ('gait_lengths_indirect', 'pred_gait_length_indirect'),
            ('regularity_eldernet', 'pred_regularity_eldernet'),
            ('regularity_sp', 'pred_regularity_sp')
        ]:
            bout_stats[metric].append(np.median(result[pred_key][bout_mask]))

    def calc_stats(data, prefix='', save_all=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        data = data.flatten()

        if len(data) == 0:
            return {f'{prefix}{stat}': np.nan for stat in
                    ['median', 'mean', 'std', 'p90', 'kurtosis', 'skewness',
                     'var', 'peak_to_peak', 'hist_values', 'hist_bins']}

        stats_dict = {
            f'{prefix}median': np.median(data),
            f'{prefix}mean': np.mean(data),
            f'{prefix}std': np.std(data),
            f'{prefix}p10': np.percentile(data, 10),
            f'{prefix}p20': np.percentile(data, 20),
            f'{prefix}p30': np.percentile(data, 30),
            f'{prefix}p40': np.percentile(data, 40),
            f'{prefix}p50': np.percentile(data, 50),
            f'{prefix}p60': np.percentile(data, 60),
            f'{prefix}p70': np.percentile(data, 70),
            f'{prefix}p80': np.percentile(data, 80),
            f'{prefix}p90': np.percentile(data, 90),
            f'{prefix}kurtosis': kurtosis(data),
            f'{prefix}skewness': skew(data),
            f'{prefix}var': np.var(data),
            f'{prefix}peak_to_peak': np.ptp(data)
        }
        if save_all:
            stats_dict[f'{prefix}all_values'] = json.dumps(data.tolist())
        return stats_dict

    return {
        'sub_id': result['subject_id'],
        'wear_days': result['wear_days'],
        **calc_stats(daily_walk_amounts, 'daily_walking_'),
        **calc_stats(daily_step_count, 'daily_step_count_'),
        **calc_stats(result['bouts_durations'], 'bout_duration_', save_all=True),
        **calc_stats(result['pred_speed'], 'gait_speed_', save_all=True),
        **calc_stats(bout_stats['speeds'], 'bout_gait_speed_', save_all=True),
        **calc_stats(result['pred_cadence'], 'cadence_', save_all=True),
        **calc_stats(bout_stats['cadences'], 'bout_cadence_', save_all=True),
        **calc_stats(result['pred_gait_length'], 'gait_length_', save_all=True),
        **calc_stats(bout_stats['gait_lengths'], 'bout_gait_length_', save_all=True),
        **calc_stats(result['pred_gait_length_indirect'], 'gait_length_indirect_', save_all=True),
        **calc_stats(bout_stats['gait_lengths_indirect'], 'bout_gait_length_indirect_', save_all=True),
        **calc_stats(result['pred_regularity_eldernet'], 'regularity_eldernet_', save_all=True),
        **calc_stats(bout_stats['regularity_eldernet'], 'bout_regularity_eldernet_', save_all=True),
        **calc_stats(result['pred_regularity_sp'], 'regularity_sp_', save_all=True),
        **calc_stats(bout_stats['regularity_sp'], 'bout_regularity_sp_', save_all=True),
        **calc_stats(result['daily_pa_mean'], 'daily_pa_mean_', save_all=True),
        **calc_stats(result['daily_pa_std'], 'daily_pa_std_', save_all=True),
        **calc_stats(result['daily_pa_max'], 'daily_pa_max_', save_all=True),
        **calc_stats(result['daily_pa_min'], 'daily_pa_min_', save_all=True),
        **calc_stats(result['bout_pa_mean'], 'bout_pa_mean_', save_all=True),
        **calc_stats(result['bout_pa_std'], 'bout_pa_std_', save_all=True)
    }


def main():
    # parser = argparse.ArgumentParser(
    #     description="Process gait metrics from wrist-worn accelerometer data.",
    #     add_help=True
    # )
    # parser.add_argument('--config', default='conf/main_config.yaml', help='Path to the YAML config file.')
    # parser.add_argument("filepath", help="Enter file to be processed")
    # parser.add_argument("--outdir", "-o", help="Enter folder location to save output files", default="outputs/")
    # # Model-specific overrides
    # parser.add_argument('--gait_detection_model_path', default=None, help='Override gait detection model path.')
    # parser.add_argument('--gait_speed_model_path', default=None, help='Override gait speed model path.')
    # parser.add_argument('--cadence_model_path', default=None, help='Override cadence model path.')
    # parser.add_argument('--stride_length_model_path', default=None, help='Override gait length model path.')
    # parser.add_argument('--regularity_model_path', default=None, help='Override regularity model path.')
    # parser.add_argument("--pytorch-device", "-d", help="Pytorch device to use, e.g.: 'cpu' or 'cuda:0' (for SSL only)",
    #                     type=str, default='cpu')
    # parser.add_argument("--sample-rate", "-r", help="Sample rate for measurement, otherwise inferred.",
    #                     type=int, default=None)
    # parser.add_argument("--txyz",
    #                     help=("Use this option to specify the column names for time, x, y, z "
    #                           "in the input file, in that order. Use a comma-separated string. "
    #                           "Default: 'time,x,y,z'"),
    #                     type=str, default="time,x,y,z")
    # parser.add_argument("--exclude-wear-below", "-w",
    #                     help="Exclude days with wear time below threshold. Pass values as strings, e.g.: '12H', '30min'. "
    #                          "Default: None (no exclusion)",
    #                     type=str, default=None)
    # parser.add_argument("--exclude-first-last", "-e",
    #                     help="Exclude first, last or both days of data. Default: None (no exclusion)",
    #                     type=str, choices=['first', 'last', 'both'], default=None)
    # parser.add_argument("--min-wear-per-day",
    #                     help="The minimum required wear time (in minutes) for a day to be considered valid.",
    #                     type=float, default=21 * 60)
    # parser.add_argument("--min-wear-per-hour",
    #                     help="The minimum required wear time (in minutes) for an hour bin to be considered valid.",
    #                     type=float, default=50)
    # parser.add_argument("--min-wear-per-minute",
    #                     help="The minimum required wear time (in minutes) for a minute bin to be considered valid.",
    #                     type=float, default=0.5)
    # parser.add_argument("--min-walk-per-day",
    #                     help="The minimum required walking time (in minutes) in a day for metrics calculation.",
    #                     type=float, default=5)
    # parser.add_argument("--start",
    #                     help=("Specicfy a start time for the data to be processed (otherwise, process all). "
    #                           "Pass values as strings, e.g.: '2024-01-01 10:00:00'. Default: None"),
    #                     type=str, default=None)
    # parser.add_argument("--end",
    #                     help=("Specicfy an end time for the data to be processed (otherwise, process all). "
    #                           "Pass values as strings, e.g.: '2024-01-02 09:59:59'. Default: None"),
    #                     type=str, default=None)
    # parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    # args = parser.parse_args()

    args = argparse.Namespace(
        config="conf/main_config.yaml",  # Path to the YAML config file
        filepath=r"C:\Users\yonbr\gait-metrics\tiny-sample.cwa",  # Replace with your test file path
        outdir="outputs/",
        model_path=None,
        force_download=False,
        model_type="ssl",
        pytorch_device="cuda:0",
        sample_rate=None,
        txyz="time,x,y,z",
        exclude_wear_below=None,
        exclude_first_last=None,
        min_wear_per_day=21 * 60,
        min_wear_per_hour=50,
        min_wear_per_minute=0.5,
        min_walk_per_day=5,
        bouts_min_walk=0.8,
        bouts_max_idle=3,
        start=None,
        end=None,
        quiet=False
    )

    before = time.time()
    verbose = not args.quiet

    # Output paths
    basename = utils.resolve_path(args.filepath)[1]
    outdir = os.path.join(args.outdir, basename)
    os.makedirs(outdir, exist_ok=True)

    # Info.json contains high-level summary of the data and results
    info = {}
    info['GaitArgs'] = vars(args)

    # Load file
    data, info_read = utils.read(
        args.filepath,
        usecols=args.txyz,
        start_time=args.start,
        end_time=args.end,
        sample_rate=args.sample_rate,
        resample_hz=30,
        verbose=verbose
    )
    info.update(info_read)

    # Exclusion: first/last days
    if args.exclude_first_last is not None:
        data = utils.drop_first_last_days(data, args.exclude_first_last)

    # Exclusion: days with wear time below threshold
    if args.exclude_wear_below is not None:
        data = utils.flag_wear_below_days(data, args.exclude_wear_below)

    # Update wear time stats after exclusions
    info.update(utils.calculate_wear_stats(data))

    # If no data, save Info.json and exit
    # if len(data) == 0 or data[['x', 'y', 'z']].isna().any(axis=1).all():
    #     # Save Info.json
    #     with open(f"{outdir}/{basename}-Info.json", 'w') as f:
    #         json.dump(info, f, indent=4, cls=utils.NpEncoder)
    #     # Print
    #     print("\nSummary\n-------")
    #     print(json.dumps(
    #         {k: v for k, v in info.items() if not re.search(r'_Weekend|_Weekday|_Hour\d{2}', k)},
    #         indent=4, cls=utils.NpEncoder
    #     ))
    #     print("No data to process. Exiting early...")
    #     sys.exit(0)

    # Load the full config efficiently (single read)
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Model path overrides
    # if args.gait_detection_model_path:
    #     cfg['gait_detection']['trained_model_path'] = args.gait_detection_model_path
    # if args.step_count_model_path:
    #     cfg['step_count']['trained_model_path'] = args.step_count_model_path
    # if args.gait_speed_model_path:
    #     cfg['gait_speed']['trained_model_path'] = args.gait_speed_model_path
    # if args.cadence_model_path:
    #     cfg['cadence']['trained_model_path'] = args.cadence_model_path
    # if args.stride_length_model_path:
    #     cfg['stride_length']['trained_model_path'] = args.stride_length_model_path
    # if args.regularity_model_path:
    #     cfg['regularity']['trained_model_path'] = args.regularity_model_path

    # Load configs for each model (efficient: sub-dict extraction from loaded cfg)
    gait_detection_model_cfg = get_config(cfg, model_type='gait_detection')
    step_count_cfg = get_config(cfg, model_type='step_count')
    gait_speed_model_cfg = get_config(cfg, model_type='gait_speed')
    cadence_model_cfg = get_config(cfg, model_type='cadence')
    stride_length_model_cfg = get_config(cfg, model_type='stride_length')
    regularity_model_cfg = get_config(cfg, model_type='regularity')
    fs = info['ResampleRate']
    window_sec = cfg['window_len'] # Length of the window in seconds
    window_len = int(window_sec * fs)  # Length of the window in samples
    device = args.pytorch_device

    # Setup models
    gait_detection_model = setup_model(
        net=gait_detection_model_cfg['net'],
        eldernet_linear_output=gait_detection_model_cfg['feature_vector_size'] if gait_detection_model_cfg[
                                                                                      'net'] == 'ElderNet' else None,
        is_classification=True,
        pretrained=gait_detection_model_cfg['pretrained'],
        trained_model_path=gait_detection_model_cfg['trained_model_path'],
        output_size=gait_detection_model_cfg['output_size'],
        device=device)

    step_count_model = setup_model(
        net=step_count_cfg['net'],
        eldernet_linear_output=step_count_cfg['feature_vector_size'] if step_count_cfg[
                                                                                  'net'] == 'ElderNet' else None,
        is_regression=True,
        max_mu=step_count_cfg['max_mu'],
        num_layers_regressor=step_count_cfg['num_layers_regressor'],
        batch_norm=step_count_cfg['batch_norm'],
        pretrained=step_count_cfg['pretrained'],
        trained_model_path=step_count_cfg['trained_model_path'],
        device=device)

    gait_speed_model = setup_model(
        net=gait_speed_model_cfg['net'],
        eldernet_linear_output=gait_speed_model_cfg['feature_vector_size'] if gait_speed_model_cfg[
                                                                                  'net'] == 'ElderNet' else None,
        is_regression=True,
        max_mu=gait_speed_model_cfg['max_mu'],
        num_layers_regressor=gait_speed_model_cfg['num_layers_regressor'],
        batch_norm=gait_speed_model_cfg['batch_norm'],
        pretrained=gait_speed_model_cfg['pretrained'],
        trained_model_path=gait_speed_model_cfg['trained_model_path'],
        device=device)

    cadence_model = setup_model(
        net=cadence_model_cfg['net'],
        eldernet_linear_output=cadence_model_cfg['feature_vector_size'] if cadence_model_cfg[
                                                                               'net'] == 'ElderNet' else None,
        is_regression=True,
        max_mu=cadence_model_cfg['max_mu'],
        num_layers_regressor=cadence_model_cfg['num_layers_regressor'],
        batch_norm=cadence_model_cfg['batch_norm'],
        pretrained=cadence_model_cfg['pretrained'],
        trained_model_path=cadence_model_cfg['trained_model_path'],
        device=device)

    stride_length_model = setup_model(
        net=stride_length_model_cfg['net'],
        eldernet_linear_output=stride_length_model_cfg['feature_vector_size'] if stride_length_model_cfg[
                                                                                     'net'] == 'ElderNet' else None,
        is_regression=True,
        max_mu=stride_length_model_cfg['max_mu'],
        num_layers_regressor=stride_length_model_cfg['num_layers_regressor'],
        batch_norm=stride_length_model_cfg['batch_norm'],
        pretrained=stride_length_model_cfg['pretrained'],
        trained_model_path=stride_length_model_cfg['trained_model_path'],
        device=device)

    regularity_model = setup_model(
        net=regularity_model_cfg['net'],
        eldernet_linear_output=regularity_model_cfg['feature_vector_size'] if regularity_model_cfg[
                                                                                  'net'] == 'ElderNet' else None,
        is_regression=True,
        max_mu=regularity_model_cfg['max_mu'],
        num_layers_regressor=regularity_model_cfg['num_layers_regressor'],
        batch_norm=regularity_model_cfg['batch_norm'],
        pretrained=regularity_model_cfg['pretrained'],
        trained_model_path=regularity_model_cfg['trained_model_path'],
        device=device)

    processed_acc = data[['x', 'y', 'z']].to_numpy()
    samples_per_day_resampled = int(24 * 60 * 60 * fs)
    num_days = processed_acc.shape[0] // samples_per_day_resampled
    acc_win_all = np.array(
        [processed_acc[i:i + 300] for i in range(0, len(processed_acc) - 300 + 1, 300)]
    )
    # Process data in batches
    batch_size = acc_win_all.shape[0] // gait_detection_model_cfg['num_batches']
    pred_walk = []
    for i in range(0, len(acc_win_all), batch_size):
        batch = acc_win_all[i:i + batch_size]
        batch_pred_walk = process_batch(batch, gait_detection_model, device)
        pred_walk.extend(batch_pred_walk)

    pred_walk = np.array(pred_walk)
    # Map window predictions to seconds based on the middle-second approach
    second_predictions = np.zeros(len(processed_acc) // fs, dtype=int)
    second_predictions_walk = calc_second_prediction(pred_walk, window_sec, second_predictions)
    # Merge near bouts
    sec_per_sample = window_len / fs  # 1 for 90% overlap, 10 for no overlap
    merged_predictions_walk = merge_gait_bouts(second_predictions_walk, sec_per_sample, 10, 3)
    flat_predictions = np.repeat(merged_predictions_walk, fs)
    bouts_id = detect_bouts(flat_predictions)
    bout_mask = bouts_id != 0
    unq_bouts = np.unique(bouts_id[bout_mask])

    bouts_win_all = []
    walking_bouts_id = []
    bouts_durations = []
    bout_pa_means = []
    bout_pa_stds = []
    for bout in unq_bouts:
        bout_predictions = np.where(bouts_id == bout)[0]
        acc_bout = processed_acc[bout_predictions]
        bout_win = np.array(
            [acc_bout[i:i + window_len] for i in range(0, len(acc_bout) - window_len + 1, window_len)]
        )
        bout_duration = int(len(acc_bout) / fs)
        bouts_win_all.append(bout_win)
        current_bout_id = np.ones(bout_win.shape[0]) * bout
        walking_bouts_id.append(current_bout_id)
        bouts_durations.append(bout_duration)

        # Compute bout-level PA features
        bout_magnitude = np.linalg.norm(acc_bout, axis=1)
        bout_pa_means.append(np.mean(bout_magnitude))
        bout_pa_stds.append(np.std(bout_magnitude))

    if walking_bouts_id:
        walking_bouts_id = np.concatenate(walking_bouts_id)
    else:
        walking_bouts_id = np.array([])

    walking_batch = np.concatenate(bouts_win_all) if bouts_win_all else np.array([])

    if walking_batch.size > 0:
        # Process walking windows with gait quality models
        pred_steps = []
        pred_speed = []
        pred_cadence = []
        pred_gait_length = []
        pred_regularity_eldernet = []
        for i in range(0, len(walking_batch), batch_size):
            batch = walking_batch[i:i + batch_size]
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

        pred_steps = np.array(np.round(pred_steps))  # round the predictions to whole numbers
        pred_speed = np.array(pred_speed)
        pred_cadence = np.array(pred_cadence)
        pred_gait_length = np.array(pred_gait_length)
        pred_gait_length_indirect = np.array(120 * pred_speed / pred_cadence)
        pred_regularity_eldernet = np.array(pred_regularity_eldernet)
        pred_regularity_sp = process_signal_regularity(walking_batch, fs)

        # Step count processing
        # Find the day correspond to each bout
        days_array = np.repeat(np.arange(num_days), len(pred_walk) // num_days)
        day_per_bout = []
        pred_bout_steps = []
        for bout_idx, bout in enumerate(unq_bouts):
            bout_samples = np.where(bouts_id == bout)[0]
            if len(bout_samples) > 0:
                bout_day = days_array[bout_samples[0]]  # Day of the first sample
                day_per_bout.append(bout_day)
            # Existing step calculation
            bout_predictions = np.where(walking_bouts_id == bout)[0]
            bout_len = len(bout_predictions)
            median_bout_steps = np.median(pred_steps[bout_predictions])
            adj_factor = 0.1 * (bout_len - 1) + 1
            adj_bout_steps = adj_factor * median_bout_steps
            pred_bout_steps.append(adj_bout_steps)
        # Convert to np arrays
        pred_bout_steps = np.array(pred_bout_steps)
        day_per_bout = np.array(day_per_bout)

    else:
        pred_speed = np.array([])
        pred_cadence = np.array([])
        pred_gait_length = np.array([])
        pred_gait_length_indirect = np.array([])
        pred_regularity_eldernet = np.array([])
        pred_regularity_sp = np.array([])
        pred_bout_steps = np.array([])
        day_per_bout = np.array([])

    file_path = Path(args.filepath)
    sub_id = '_'.join(file_path.stem.split('-')[:2])

    # Create result dictionary and include new PA features
    result = {
        'subject_id': sub_id,
        'wear_days': num_days,
        'bout_days': day_per_bout,
        'pred_walk': pred_walk,
        'pred_bout_steps': pred_bout_steps,
        'pred_speed': pred_speed,
        'pred_cadence': pred_cadence,
        'pred_gait_length': pred_gait_length,
        'pred_gait_length_indirect': pred_gait_length_indirect,
        'pred_regularity_eldernet': pred_regularity_eldernet,
        'pred_regularity_sp': pred_regularity_sp,
        'bouts_id': walking_bouts_id,
        'bouts_durations': bouts_durations
    }

    # Calculate statistics (including PA features)
    stats = calculate_statistics(result, fs, window_len)


if __name__ == '__main__':
    main()
