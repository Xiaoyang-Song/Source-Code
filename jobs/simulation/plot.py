import os
import re
import argparse
import numpy as np
from matplotlib import pyplot as plt


# =========================
# Defaults
# =========================
DEFAULT_SIGMAS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
DEFAULT_PS = [4, 6, 8]
DEFAULT_NS = [20, 25, 30]

METRICS = ['F1', 'I', 'II']
METHODS = ['DCCA', 'PCA', 'PLS', 'AE', 'SIMCLR', 'SIMCLR-CLIP', 'CLS']

METRIC_NAME = {
    'F1': 'F1 Score',
    'I': 'Type-I Error',
    'II': 'Type-II Error'
}

LABELS = {
    'DCCA': 'OURS',
    'PCA': r'PCA+$T^2$',
    'PLS': r'PLS+$T^2$',
    'AE': r'AE+$T^2$',
    'SIMCLR': r'SimCLR+$T^2$',
    'SIMCLR-CLIP': r'SimCLR-CLIP+$T^2$',
    'CLS': 'CLS'
}

MARKERS = {
    'DCCA': 's',
    'PCA': 'o',
    'PLS': 'x',
    'AE': '^',
    'SIMCLR': 'D',
    'SIMCLR-CLIP': 'v',
    'CLS': 'P'
}

COLORS = {
    'DCCA': 'tab:blue',
    'PCA': 'tab:orange',
    'PLS': 'tab:green',
    'AE': 'tab:red',
    'SIMCLR': 'tab:purple',
    'SIMCLR-CLIP': 'tab:brown',
    'CLS': 'tab:gray'
}


# =========================
# CLI helpers
# =========================
def parse_number_spec(spec, cast_func=int):
    """
    Supports:
      "4,6,8"
      "4-8"
      "4-8:2"
      "20"
    """
    if spec is None:
        return None

    spec = spec.strip()
    if not spec:
        return None

    values = []

    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue

        if '-' in part:
            if ':' in part:
                range_part, step_part = part.split(':')
                start_str, end_str = range_part.split('-')
                start = cast_func(start_str)
                end = cast_func(end_str)
                step = cast_func(step_part)
            else:
                start_str, end_str = part.split('-')
                start = cast_func(start_str)
                end = cast_func(end_str)
                step = 1 if cast_func is int else 0.1

            if cast_func is int:
                values.extend(list(range(start, end + 1, step)))
            else:
                current = start
                while current <= end + 1e-12:
                    values.append(round(current, 10))
                    current += step
        else:
            values.append(cast_func(part))

    values = sorted(set(values))
    return values


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--base-dir',
        type=str,
        default=os.path.join('checkpoint', 'simulation-circle-crack', 'sensitivity_analysis'),
        help='Directory containing subfolders named like p-n'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=os.path.join('Document', 'simulation-circle-crack', 'sensitivity_analysis'),
        help='Directory to save plots'
    )

    parser.add_argument(
        '--p-values',
        type=str,
        default=None,
        help='p values to include, e.g. "4,6,8" or "4-8:2"'
    )
    parser.add_argument(
        '--n-values',
        type=str,
        default=None,
        help='n values to include, e.g. "20,25,30" or "20-30:5"'
    )
    parser.add_argument(
        '--sigmas',
        type=str,
        default=None,
        help='sigma values to include, e.g. "1.0,1.2,1.4" or "1.0-2.0:0.1"'
    )

    parser.add_argument(
        '--methods',
        nargs='+',
        default=METHODS,
        choices=METHODS,
        help='Methods to include in plots'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=METRICS,
        choices=METRICS,
        help='Metrics to plot'
    )

    parser.add_argument(
        '--plot-best-baseline',
        action='store_true',
        help='Also plot best baseline curves across selected p,n combinations'
    )
    parser.add_argument(
        '--no-shaded-error',
        action='store_true',
        help='Disable error bands entirely'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print parsed values'
    )

    return parser.parse_args()


# =========================
# Parsing helpers
# =========================
def find_float_in_line(line):
    m = re.search(r'-?\d+(?:\.\d+)?', line)
    return float(m.group()) if m else None


def extract_metric_from_block(block_lines, label):
    for i, line in enumerate(block_lines):
        stripped = line.strip()

        if stripped.startswith(label):
            if ':' in line:
                after_colon = line.split(':', 1)[1]
                val = find_float_in_line(after_colon)
                if val is not None:
                    return val

            for j in range(i + 1, len(block_lines)):
                nxt = block_lines[j].strip()
                if nxt == '':
                    continue
                val = find_float_in_line(nxt)
                if val is not None:
                    return val
                break

    raise ValueError(f'Could not find metric "{label}" in block.')


def get_block(lines, header, stop_headers):
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i + 1
            break

    if start is None:
        raise ValueError(f'Header "{header}" not found.')

    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].strip() in stop_headers:
            end = i
            break

    return lines[start:end]


def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    dcca_block = get_block(
        lines,
        header='Merged Regime...',
        stop_headers=['Baseline...']
    )

    cls_block = get_block(
        lines,
        header='Baseline...',
        stop_headers=['Hotelling T2']
    )

    pca_block = get_block(
        lines,
        header='PCA',
        stop_headers=['PLS']
    )
    pls_block = get_block(
        lines,
        header='PLS',
        stop_headers=['AutoEncoder']
    )
    ae_block = get_block(
        lines,
        header='AutoEncoder',
        stop_headers=['SIMCLR']
    )
    simclr_block = get_block(
        lines,
        header='SIMCLR',
        stop_headers=['SIMCLR-CLIP']
    )
    simclr_clip_block = get_block(
        lines,
        header='SIMCLR-CLIP',
        stop_headers=[]
    )

    parsed = {
        'DCCA': {
            'I': extract_metric_from_block(dcca_block, 'False Alarm Rate'),
            'II': extract_metric_from_block(dcca_block, 'Mis-Detection Rate'),
            'F1': extract_metric_from_block(dcca_block, 'F1')
        },
        'CLS': {
            'I': extract_metric_from_block(cls_block, 'Mean False Alarms'),
            'II': extract_metric_from_block(cls_block, 'Mean Mis Detections'),
            'F1': extract_metric_from_block(cls_block, 'Mean F1')
        },
        'PCA': {
            'I': extract_metric_from_block(pca_block, 'False Alarms'),
            'II': extract_metric_from_block(pca_block, 'Mis Detections'),
            'F1': extract_metric_from_block(pca_block, 'F1')
        },
        'PLS': {
            'I': extract_metric_from_block(pls_block, 'False Alarms'),
            'II': extract_metric_from_block(pls_block, 'Mis Detections'),
            'F1': extract_metric_from_block(pls_block, 'F1')
        },
        'AE': {
            'I': extract_metric_from_block(ae_block, 'False Alarms'),
            'II': extract_metric_from_block(ae_block, 'Mis Detections'),
            'F1': extract_metric_from_block(ae_block, 'F1')
        },
        'SIMCLR': {
            'I': extract_metric_from_block(simclr_block, 'False Alarms'),
            'II': extract_metric_from_block(simclr_block, 'Mis Detections'),
            'F1': extract_metric_from_block(simclr_block, 'F1')
        },
        'SIMCLR-CLIP': {
            'I': extract_metric_from_block(simclr_clip_block, 'False Alarms'),
            'II': extract_metric_from_block(simclr_clip_block, 'Mis Detections'),
            'F1': extract_metric_from_block(simclr_clip_block, 'F1')
        }
    }

    return parsed


# =========================
# Plot helpers
# =========================
def plot_mean_std_curves(results, methods, metrics, sigmas, save_dir, draw_error=True):
    for metric in metrics:
        plt.figure(figsize=(8, 6))

        for method in methods:
            means = []
            stds = []

            for sigma in sigmas:
                vals = results[method][metric][sigma]
                if len(vals) == 0:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))

            means = np.array(means)
            stds = np.array(stds)

            plt.plot(
                sigmas,
                means,
                marker=MARKERS[method],
                label=LABELS[method],
                color=COLORS[method],
                linewidth=2
            )

            if draw_error:
                valid = ~(np.isnan(means) | np.isnan(stds))
                if np.any(valid):
                    plt.fill_between(
                        np.array(sigmas)[valid],
                        (means - stds)[valid],
                        (means + stds)[valid],
                        color=COLORS[method],
                        alpha=0.2
                    )

        plt.title(rf'{METRIC_NAME[metric]} vs Noise ($\delta$)', fontsize=16)
        plt.xlabel(r'Noise Level ($\delta$)', fontsize=14)
        plt.ylabel(f'{METRIC_NAME[metric]} (%)', fontsize=14)
        plt.xticks(sigmas)
        plt.grid(alpha=0.3)
        plt.legend()

        plt.savefig(
            os.path.join(save_dir, f'plot-{metric}-sensitivity.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


def compute_best_baseline(results, methods, metrics, sigmas):
    best_baseline = {
        method: {metric: [] for metric in metrics}
        for method in methods if method != 'DCCA'
    }

    for method in best_baseline:
        for metric in metrics:
            for sigma in sigmas:
                vals = results[method][metric][sigma]
                if len(vals) == 0:
                    best_baseline[method][metric].append(np.nan)
                else:
                    if metric == 'F1':
                        best_baseline[method][metric].append(np.max(vals))
                    else:
                        best_baseline[method][metric].append(np.min(vals))

    return best_baseline


def plot_best_baseline_curves(results, best_baseline, methods, metrics, sigmas, save_dir, draw_error=True):
    for metric in metrics:
        plt.figure(figsize=(8, 6))

        if 'DCCA' in methods:
            dcca_means = []
            dcca_stds = []

            for sigma in sigmas:
                vals = results['DCCA'][metric][sigma]
                if len(vals) == 0:
                    dcca_means.append(np.nan)
                    dcca_stds.append(np.nan)
                else:
                    dcca_means.append(np.mean(vals))
                    dcca_stds.append(np.std(vals))

            dcca_means = np.array(dcca_means)
            dcca_stds = np.array(dcca_stds)

            plt.plot(
                sigmas,
                dcca_means,
                marker=MARKERS['DCCA'],
                label=LABELS['DCCA'],
                color=COLORS['DCCA'],
                linewidth=2
            )

            if draw_error:
                valid = ~(np.isnan(dcca_means) | np.isnan(dcca_stds))
                if np.any(valid):
                    plt.fill_between(
                        np.array(sigmas)[valid],
                        (dcca_means - dcca_stds)[valid],
                        (dcca_means + dcca_stds)[valid],
                        color=COLORS['DCCA'],
                        alpha=0.2
                    )

        for method in methods:
            if method == 'DCCA':
                continue
            if method not in best_baseline:
                continue

            plt.plot(
                sigmas,
                best_baseline[method][metric],
                marker=MARKERS[method],
                label=LABELS[method],
                color=COLORS[method],
                linewidth=2
            )

        plt.title(rf'{METRIC_NAME[metric]} vs Noise ($\delta$)', fontsize=16)
        plt.xlabel(r'Noise Level ($\delta$)', fontsize=14)
        plt.ylabel(f'{METRIC_NAME[metric]} (%)', fontsize=14)
        plt.xticks(sigmas)
        plt.grid(alpha=0.3)
        plt.legend()

        plt.savefig(
            os.path.join(save_dir, f'plot-{metric}-sensitivity_best.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


# =========================
# Main
# =========================
def main():
    args = get_args()

    ps = parse_number_spec(args.p_values, int) if args.p_values else DEFAULT_PS
    ns = parse_number_spec(args.n_values, int) if args.n_values else DEFAULT_NS
    sigmas = parse_number_spec(args.sigmas, float) if args.sigmas else DEFAULT_SIGMAS
    methods = args.methods
    metrics = args.metrics

    os.makedirs(args.save_dir, exist_ok=True)

    results = {
        method: {
            metric: {sigma: [] for sigma in sigmas}
            for metric in metrics
        }
        for method in methods
    }

    selected_pairs = [(p, n) for p in ps for n in ns]

    draw_error = (len(selected_pairs) > 1) and (not args.no_shaded_error)

    print('Using p values:', ps)
    print('Using n values:', ns)
    print('Using sigma values:', sigmas)
    print('Using methods:', methods)
    print('Using metrics:', metrics)
    print('Selected (p, n) combinations:', selected_pairs)
    print('Draw error bands:', draw_error)

    for p in ps:
        for n in ns:
            combo_dir = os.path.join(args.base_dir, f'{p}-{n}')

            for sigma in sigmas:
                log_file_path = os.path.join(combo_dir, f'results-sigma{sigma}.log')

                if not os.path.exists(log_file_path):
                    print(f'Missing {log_file_path}')
                    continue

                try:
                    parsed = parse_log_file(log_file_path)

                    if args.verbose:
                        print(f'\nParsed: p={p}, n={n}, sigma={sigma}')
                        for method in methods:
                            print(method, parsed[method])

                    for method in methods:
                        for metric in metrics:
                            results[method][metric][sigma].append(parsed[method][metric])

                except Exception as e:
                    print(f'Failed to parse {log_file_path}: {e}')

    plot_mean_std_curves(
        results=results,
        methods=methods,
        metrics=metrics,
        sigmas=sigmas,
        save_dir=args.save_dir,
        draw_error=draw_error
    )

    if args.plot_best_baseline:
        best_baseline = compute_best_baseline(
            results=results,
            methods=methods,
            metrics=metrics,
            sigmas=sigmas
        )
        plot_best_baseline_curves(
            results=results,
            best_baseline=best_baseline,
            methods=methods,
            metrics=metrics,
            sigmas=sigmas,
            save_dir=args.save_dir,
            draw_error=draw_error
        )


if __name__ == '__main__':
    main()