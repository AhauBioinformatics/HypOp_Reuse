import matplotlib.pyplot as plt
import re
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 2

linewidth = 4
markersize = 16
s = 200

label_fontsize = 53
tick_fontsize = 48
legend_fontsize = 44
annotate_fontsize = 60

def read_data(file_path):
    log_d = {}
    current_name = ""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                match = re.match(r'Hyp_(\d+)_(\d+).txt', line)
                if match:
                    n = int(match.group(1))
                    m = int(match.group(2))
                    current_name = match.group()
                    log_d[current_name] = {'scores': [], 'times': [], 'm': m, 'n': n}
                elif 'Iteration' in line:
                    parts = line.split(', ')
                    score = int(parts[1].split(': ')[1])
                    time = float(parts[2].split(': ')[1])
                    log_d[current_name]['scores'].append(score)
                    log_d[current_name]['times'].append(time)
    return log_d


def plot_scores_and_times(log_d):
    iters = list(range(0, 31, 5))

    plt.figure(figsize=(28, 14))
    subplot_labels = ['a', 'b']

    plt.subplot(1, 2, 1)
    for name, data in log_d.items():
        scores = np.array(data['scores'])
        m = data['m']
        y_axis = scores / m
        n = data['n']
        plt.plot(iters, y_axis, marker='o', label=f'n={n}', linewidth=linewidth, markersize=markersize, alpha=0.7)
    yticks = np.arange(0.940, 0.985, 0.01)
    plt.yticks(yticks, fontsize=tick_fontsize)
    plt.xticks(iters, rotation=0, fontsize=tick_fontsize)
    plt.ylabel('Cut size over hyperedge count', fontsize=label_fontsize)
    plt.xlabel('Iteration count', fontsize=label_fontsize)
    plt.legend(loc='lower right', frameon=False, fontsize=legend_fontsize)
    plt.annotate(subplot_labels[0], xy=(-0.20, 1.10), xycoords='axes fraction',
                 xytext=(-10, 5), textcoords='offset points',
                 ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

    plt.subplot(1, 2, 2)
    for name, data in log_d.items():
        times = np.array(data['times'])
        n = data['n']
        plt.plot(iters, times, marker='o', label=f'n={n}', linewidth=linewidth, markersize=markersize, alpha=0.7)
    plt.ylabel('Run time (s)', fontsize=label_fontsize)
    plt.xlabel('Iteration count', fontsize=label_fontsize)
    plt.xticks(iters, rotation=0, fontsize=tick_fontsize)
    yticks = np.arange(0, 3500, 500)
    plt.yticks(yticks, fontsize=tick_fontsize)
    plt.legend(loc='upper left', frameon=False, fontsize=legend_fontsize)
    plt.annotate(subplot_labels[1], xy=(-0.20, 1.10), xycoords='axes fraction',
                 xytext=(-10, 5), textcoords='offset points',
                 ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

    plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
    plt.savefig('figure/iterations.svg')
    plt.show()


if __name__ == "__main__":
    file_path = 'iteration.txt'
    log_d = read_data(file_path)
    plot_scores_and_times(log_d)