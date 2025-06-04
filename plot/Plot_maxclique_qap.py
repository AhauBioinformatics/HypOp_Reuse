import numpy as np
import math
import matplotlib.pyplot as plt
import random
import re

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 2

linewidth = 4
markersize = 16
markeredgewidth = 4
s = 200

label_fontsize = 54
tick_fontsize = 44
legend_fontsize = 48
annotate_fontsize = 90

subplot_labels = ['a', 'b', 'c']

plt.figure(figsize=(34, 26))

def parse_log_file(file_path):
    data = {}
    pattern = r'n(\d+)_random_p(\d+)\.txt:.*, clique_size: (\d+), running_time: ([\d\.]+)'
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                n = int(match.group(1))
                p = int(match.group(2))
                clique_size = float(match.group(3))/n
                running_time = float(match.group(4))

                if n not in data:
                    data[n] = {}

                if p not in data[n]:
                    data[n][p] = {'clique_size': [], 'running_time': []}

                data[n][p]['clique_size'].append(clique_size)
                data[n][p]['running_time'].append(running_time)

    return data

def calculate_averages(data):
    averages = {}

    for n, p_values in data.items():
        averages[n] = {'p': [], 'avg_clique_size': [], 'avg_running_time': []}
        for p, values in p_values.items():
            avg_clique_size = np.mean(values['clique_size'])
            avg_running_time = np.mean(values['running_time'])

            averages[n]['p'].append(p)
            averages[n]['avg_clique_size'].append(avg_clique_size)
            averages[n]['avg_running_time'].append(avg_running_time)

    return averages

def process_log(paths, d_value):
    log_data = {}
    for path in paths:
        with open(path) as f:
            Log = f.readlines()
        for lines in Log:
            temp = lines.split(',')
            temp0 = temp[0].split(':')
            temp1 = temp[1].split(':')
            temp2 = temp[2].split(':')
            name = temp0[2]
            res = int(temp1[1][1:])
            if name not in log_data:
                log_data[name] = {'res': [], 'time': []}
            log_data[name]['res'].append(abs(int(res)))
            time = float(temp2[1][1:-1])
            log_data[name]['time'].append(time)
            n = int(name[6:-6])
            log_data[name]['n'] = n
            log_data[name]['d'] = d_value
    return log_data

hy_files = 'log/reusibility/maxclique_data.log'

log_d = parse_log_file(hy_files)

averages = calculate_averages(log_d)
plt.subplot(2, 2, 1)
for n, values in averages.items():
    plt.plot(values['p'], values['avg_clique_size'], label=f'n={n}', marker='o', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.7)

yticks = np.arange(0, 0.09, 0.015)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(yticks, fontsize=tick_fontsize)
plt.ylabel('Clique size over node count', fontsize=label_fontsize)
plt.xlabel('Edge density (%)', fontsize=label_fontsize)
plt.legend(loc='upper left', fontsize=legend_fontsize, frameon=False)
plt.annotate(subplot_labels[0], xy=(-0.18, 1.10), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

# plt.title('Average Clique Size vs. Connection Probability for Different n')
# plt.savefig('./res/plots/maxclique_sorted.png')
# plt.show()

plt.subplot(2, 2, 2)
for n, values in averages.items():
    plt.plot(values['p'], values['avg_running_time'], label=f'n={n}', linestyle='-', marker='o', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.7)
plt.xticks(fontsize=tick_fontsize)
yticks = np.arange(0, 1200, 200)
plt.yticks(yticks, fontsize=tick_fontsize)
plt.ylabel('Run time (s)', fontsize=label_fontsize)
plt.xlabel('Edge density (%)', fontsize=label_fontsize)
plt.legend(loc='upper left', fontsize=legend_fontsize, frameon=False)
plt.annotate(subplot_labels[1], xy=(-0.18, 1.10), xycoords='axes fraction',
                 xytext=(-10, 5), textcoords='offset points',
                 ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)
# plt.savefig('./res/plots/maxclique_time.svg')
# plt.show()

random.seed(44)

path = 'log/reusibility/qap.log'
with open(path) as f:
    Log = f.readlines()

log_d = {}

for line in Log:
    temp = line.split(',')
    temp1 = temp[0].split(':')
    temp2 = temp[1].split(':')

    cls = temp1[1]
    name = temp1[2]
    try:
        rel_loss = float(temp2[1][1:-1])
    except ValueError:
        continue

    if math.isinf(rel_loss):
        continue

    if name not in log_d:
        log_d[name] = []
    log_d[name].append(rel_loss)

avg_rel_loss = {}
for name, losses in log_d.items():
    avg_rel_loss[name] = sum(losses) / len(losses)

names = list(avg_rel_loss.keys())
avg_losses = list(avg_rel_loss.values())

n_points = 60

random_indices = random.sample(range(len(names)), n_points)
selected_names = [names[i] for i in random_indices]
selected_losses = [avg_losses[i] for i in random_indices]
selected_names, selected_losses = zip(*sorted(zip(selected_names, selected_losses)))

plt.subplot(2, 1, 2)
plt.plot(selected_names, selected_losses, marker='x', linestyle='-', color='#3483BB', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.scatter(selected_names, selected_losses, color='#3483BB', marker='x', label=f'HypOp', s=s, linewidth=linewidth)
# plt.title(f'Randomly selected {n_points} points from Average rel_loss', fontsize=16)
# plt.xlabel(fontsize=label_fontsize)
plt.ylabel('Relative loss', fontsize=label_fontsize)
plt.xticks(rotation=90, fontsize=35)
plt.yticks(fontsize=tick_fontsize)
plt.axhline(y=0, color='r', linestyle='--', label='Known optimal solution', linewidth=linewidth)
plt.legend(fontsize=legend_fontsize, frameon=False)
plt.annotate(subplot_labels[2], xy=(-0.08, 1.10), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)
plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)
plt.savefig('figure/maxclique_qap.svg')

plt.show()

