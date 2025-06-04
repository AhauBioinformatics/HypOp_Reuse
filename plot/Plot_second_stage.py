import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 3

linewidth = 4
markersize = 16
markeredgewidth = 4

label_fontsize = 68
tick_fontsize = 60
# legend_fontsize = 42
annotate_fontsize = 90
labelpad = 15

xpad = 20

handles = []
labels = []

data = {}
with open('second_stage.txt', 'r', encoding='utf - 8') as f:
    lines = f.readlines()
current_feed_value = None
for line in lines:
    if line.startswith('feed_value:'):
        current_feed_value = int(line.split(':')[1].strip())
        if current_feed_value not in data:
            data[current_feed_value] = {'time': {}, 'performance': {}}
    elif ':' in line:
        key, value = line.split(':')
        time_str, performance_str = value.split()
        time = float(time_str[: - 1])
        performance = int(performance_str.replace(',', '')) / 19990

        if key in ['SA', 'ACO', 'GA', 'PSO']:
            data[current_feed_value]['time'][key] = time
            data[current_feed_value]['performance'][key] = performance
        else:
            data[current_feed_value]['time'][key] = time
            data[current_feed_value]['performance'][key] = performance

algorithms = ['SA', 'PSO', 'GA', 'ACO']
categories = ['Original', 'Round', 'Rand1', 'Rand2', 'Rand2']
GPU_count = ['1GPU', '2GPUs', '4GPUs', '8GPUs']
feed_value_mapping = {0: 1, 25: 2, 50: 3, 75: 4, 100: 5}

plt.figure(figsize=(44, 28))

subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']

plt.subplot(2, 3, 1)
for feed_value in sorted(data.keys()):
    x_labels = GPU_count
    y_values = [data[feed_value]['performance'].get(method, None) for method in GPU_count]
    new_label = feed_value_mapping.get(feed_value, feed_value)
    handle, = plt.plot(x_labels, y_values, marker='o', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.7)
    handles.append(handle)
    labels.append(f'Trail {new_label}')
yticks = np.arange(0.63, 0.68, 0.01)
plt.yticks(yticks, fontsize=tick_fontsize)
plt.xlabel('Different number of GPUs', fontsize=label_fontsize)
plt.ylabel('Cut size over edge count', fontsize=label_fontsize)
plt.xticks(rotation=0, fontsize=tick_fontsize)
ax = plt.gca()
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, pad=xpad)
plt.gca().xaxis.labelpad = labelpad
# plt.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
plt.annotate(subplot_labels[0], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.subplot(2, 3, 2)
for feed_value in sorted(data.keys()):
    x_labels = categories
    y_values = [data[feed_value]['performance'].get(method, None) for method in categories]
    new_label = feed_value_mapping.get(feed_value, feed_value)
    handle, = plt.plot(x_labels, y_values, marker='o', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.7)

yticks = np.arange(0.63, 0.68, 0.01)
plt.yticks(yticks, fontsize=tick_fontsize)
plt.xlabel('Different graph partition methods', fontsize=label_fontsize)
plt.ylabel('Cut size over edge count', fontsize=label_fontsize)
plt.xticks(rotation=0, fontsize=tick_fontsize)
ax = plt.gca()
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, pad=xpad)
plt.gca().xaxis.labelpad = labelpad
# plt.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
plt.annotate(subplot_labels[2], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.subplot(2, 3, 3)
for feed_value in sorted(data.keys()):
    x_labels = algorithms
    y_values = [data[feed_value]['performance'].get(algo, None) for algo in algorithms]
    new_label = feed_value_mapping.get(feed_value, feed_value)
    handle, = plt.plot(x_labels, y_values, marker='o', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.7)

yticks = np.arange(0.35, 0.95, 0.1)
plt.yticks(yticks, fontsize=tick_fontsize)
plt.xticks(algorithms, rotation=0, fontsize=tick_fontsize)
ax = plt.gca()
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, pad=xpad)
plt.xlabel('Different fine-tuning algorithms', fontsize=label_fontsize)
plt.ylabel('Cut size over edge count', fontsize=label_fontsize)
plt.gca().xaxis.labelpad = labelpad
# plt.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
plt.annotate(subplot_labels[4], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.subplot(2, 3, 4)
for feed_value in sorted(data.keys()):
    x_labels = GPU_count
    y_values = [data[feed_value]['time'].get(method, None) for method in GPU_count]
    new_label = feed_value_mapping.get(feed_value, feed_value)
    handle, = plt.plot(x_labels, y_values, marker='o', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.7)

yticks = np.arange(0.50, 12, 2.25)
plt.yticks(yticks, fontsize=tick_fontsize)
plt.xlabel('Different number of GPUs', fontsize=label_fontsize)
plt.ylabel('Run time (s)', fontsize=label_fontsize)
plt.xticks(rotation=0, fontsize=tick_fontsize)
ax = plt.gca()
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, pad=xpad)
plt.gca().xaxis.labelpad = labelpad
# plt.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
plt.annotate(subplot_labels[1], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.subplot(2, 3, 5)
for feed_value in sorted(data.keys()):
    x_labels = categories
    y_values = [data[feed_value]['time'].get(method, None) for method in categories]
    new_label = feed_value_mapping.get(feed_value, feed_value)
    handle, = plt.plot(x_labels, y_values, marker='o', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.7)

yticks = np.arange(3.25, 4.65, 0.25)
plt.yticks(yticks, fontsize=tick_fontsize)
plt.xlabel('Different graph partition methods', fontsize=label_fontsize)
plt.ylabel('Run time (s)', fontsize=label_fontsize)
plt.xticks(rotation=0, fontsize=tick_fontsize)
ax = plt.gca()
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, pad=xpad)
plt.gca().xaxis.labelpad = labelpad
# plt.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
plt.annotate(subplot_labels[3], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.subplot(2, 3, 6)
for feed_value in sorted(data.keys()):
    x_labels = algorithms
    y_values = [data[feed_value]['time'].get(algo, None) for algo in algorithms]
    new_label = feed_value_mapping.get(feed_value, feed_value)
    handle, = plt.plot(x_labels, y_values, marker='o', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.7)

yticks = np.arange(3.25, 4.35, 0.20)
plt.yticks(yticks, fontsize=tick_fontsize)
plt.xlabel('Different fine-tuning algorithms', fontsize=label_fontsize)
plt.ylabel('Run time (s)', fontsize=label_fontsize)
plt.xticks(rotation=0, fontsize=tick_fontsize)
ax = plt.gca()
ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, pad=xpad)
plt.gca().xaxis.labelpad = labelpad
# plt.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
plt.annotate(subplot_labels[5], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)
# plt.figlegend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=legend_fontsize, bbox_to_anchor=(0.55, 1))
plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
plt.savefig('second_stage.svg')
plt.savefig('second_stage.png')
plt.show()