import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 2

s = 300
linewidth = 6
markersize = 16
markeredgewidth = 6

label_fontsize = 78
tick_fontsize = 70
legend_fontsize = 64
annotate_fontsize = 100

plt.figure(figsize=(50, 32))

def read_log_file(path, log_dict):
    with open(path) as f:
        Log = f.readlines()

    file_name = os.path.basename(path)
    match = re.search(r'new(\d)', file_name)
    index = match.group(1)

    for line in Log:
        try:
            temp = line.split(',')
            name = temp[0].split(':')[2]
            time = float(temp[1].split(':')[1][1:-1])
            res = float(temp[2].split(':')[1][3:-2])
            res_th = float(temp[3].split(':')[1][3:-2])

            path_data = f"../data/hypergraph_data/synthetic/sys_random/new{index}/" + name
            with open(path_data) as f:
                file = f.read()
            lines = file.split('\n')
            info = lines[0].split(' ')
            n = int(info[0])
            m = int(info[1])
            log_dict[name] = {
                'time': time,
                'res': abs(int(res)),
                'res_th': abs(int(res_th)),
                'n': n,
                'm': m
            }
        except Exception as e:
            print(f"Error processing line in {path}: {e}")

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
            res = float(temp2[1][9:-2])
            if name not in log_data:
                log_data[name] = {'res': [], 'time': []}
            log_data[name]['res'].append(abs(int(res)))

            time = float(temp1[1][1:])
            log_data[name]['time'].append(time)

            n = int(name[6:-6])
            log_data[name]['n'] = n
            log_data[name]['d'] = d_value
    return log_data

def group_logs_by_n(logs, n_threshold=100):
    sorted_names = sorted(logs.keys(), key=lambda x: logs[x]['n'])
    groups = []
    current_group = []

    for name in sorted_names:
        if not current_group:
            current_group.append(name)
        else:
            last_name = current_group[-1]
            if abs(logs[name]['n'] - logs[last_name]['n']) <= n_threshold:
                current_group.append(name)
            else:
                groups.append(current_group)
                current_group = [name]
    if current_group:
        groups.append(current_group)

    return groups

def aggregate_metrics(grouped_logs, logs, metric_name='res', time_metric='time'):
    aggregated_data = []
    for group in grouped_logs:
        sum_res = 0
        sum_time = 0
        count = len(group)
        res_values = []
        for name in group:
            normalized_res = logs[name][metric_name] / logs[name]['m']
            sum_res += normalized_res
            sum_time += logs[name][time_metric]
            res_values.append(normalized_res)
        avg_res = sum_res / count if count > 0 else 0
        avg_time = sum_time / count if count > 0 else 0
        std_dev = np.std(res_values) if res_values else 0
        aggregated_data.append({
            'n': logs[group[0]]['n'],
            'avg_res': avg_res,
            'avg_time': avg_time,
            'std_dev': std_dev
        })
    return aggregated_data

def model_l(x, a,b):
  return b*(x)+a

def model_ex(x, a, b):
    return b * np.exp(a * x)

def model_quad(x, a, b):
    y = [b*xi**2+a for xi in x]
    return y

all_logs = {
    'hypop': {},
    'adam': {},
    'sa': {}
}

log_files_read = 0
for i in range(10):
    for method, log_dict in zip(['hypop', 'adam','sa'], [all_logs['hypop'], all_logs['adam'], all_logs['sa']]):
        file_path = os.path.join('log', 'hypermaxcut', f'Hypermaxcut_syn_new{i}_{method}.log')
        if os.path.exists(file_path):
            read_log_file(file_path, log_dict)
            log_files_read += 1

log_bi = {}
bipartite_log_path = './log/bipartite/Hypermaxcut_bipartite.log'
if os.path.exists(bipartite_log_path):
    with open(bipartite_log_path) as f:
        Log = f.readlines()
    for line in Log:
        temp = line.split(',')
        name = temp[0][10:-5] + '.txt'
        reslist = [abs(int(temp[2][8:-1]))]
        num = temp[0][10:-5].split('_')
        time_tlist = [float(re.search(r'\d+\.\d+', temp[4]).group())]
        time_mlist = [float(re.search(r'\d+\.\d+', temp[5]).group())]
        time_total = [time_mlist[i] + time_tlist[i] for i in range(len(time_tlist))]
        log_bi[name] = {
            'time_train': time_tlist,
            'time_map': time_mlist,
            'time_total': time_total,
            'res': reslist,
            'res_th': [abs(int(temp[3][11:-1]))],
            'n': int(num[1]),
            'm': int(num[2]),
        }


methods = ['HypOp', 'Adam', 'SA']
method_log_keys = ['hypop', 'adam','sa']

aggregated_data = {}
for method, log_key in zip(methods, method_log_keys):
    log_dict = all_logs[log_key]
    grouped_logs = group_logs_by_n(log_dict)
    aggregated_data[method] = aggregate_metrics(grouped_logs, log_dict)

aggregated_data['Bipartite'] = [
    {
        'n': log_bi[k]['n'],
        'avg_res': np.mean(log_bi[k]['res']) / log_bi[k]['m'],
        'avg_time': np.mean(log_bi[k]['time_total'])
    } for k in log_bi]

all_x_axis = {}
all_y_axis_res = {}
all_y_axis_time = {}
all_std_dev = {}

for method in methods + ['Bipartite']:
    data = aggregated_data[method]
    all_x_axis[method] = np.array([entry['n'] for entry in data])
    all_y_axis_res[method] = np.array([entry['avg_res'] for entry in data])
    all_y_axis_time[method] = np.array([entry['avg_time'] for entry in data])
    if method == 'HypOp':
        all_std_dev[method] = np.array([entry['std_dev'] for entry in data])

fit_models = {'Linear': model_l, 'Quadratic': model_quad}
model_colors = {'HypOp': '#3483BB', 'SA': '#3FA83F', 'Adam': '#FF8A24', 'Bipartite': '#9E74C2'}


subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']

plt.subplot(2, 3, 1)
for method in methods + ['Bipartite']:
    plt.plot(all_x_axis[method], all_y_axis_res[method], color=model_colors[method], marker='o', label=method, linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)

plt.fill_between(all_x_axis['HypOp'],
                 all_y_axis_res['HypOp'] - all_std_dev['HypOp'],
                 all_y_axis_res['HypOp'] + all_std_dev['HypOp'],
                 color='#1f77b4', alpha=0.2)

yticks = np.arange(0.945, 0.995, 0.01)
plt.yticks(yticks, fontsize=tick_fontsize)
xticks = np.arange(1000, 11000, 2000)
plt.xticks(xticks, fontsize=tick_fontsize)
plt.ylabel('Cut size over hyperedge count', fontsize=label_fontsize-4)
plt.xlabel('Number of nodes', fontsize=label_fontsize)
plt.legend(loc='upper left', frameon=False, fontsize=legend_fontsize)
plt.annotate(subplot_labels[0], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.subplot(2, 3, 4)
for method in methods + ['Bipartite']:
    plt.scatter(all_x_axis[method], all_y_axis_time[method],color=model_colors[method], label=method, marker='o', s=s)

    model_func = fit_models['Linear'] if method == 'HypOp' else fit_models['Quadratic']
    try:
        popt, _ = curve_fit(model_func, all_x_axis[method], all_y_axis_time[method])
        x_model = np.linspace(min(all_x_axis[method]), max(all_x_axis[method]), 100)
        y_model = model_func(x_model, *popt)
        plt.plot(x_model, y_model, color=model_colors[method], linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
    except Exception as e:
        print(f"Error fitting model to {method}: {e}")

plt.xticks(fontsize=tick_fontsize)
yticks = np.arange(0, 1600, 200)
plt.yticks(yticks, fontsize=tick_fontsize)
xticks = np.arange(1000, 11000, 2000)
plt.xticks(xticks, fontsize=tick_fontsize)
plt.ylabel('Run time (s)', fontsize=label_fontsize-4)
plt.xlabel('Number of nodes', fontsize=label_fontsize)
plt.legend(loc='upper left', frameon=False, fontsize=legend_fontsize, handletextpad=0.1)
plt.annotate(subplot_labels[1], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

hy_d3_files = ['log/maxind/maxind_syn_d3_ep2w_it20.log']
pi_d3_files = ['log/maxind/maxind_pignn_d3_ep4w_it1.log']
hy_d5_files = ['log/maxind/maxind_syn_d5_ep2w_it20.log']
pi_d5_files = ['log/maxind/maxind_pignn_d5_ep4w_it1.log']

log_d3 = process_log(hy_d3_files, 3)
log_d5 = process_log(hy_d5_files, 5)
log_r3 = process_log(pi_d3_files, 3)
log_r5 = process_log(pi_d5_files, 5)

x_axis_d3 = np.array([log_d3[name]['n'] for name in log_d3])
x_axis_d5 = np.array([log_d5[name]['n'] for name in log_d5])

y_axis_d3 = np.array([np.mean(log_d3[name]['res']) / log_d3[name]['n'] for name in log_d3])
y_axis_r3 = np.array([np.mean(log_r3[name]['res']) / log_d3[name]['n'] for name in log_r3])

y_axis_d5 = np.array([np.mean(log_d5[name]['res']) / log_d5[name]['n'] for name in log_d5])
y_axis_r5 = np.array([np.mean(log_r5[name]['res']) / log_d5[name]['n'] for name in log_r5])

y_time_d3 = np.array([np.mean(log_d3[name]['time']) for name in log_d3])
y_time_r3 = np.array([np.mean(log_r3[name]['time']) for name in log_r3])

y_time_d5 = np.array([np.mean(log_d5[name]['time']) for name in log_d5])
y_time_r5 = np.array([np.mean(log_r5[name]['time']) for name in log_r5])

y_axis_d3_std = np.array([np.std(log_d3[name]['res']) / log_d3[name]['n'] for name in log_d3])
y_axis_r3_std = np.array([np.std(log_r3[name]['res']) / log_d3[name]['n'] for name in log_r3])

y_axis_d5_std = np.array([np.std(log_d5[name]['res']) / log_d5[name]['n'] for name in log_d5])
y_axis_r5_std = np.array([np.std(log_r5[name]['res']) / log_d5[name]['n'] for name in log_r5])

nd = len(y_axis_d3)
plotm = np.zeros([14, nd])
plotm[0, :] = x_axis_d3
plotm[1, :] = y_axis_d3
plotm[2, :] = y_axis_r3
plotm[3, :] = y_time_d3
plotm[4, :] = y_time_r3
plotm[5, :] = y_axis_d3 + y_axis_d3_std
plotm[6, :] = y_axis_d3 - y_axis_d3_std
plotm[7, :] = x_axis_d5
plotm[8, :] = y_axis_d5
plotm[9, :] = y_axis_r5
plotm[10, :] = y_time_d5
plotm[11, :] = y_time_r5
plotm[12, :] = y_axis_d5 + y_axis_d5_std
plotm[13, :] = y_axis_d5 - y_axis_d5_std

plotms = plotm[:, plotm[0].argsort()]

plt.subplot(2, 3, 2)
plt.plot(plotms[0, :], plotms[1, :], marker='o', label='HypOp', color='#1f77b4', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.plot(plotms[0, :], plotms[2, :], marker='o', label='PI-GNN', color='#ff7f0e', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.fill_between(plotms[0, :], plotms[5, :], plotms[6, :], color='#1f77b4', alpha=0.2)
plt.ylim(0.40, 0.45)
plt.ylabel('MIS size over node count', fontsize=label_fontsize-4)
plt.xlabel('Number of nodes', fontsize=label_fontsize)
xticks = [500, 1500, 2500, 3500, 4500]
plt.xticks(xticks, fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(loc='upper left', frameon=False, fontsize=legend_fontsize)
plt.annotate(subplot_labels[2], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.subplot(2, 3, 5)
popt, pcov = curve_fit(model_ex, plotms[0, :], plotms[3, :], p0=[0, 0])
a_r3, b_r3 = popt
x_model = np.linspace(min(plotms[0, :]), max(plotms[0, :]), 100)
y_model_d3 = model_ex(x_model, a_r3, b_r3)
popt, pcov = curve_fit(model_ex, plotms[0, :], plotms[4, :], p0=[0, 0])
a_l3, b_l3 = popt
y_model_r3 = model_ex(x_model, a_l3, b_l3)

plt.scatter(plotms[0, :], plotms[3, :], color='#1f77b4', label='HypOp', s=s)
plt.scatter(plotms[0, :], plotms[4, :], color='#ff7f0e', label='PI-GNN', s=s)
xticks = [500, 1500, 2500, 3500, 4500]
plt.xticks(xticks, fontsize=tick_fontsize)
plt.ylim(0, 1000)
yticks = np.arange(0, 1200, 200)
plt.yticks(yticks, fontsize=tick_fontsize)
plt.plot(x_model, y_model_d3, color='#1f77b4', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.plot(x_model, y_model_r3, color='#ff7f0e', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.ylabel('Run time (s)', fontsize=label_fontsize-4)
plt.xlabel('Number of nodes', fontsize=label_fontsize)
plt.legend(loc='upper left', frameon=False, fontsize=legend_fontsize, handletextpad=0.1)
plt.annotate(subplot_labels[3], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.subplot(2, 3, 3)
plt.plot(plotms[7, :], plotms[8, :], marker='o', label='HypOp', color='#1f77b4', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.plot(plotms[7, :], plotms[9, :], marker='o', label='PI-GNN', color='#ff7f0e', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.fill_between(plotms[7, :], plotms[12, :], plotms[13, :], color='#1f77b4', alpha=0.2)
plt.ylim(0.31, 0.37)
plt.ylabel('MIS size over node count', fontsize=label_fontsize-4)
plt.xlabel('Number of nodes', fontsize=label_fontsize)
xticks = [500, 1500, 2500, 3500, 4500]
plt.xticks(xticks, fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(loc='upper left', frameon=False, fontsize=legend_fontsize)
plt.annotate(subplot_labels[4], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.subplot(2, 3, 6)
popt, pcov = curve_fit(model_ex, plotms[7, :], plotms[10, :], p0=[0, 0])
a_r5, b_r5 = popt
x_model = np.linspace(min(plotms[7, :]), max(plotms[7, :]), 100)
y_model_d5 = model_ex(x_model, a_r5, b_r5)
popt, pcov = curve_fit(model_ex, plotms[7, :], plotms[11, :], p0=[0, 0])
a_l5, b_l5 = popt
y_model_r5 = model_ex(x_model, a_l5, b_l5)

plt.scatter(plotms[7, :], plotms[10, :], color='#1f77b4', label='HypOp', s=s)
plt.scatter(plotms[7, :], plotms[11, :], color='#ff7f0e', label='PI-GNN', s=s)
xticks = [500, 1500, 2500, 3500, 4500]
plt.xticks(xticks, fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.plot(x_model, y_model_d5, color='#1f77b4', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.plot(x_model, y_model_r5, color='#ff7f0e', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
plt.ylabel('Run time (s)', fontsize=label_fontsize-4)
plt.xlabel('Number of nodes', fontsize=label_fontsize)
plt.legend(loc='upper left', frameon=False, fontsize=legend_fontsize, handletextpad=0.1)
plt.annotate(subplot_labels[5], xy=(-0.25, 1.05), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
plt.savefig('Hypermaxcut_maxind.png')
plt.savefig('Hypermaxcut_maxind.svg')
# plt.show()
