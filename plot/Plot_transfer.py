import matplotlib.pyplot as plt
import json
import os
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 2

linewidth = 4
markersize = 16
markeredgewidth = 4
s = 200

label_fontsize = 54
tick_fontsize = 44
legend_fontsize = 48
annotate_fontsize = 60

def model_ex(x, a, b):
    return b * np.exp(a * x)

def model_l(x, a, b):
    return b * (x) + a

subplot_labels = ['a', 'b', 'c', 'd']

plt.figure(figsize=(30, 28))

def process_hypermaxind_data():
    with open('../log/transfer/Hypermaxind_vanilla.log') as f:
        Log = f.readlines()
    log_d = {}
    for lines in Log:
        temp = lines.split(',')
        temp1 = temp[1].split(':')
        temp2 = temp[2].split(':')
        temp3 = temp[3].split(':')
        name = temp[0].split(':')[2]
        time = float(temp1[1][1:-1])
        res = float(temp2[1][3:-1])
        res_th = -1 * float(temp3[1][2:-1])
        log_d[name] = {}
        log_d[name]['time'] = time
        log_d[name]['res'] = abs(int(res))
        log_d[name]['res_th'] = abs(int(res_th))
        n = int(name[6:-6])
        d = 3
        log_d[name]['n'] = n
        log_d[name]['d'] = d

    with open('../log/transfer/Hypermaxind_transfer.log') as f:
        Log = f.readlines()
    log_r = {}
    for lines in Log:
        temp = lines.split(',')
        temp1 = temp[1].split(':')
        temp2 = temp[2].split(':')
        temp3 = temp[3].split(':')
        name = temp[0].split(':')[2]
        time = float(temp1[1][1:-1])
        res = float(temp2[1][3:-1])
        res_th = -1 * float(temp3[1][2:-1])
        log_r[name] = {}
        log_r[name]['time'] = time
        log_r[name]['res'] = abs(int(res))
        log_r[name]['res_th'] = abs(int(res_th))

    x_axis = np.array([log_d[name]['n'] for name in log_d])
    y_axis = np.array([log_d[name]['res'] / log_d[name]['n'] for name in log_d])
    y_axis2 = np.array([log_r[name]['res'] / log_d[name]['n'] for name in log_r])
    y_axis_t = np.array([log_d[name]['time'] for name in log_d])
    y_axis_t2 = np.array([log_r[name]['time'] for name in log_r])
    nd = len(y_axis)

    plotm = np.zeros([10, nd])
    plotm[0, :] = x_axis
    plotm[1, :] = y_axis
    plotm[2, :] = y_axis2
    plotm[3, :] = y_axis_t
    plotm[4, :] = y_axis_t2
    plotms = plotm[:, plotm[0].argsort()]


    plt.subplot(2, 2, 1)
    plt.plot(plotms[0, :], plotms[1, :], marker='x', label='Vanilla Training', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
    plt.plot(plotms[0, :], plotms[2, :], marker='o', label='Transfer Learning', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
    yticks = np.arange(0.36, 0.44, 0.02)
    plt.yticks(yticks, fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel('MIS size over node count', fontsize=label_fontsize)
    plt.xlabel('Number of nodes', fontsize=label_fontsize+2)
    plt.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
    plt.annotate(subplot_labels[0], xy=(-0.12, 1.15), xycoords='axes fraction',
             xytext=(-10, 5), textcoords='offset points',
             ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

    popt, pcov = curve_fit(model_l, plotms[0, :], plotms[3, :], p0=[0, 0])
    popt_r, pcov_r = curve_fit(model_l, plotms[0, :], plotms[4, :], p0=[0, 0])
    a, b = popt
    a_r, b_r = popt_r
    x_model = np.linspace(min(plotms[0, :]), max(plotms[0, :]), 100)
    y_model = model_l(x_model, a, b)
    y_model2 = model_l(x_model, a_r, b_r)

    plt.subplot(2, 2, 2)
    plt.scatter(plotms[0, :], plotms[3, :], marker='x', label='Vanilla Training', s=s, linewidth=linewidth)
    plt.scatter(plotms[0, :], plotms[4, :], marker='o', label='Transfer Learning', s=s, linewidth=linewidth)
    plt.plot(x_model, y_model, linestyle='-', linewidth=linewidth)
    plt.plot(x_model, y_model2, linestyle='-', linewidth=linewidth)
    yticks = np.arange(0, 500, 100)
    plt.yticks(yticks, fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel('Run time (s)', fontsize=label_fontsize)
    plt.xlabel('Number of nodes', fontsize=label_fontsize+2)
    plt.legend(loc='upper left', frameon=False, fontsize=legend_fontsize, handletextpad=0.1)
    plt.annotate(subplot_labels[1], xy=(-0.12, 1.15), xycoords='axes fraction',
                 xytext=(-10, 5), textcoords='offset points',
                 ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

def process_hypermincut_data():
    with open('../log/transfer/Hypermincut_vanilla.log') as f:
        Log = f.readlines()
    log_d = {}
    for lines in Log:
        temp = lines.split(',')
        time = float(temp[1][14:])
        temp2 = temp[3][1:].split(':')
        res = int(temp2[0][0:-1])
        temp3 = temp[0].split(':')
        name = temp3[2]
        log_d[name] = {}
        log_d[name]['time'] = time
        log_d[name]['res'] = abs(int(res))
        n = int(name.split('_')[1])
        m = int(name.split('_')[2][:-4])
        log_d[name]['n'] = n
        log_d[name]['m'] = m

    with open('../log/transfer/Hypermincut_transfer.log') as f:
        Log = f.readlines()
    log_r = {}
    for lines in Log:
        temp4 = lines.split(',')
        time2 = float(temp4[1][14:])
        res2 = temp4[3][1:-1]
        temp6 = temp4[0].split(':')
        name2 = temp6[2]
        log_r[name2] = {}
        log_r[name2]['time'] = time2
        log_r[name2]['res'] = abs(int(res2))
        n = int(name2.split('_')[1])
        m = int(name2.split('_')[2][:-4])
        log_r[name2]['n'] = n
        log_r[name2]['m'] = m

    x_axis = np.array([log_d[name]['n'] for name in log_d])
    y_axis = np.array([log_d[name]['res'] / log_d[name]['m'] for name in log_d])
    y_axis2 = np.array([log_r[name]['res'] / log_d[name]['m'] for name in log_d])
    y_axis_t = np.array([log_d[name]['time'] for name in log_d])
    y_axis_t2 = np.array([log_r[name]['time'] for name in log_d])
    nd = len(y_axis)

    plotm = np.zeros([10, nd])
    plotm[0, :] = x_axis
    plotm[1, :] = y_axis
    plotm[2, :] = y_axis2
    plotm[3, :] = y_axis_t
    plotm[4, :] = y_axis_t2
    plotms = plotm[:, plotm[0].argsort()]

    plt.subplot(2, 2, 3)
    plt.plot(plotms[0, :], plotms[1, :], marker='x', label='Vanilla Training', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
    plt.plot(plotms[0, :], plotms[2, :], marker='o', label='Transfer Learning', linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
    yticks = np.arange(0.50, 0.85, 0.05)
    plt.yticks(yticks, fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel('Cut size over hyperedge count', fontsize=label_fontsize)
    plt.xlabel('Number of nodes', fontsize=label_fontsize+2)
    plt.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
    plt.annotate(subplot_labels[2], xy=(-0.12, 1.15), xycoords='axes fraction',
                 xytext=(-10, 5), textcoords='offset points',
                 ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

    popt, pcov = curve_fit(model_l, plotms[0, :], plotms[3, :], p0=[0, 0])
    popt_r, pcov_r = curve_fit(model_l, plotms[0, :], plotms[4, :], p0=[0, 0])
    a, b = popt
    a_r, b_r = popt_r
    x_model = np.linspace(min(plotms[0, :]), max(plotms[0, :]), 100)
    y_model = model_l(x_model, a, b)
    y_model2 = model_l(x_model, a_r, b_r)

    plt.subplot(2, 2, 4)
    plt.scatter(plotms[0, :], plotms[3, :], marker='x', label='Vanilla Training', s=s, linewidth=linewidth)
    plt.scatter(plotms[0, :], plotms[4, :], marker='o', label='Transfer Learning', s=s, linewidth=linewidth)
    plt.plot(x_model, y_model, linestyle='-', linewidth=linewidth)
    plt.plot(x_model, y_model2, linestyle='-', linewidth=linewidth)
    yticks = np.arange(0, 250, 50)
    plt.yticks(yticks, fontsize=tick_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.ylabel('Run time (s)', fontsize=label_fontsize)
    plt.xlabel('Number of nodes', fontsize=label_fontsize+2)
    plt.legend(loc='upper left', frameon=False, fontsize=legend_fontsize, handletextpad=0.1)
    plt.annotate(subplot_labels[3], xy=(-0.12, 1.15), xycoords='axes fraction',
                 xytext=(-10, 5), textcoords='offset points',
                 ha='left', va='bottom', fontsize=annotate_fontsize, weight='bold', clip_on=False)

process_hypermaxind_data()
process_hypermincut_data()

plt.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
plt.savefig('../figure/transfer.svg')
plt.show()