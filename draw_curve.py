import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
'''
To run this script, pleasure run main.py first to produce the curve_cache.
'''
curve_cache_dir='./score/curve_cache/' #'./Your_Method_Name/score/curve_cache/'
curve_save_dir='./score/'
if not os.path.exists(curve_save_dir):
    os.makedirs(curve_save_dir)
datasets=os.listdir(curve_cache_dir)
for dataset in datasets:
    plot_pr_vals = {}
    plot_fm_vals = {}
    dataset_dir=os.path.join(curve_cache_dir,dataset)
    methods=os.listdir(dataset_dir)
    for method in methods:
        method_dir=os.path.join(dataset_dir,method)
        pr_cache_path=os.path.join(method_dir,'pr.txt')
        fm_cache_path=os.path.join(method_dir,'fm.txt')
        prec=np.loadtxt(pr_cache_path)[:,0]
        recall=np.loadtxt(pr_cache_path)[:,1]
        fm=np.loadtxt(fm_cache_path)
        fm_x=np.array([i for i in range(1,256)])
        plot_pr_vals[method]=(recall,prec)
        plot_fm_vals[method]=(fm_x,fm)
    plt.clf()
    colors = 'rkbmc';
    ticks = ['-', '--']
    for i, m in enumerate(methods):
        x, y = plot_pr_vals[m]
        marker = colors[i % len(colors)] + ticks[i % 2]
        plt.plot(x, y, marker, linewidth=2, label=m)

    plt.grid(True)
    _font_size_ = 16
    plt.title(dataset, fontsize=_font_size_ + 2)
    plt.xlim([0.55, 1.0]);  # plt.ylim([0.0, 1.0])
    plt.xlabel("Recall", fontsize=_font_size_);
    plt.xticks(fontsize=_font_size_ - 4)
    plt.ylabel("Precision", fontsize=_font_size_);
    plt.yticks(fontsize=_font_size_ - 4)
    plt.legend(methods, loc='lower left', fontsize=_font_size_ - 2, framealpha=0.75)
    plt.savefig(os.path.join(curve_save_dir, '{}_pr.png'.format(dataset)), bbox_inches='tight')
    # plt.show()

    plt.clf()
    colors = 'rkbmc';
    ticks = ['-', '--']
    for i, m in enumerate(methods):
        x, y = plot_fm_vals[m]
        marker = colors[i % len(colors)] + ticks[i % 2]
        plt.plot(x, y, marker, linewidth=2, label=m)

    plt.grid(True)
    _font_size_ = 16
    plt.title(dataset, fontsize=_font_size_ + 2)
    plt.xlim([0, 255]);  # plt.ylim([0.0, 1.0])
    plt.xlabel("Threshold", fontsize=_font_size_);
    plt.xticks(fontsize=_font_size_ - 4)
    plt.ylabel("F-measure", fontsize=_font_size_);
    plt.yticks(fontsize=_font_size_ - 4)
    plt.legend(methods, loc='lower left', fontsize=_font_size_ - 2, framealpha=0.75)
    plt.savefig(os.path.join(curve_save_dir, '{}_fm.png'.format(dataset)), bbox_inches='tight')

