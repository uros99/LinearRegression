import matplotlib.pyplot as plt
import numpy as np


def draw_dependency(x, y,  label1='label1', label2='price'):
    plt.figure('House prices')
    plt.scatter(x, y, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2, label='Houses')
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend()
    plt.show()
