# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import matplotlib.pyplot as plt

def plot_graph(num_epoch, data, x, y, title, path):
    plt.plot([i for i in range(0, num_epoch)], data)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')

def plot_heatmap(params, epochs, data, x, y, title, path):
    f, ax = plt.subplots(1, 1)
    cax = ax.matshow(data, interpolation='nearest')
    f.colorbar(cax).set_label('Cumulative Reward')
    ax.set_xticklabels(['']+list(params))
    ax.set_yticklabels(['']+list(epochs))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')