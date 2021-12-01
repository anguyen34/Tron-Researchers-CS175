# Trent Buckholz
# Anthony Nguyen
# Brain Thai

import matplotlib.pyplot as plt

DIR_PATH = "docs/images/"

def plot_graph(num_epoch, data, x, y, title, path):
    f, ax = plt.subplots(1, 1)
    ax.plot([i for i in range(0, num_epoch)], data)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.savefig(DIR_PATH + path, bbox_inches='tight')

def plot_scatter(data, x, y, title, path):
    f, ax = plt.subplots(1, 1)
    for i, p in data:
        ax.scatter(i, p)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.savefig(DIR_PATH + path, bbox_inches='tight')

def plot_heatmap(params, epochs, data, x, y, title, path):
    data = transpose(data, [])
    f, ax = plt.subplots(1, 1)
    cax = ax.matshow(data, interpolation='nearest')
    f.colorbar(cax).set_label('Cumulative Reward')
    ax.set_xticks(list(range(len(params))))
    ax.set_xticklabels([str(i) for i in params])
    ax.set_yticks(list(range(len(epochs))))
    ax.set_yticklabels([str(i) for i in epochs])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.savefig(DIR_PATH + path, bbox_inches='tight')

def transpose(l1, l2): 
    # iterate over list l1 to the length of an item
    for i in range(len(l1[0])):
        # print(i)
        row =[]
        for item in l1:
            # appending to new list with values and index positions
            # i contains index position and item contains values
            row.append(item[i])
        l2.append(row)
    return l2
