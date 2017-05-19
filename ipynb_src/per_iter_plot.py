# from per_iter_funcs import *
from matplotlib import pyplot as plt

L = 4000
realLabel = {
    "Testing auPRC": "Validation auPRC",
    "Testing (ref) auPRC": "Testing auPRC",
    "Training average score =": "Training average score",
    "Testing average score =": "Validation average score",
    "Testing (ref) average score =": "Testing average score"
}


def plot(zipLabels, nums, vlines=None, xlim=None, ylim=None, title=None, xlabel=None, ylabel=None):
    plt.figure(dpi=250)
    for idx, label in zipLabels:
        label = realLabel.get(label, label)
        plt.plot(range(1, len(nums[idx]) + 1)[:L], nums[idx][:L], label=label)
    plt.grid()
    if ylabel == "average score":
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")
    if vlines:
        for a, b in vlines:
            plt.axvline(a, c='blue', ls="dashed")
            plt.axvline(b, c='red')
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)


def plotSample(labels, joint, vlines, sampleRatio, zoom):
    if zoom:
        plot(zip(range(3), labels[:3]), joint, vlines=vlines,
             xlim=[90, 105], ylim=[0.0, 1.0])
        plot(zip(range(-3, 0), labels[-3:]), joint, vlines=vlines,
             xlim=[4000, 5500], ylim=[0.0, 1.0])
        plot(zip(range(3, len(labels), 3), labels[3:-3:3]), joint, vlines=vlines,
             xlim=[3349, 4300], ylim=[0.7, 0.85])
        # plot(zip(range(4, len(labels), 3), labels[4:-3:3]), joint, vlines=vlines,
        #      xlim=[0, 5000], ylim=[0.0, 1.0])
        # plot(zip(range(5, len(labels), 3), labels[5:-3:3]), joint, vlines=vlines,
        #      xlim=[0, 5000], ylim=[0.0, 1.0])
    else:
        plot(zip(range(3), labels[:3]), joint, vlines=vlines,
             title="auPRC measured every 100 iterations", xlabel="Iteration", ylabel="auPRC")  # auPRC
        plot(zip(range(-3, -2), labels[-3:-2]), joint, vlines=vlines,
             title="Effective number of examples measured every 100 iterations",
             xlabel="Iteration", ylabel="effective number of examples")  # auPRC
        plot(zip(range(3, len(labels), 3), labels[3:-3:3]), joint, vlines=vlines,
             title="Average score measured every 100 iterations",
             xlabel="Iteration", ylabel="average score")  # scores
        # plot(zip(range(4, len(labels), 3), labels[4:-3:3]), joint, vlines=vlines)  # scores (pos)
        # plot(zip(range(5, len(labels), 3), labels[5:-3:3]), joint, vlines=vlines)  # scores (neg)
