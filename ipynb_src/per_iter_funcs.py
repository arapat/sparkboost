from matplotlib import pyplot as plt
from operator import itemgetter

def extractLines(filename, prefix):
    rollback = []
    ret = [[] for k in prefix]
    with open(filename) as f:
        lineId = 0
        for line in f:
            line = line.strip()
            if line.startswith("Rollback"):
                a, b = line[33:-6].split(" nodes to ")
                diff = int(a) - int(b)
                rollback.append((lineId - diff, lineId))
            else:
                for idx, s in enumerate(prefix):
                    if line.startswith(s):
                        ret[idx].append(line)
                        lineId = max(lineId, len(ret[idx]))
    return ret, rollback

def extractNums(linesList, pos=-1):
    ret = []
    for ll in linesList:
        ret.append([float(t.split()[pos]) for t in ll])
    return ret

def plot(zipLabels, nums, vlines=None, xlim=None, ylim=None, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(8, 4))
    for idx, label in zipLabels:
        plt.plot(range(1, len(nums[idx]) + 1), nums[idx], label=label)
    plt.grid()
    plt.legend(loc=(1, 0))
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

