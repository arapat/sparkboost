import pickle
import numpy as np
from matplotlib import pyplot as plt

data = pickle.load(open("data.pkl", 'rb'))
trees = pickle.load(open("trees.pkl", 'rb'))

# 1. The number of examples
numExamples = [t[-1][1] for t in data]

# 2. ratio between the effective number of examples
#    and the actual number of examples on which the rule is non-zero
effectiveNum = []
for d in data:
    (_, _, _, wi, wi2, _) = d[-1]
    effectiveNum.append(wi * wi / wi2)
ratios = [a / b for a, b in zip(effectiveNum, numExamples)]

# 3. The empirical correlation at the end for all x
emprCorrAll = []
for d, tree in zip(data, trees):
    (_, _, s, _, _, sumw_all) = d[-1]
    hx = tree[0]
    emprCorrAll.append(hx * s / sumw_all)

# 4. The empirical correlation wrt the examples for which h_j(x) != 0
emprCorrNonzero = []
for d, tree in zip(data, trees):
    (_, _, s, sumw_nonzero, _, _) = d[-1]
    hx = tree[0]
    emprCorrNonzero.append(hx * s / sumw_nonzero)

# 5. The probability of the the set of examples for which h_j(x) != 0
pNonzero = []
for d in data:
    (_, _, _, sumw_nonzero, _, sumw_all) = d[-1]
    pNonzero.append(sumw_nonzero / sumw_all)


def plot(y, x=None, color=None):
    if x == None:
        x = list(range(1, len(y) + 1))
    
    if color:
        plt.plot(x, y, c=color)
    else:
        plt.plot(x, y)

    depth = [407, 878, 1441, 1760, 2363, 3007, 3186, 3378, 3478,
             3578, 3678, 3778, 3878, 3978, 4078]

    last = 0
    curSum = 0
    for d in depth[:-8]:
        if d != 1343:
            plt.axvline(d-1, c="black")

    plt.axvline(1342, c='r', ls="--")

