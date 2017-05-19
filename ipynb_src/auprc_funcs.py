from operator import itemgetter
from matplotlib import pyplot as plt

def parse(s):
    pairs = map(lambda t: t.split(','),
                s[1:-1].split("), ("))
    return [(float(a), float(b)) for a, b in pairs]

def extractTestInfo(filename):
    ret = []
    with open(filename) as f:
        curNode = 0
        for line in f:
            line = line.strip()
            if line.startswith("Node"):
                curNode = int(line.split()[1])
            elif line.startswith("Testing (ref) auPRC"):
                auPRC = float(line.split('=')[1])
            elif line.startswith("Testing (ref) PR") and curNode % 1000 < 5:
                begin = line.find(" = List(") + 8
                pr = parse(line.strip()[begin:-1])
                ret.append((curNode, auPRC, pr))
    return ret

def plot(data):
    plt.figure(dpi=250)
    for node, au, pr in data:
        x = list(map(itemgetter(0), pr))
        y = list(map(itemgetter(1), pr))
        if y[0] == 1.0:
            y[0] = y[1]
        # plt.xlim(0.0, 0.2)
        # plt.ylim(0.95, 1.0)
        if node == 0:
            label = "Iteration: " + str(node) + " (random guess): auPRC=%.4f" % (au)
        else:
            label = "Iteration: " + str(node) + ": auPRC=%.4f" % (au)
        plt.plot(x, y, label=label)
        plt.grid()
        plt.xlabel("recall")
        plt.ylabel("preceision")
        plt.legend(loc="upper right")
        plt.title("Precision-Recall curve after every 1000 iterations")
