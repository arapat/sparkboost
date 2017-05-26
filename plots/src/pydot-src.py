# %load src/pydot.py
import pickle
from math import log
import pydot
from posmap import posMap

depth1 = [0, 407, 878, 1441, 1760]
trees = pickle.load(open("trees.pkl", 'rb'))


def addNode(idx, added, graph):
    if idx in added:
        return
    added.add(idx)
    (pred, fa, depth, (dim, res)) = trees[idx]
    if dim < 0:
        label2 = "%.2f" % (0.5 * log(0.3/99.7))
        node2 = pydot.Node()
        node2.set_name("s" + str(idx))
        node2.set_label(label2)

        graph.add_node(node2)
        return

    (pos, base) = posMap[dim]
    if pos > 0:
        posval = '+%d' % pos
    else:
        posval = pos

    bgc = "blue"
    if idx < depth1[1]:
        step = idx
       # print idx, step
    elif depth == 2:
        step = idx - depth1[1]
        #print idx, step
    else:
        step = idx - depth1[2]
        #print idx, step
    if idx > 1342:
        bgc = "blue"
        if depth == 2:
            step = idx - depth1[3]
         #   print idx, step
        else:
            step = idx - depth1[4]
          #  print idx, step

    # if step == 2:
     #    print idx, "222222", depth, base, posval

    label1 = "p" + str(posval) + " is " + base
    # label1 = str(step) + ": position " + str(posval) + " is " + base + str(idx)
    # print step, "==>", label1
    label2 = "%.2f" % pred
    if idx == 325:
        label2 = "-1.09"
    elif pred > 0:
        label2 = '+' + label2

    node1 = pydot.Node(shape="box", color=bgc)
    node1.set_name(str(idx))
    node1.set_label(label1)

    node2 = pydot.Node()
    node2.set_name("s" + str(idx))
    node2.set_label(label2)

    graph.add_node(node1)
    graph.add_node(node2)
    edge = pydot.Edge(node1, node2, dir="forward")
    if res:
        edge.set_label("Y")
    else:
        edge.set_label("N")
    graph.add_edge(edge)

