## Author: Saket Choudhar [saketkc\\gmail]
## License: GPL v3
## Copyright (c) 2017 Saket Choudhary<saketkc__AT__gmail>
## Modification: Julaiti Alafate [jalafate\\gmail]

from operator import itemgetter

import seaborn
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
from matplotlib import transforms
import matplotlib.patheffects
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

import numpy as np

COLOR_SCHEME = {'G': 'orange',
                'A': 'red',
                'C': 'blue',
                'T': 'darkgreen',
                "is": "green",
                "isnot": "red"}

BASES = list(COLOR_SCHEME.keys())


class Scale(matplotlib.patheffects.RendererBase):
    def __init__(self, sx, sy=None):
        self._sx = sx
        self._sy = sy

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        affine = affine.identity().scale(self._sx, self._sy)+affine
        renderer.draw_path(gc, tpath, affine, rgbFace)


def draw_logo(all_scores, xrange, fontfamily='Arial', size=80):
    def xtick(k):
        if k >= 59:
            return '+' + str(k - 58)
        return k - 59
    # mpl.rcParams['font.family'] = fontfamily

    fig, ax = plt.subplots(figsize=(len(all_scores), 5.0))

    font = FontProperties()
    font.set_size(size)
    font.set_weight('bold')

    #font.set_family(fontfamily)

    # ax.set_xticks([k + 1 for k in xrange]) # range(1,len(all_scores)+1))
    ax.set_xticks(range(1,len(all_scores)+1))
    ax.set_yticks(range(0,5))
    # ax.set_xticklabels(range(1,len(all_scores)+1), rotation=90)
    ax.set_xticklabels([xtick(k) for k in xrange])
    ax.set_yticklabels(np.arange(-2,3,1))
    seaborn.despine(ax=ax, trim=True)

    max_scores = 0.0
    for index, scores in enumerate(all_scores):
        # if index in xrange:
            max_scores = max(max_scores, sum(map(lambda t: abs(t[1]), scores)))
    scale_fact = 2.0 / max_scores

    ax.axhline(2, lw=3.0, c="black")
    ax.axvline(59.5-xrange[0], lw=3.0, c="black")

    print("line 64")
    # Positive scores
    trans_offset = transforms.offset_copy(
        ax.transData,
        fig=fig,
        x=1,
        y=0,
        units='points'
    )
    print("line 72")
    for index, scores in enumerate(all_scores):
        # if index not in xrange:
        #     continue
        for base, score, mark in scores:
            if score <= 0:
                continue
            txt = ax.text(
                # index-xrange[0]+1,
                index+1,
                2,
                base,
                transform=trans_offset,
                fontsize=80,
                color=COLOR_SCHEME[base],
                ha='center',
                fontproperties=font
            )
            score *= scale_fact
            txt.set_path_effects([Scale(1.0, score)])
            fig.canvas.draw()
            window_ext = txt.get_window_extent(txt._renderer)
            yshift = window_ext.height*score
            trans_offset = transforms.offset_copy(
                txt._transform,
                fig=fig,
                y=yshift,
                units='points'
            )
        trans_offset = transforms.offset_copy(
            ax.transData,
            fig=fig,
            x=1,
            y=0,
            units='points'
        )

    print("line 109")
    # Positive examples
    trans_offset = transforms.offset_copy(
        ax.transData,
        fig=fig,
        x=1,
        y=0,
        units='points'
    )
    print("line 118")
    for index, scores in enumerate(all_scores):
        # if index not in xrange:
        #     continue
        offset = 0.0
        for base, score, mark in scores:
            if score < 0:
                offset -= score
        offset = 2.0 - offset * scale_fact
        for base, score, mark in scores[-1::-1]:
            if score >= 0:
                continue
            score = -score
            txt = ax.text(
                # index-xrange[0]+1,
                index+1,
                offset,
                base,
                transform=trans_offset,
                fontsize=80,
                color=COLOR_SCHEME[base],
                ha='center',
                fontproperties=font
            )
            score *= scale_fact
            txt.set_path_effects([Scale(1.0, score)])
            fig.canvas.draw()
            window_ext = txt.get_window_extent(txt._renderer)
            yshift = window_ext.height*score
            trans_offset = transforms.offset_copy(
                txt._transform,
                fig=fig,
                y=yshift,
                units='points'
            )
        trans_offset = transforms.offset_copy(
            ax.transData,
            fig=fig,
            x=1,
            y=0,
            units='points'
        )
    print("line 159")

    # plt.xlim(xrange[0], xrange[-1])
    plt.title("Sequence logo of the informative bases around the splice site\n\n", fontsize=40)
    plt.show()


def getScores(filepath, leftBound=None, rightBound=None):
    posScoresMap = {}
    negScoresMap = {}

    with open(filepath) as f:
        for line in f:
            index, depth, score, name = line.strip().split(", ")
            score = float(score)
            name = name.strip().split()
            isornot, name = name[0], name[1]
            pos, base = int(name[:-1]), name[-1]

            scoresMap = posScoresMap
            if score < 0:
                scoresMap = negScoresMap
            if pos not in scoresMap:
                scoresMap[pos] = {}
            if isornot == "is":
                scoresMap[pos][base] = scoresMap[pos].get(base, 0.0) + abs(float(score))
            else:
                for c in ["A", "C", "G", "T"]:
                    if c != base:
                        scoresMap[pos][c] = scoresMap[pos].get(c, 0.0) + abs(float(score)) / 3.0

    for pos in posScoresMap:
        for c in ["A", "C", "G", "T"]:
            if pos in posScoresMap and c in posScoresMap[pos] and \
                    pos in negScoresMap and c in negScoresMap[pos]:
                s = posScoresMap[pos][c] - negScoresMap[pos][c]
                posScoresMap[pos][c] = max(s, 0.0)
                negScoresMap[pos][c] = max(-s, 0.0)

    posScores, negScores = [], []

    for scoresMap, scores in [(posScoresMap, posScores), (negScoresMap, negScores)]:
        rng = range(0, max(scoresMap.keys()) + 1)
        if leftBound is not None and rightBound is not None:
            rng = range(leftBound, rightBound + 1)
        maxRsum = 0.0
        for i in rng:  # range(0, max(scoresMap.keys()) + 1):
            if i not in scoresMap:
                scoresMap[i] = {}
            r = []
            for c in ['A', 'C', 'G', 'T']:
                r.append((c, "is", scoresMap[i].get(c, 0.0)))
            maxRsum = max(maxRsum, sum(map(itemgetter(2), r)))
            scores.append(sorted(r, key=itemgetter(2)))
        for idx in range(len(scores)):
            for j in range(len(scores[idx])):
                ir = scores[idx][j]
                scores[idx][j] = (ir[0], ir[1], ir[2] / maxRsum)
    return posScores, negScores
