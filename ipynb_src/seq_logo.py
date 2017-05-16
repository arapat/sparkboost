## Author: Saket Choudhar [saketkc\\gmail]
## License: GPL v3
## Copyright © 2017 Saket Choudhary<saketkc__AT__gmail>
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


def draw_logo(all_scores, fontfamily='Arial', size=80):
    mpl.rcParams['font.family'] = fontfamily

    fig, ax = plt.subplots(figsize=(len(all_scores), 5.0))

    font = FontProperties()
    font.set_size(size)
    font.set_weight('bold')

    #font.set_family(fontfamily)

    ax.set_xticks(range(1,len(all_scores)+1))
    ax.set_yticks(range(0,5))
    ax.set_xticklabels(range(1,len(all_scores)+1), rotation=90)
    ax.set_yticklabels(np.arange(-2,3,1))
    seaborn.despine(ax=ax, trim=True)

    max_scores = 0.0
    for index, scores in enumerate(all_scores):
        max_scores = max(max_scores, sum(map(lambda t: abs(t[1]), scores)))
    scale_fact = 2.0 / max_scores

    ax.axhline(2, lw=10.0, c="gray")
    ax.axvline(59.5, lw=10.0, c="blue")

    # Positive scores
    trans_offset = transforms.offset_copy(
        ax.transData,
        fig=fig,
        x=1,
        y=0,
        units='points'
    )
    for index, scores in enumerate(all_scores):
        for base, score, mark in scores:
            if score <= 0:
                continue
            txt = ax.text(
                index+1,
                2,
                base,
                transform=trans_offset,
                fontsize=80,
                color=COLOR_SCHEME[mark],
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

    # Positive examples
    trans_offset = transforms.offset_copy(
        ax.transData,
        fig=fig,
        x=1,
        y=0,
        units='points'
    )
    for index, scores in enumerate(all_scores):
        offset = 0.0
        for base, score, mark in scores:
            if score < 0:
                offset -= score
        offset = 2.0 - offset * scale_fact
        for base, score, mark in scores:
            if score >= 0:
                continue
            score = -score
            txt = ax.text(
                index+1,
                offset,
                base,
                transform=trans_offset,
                fontsize=80,
                color=COLOR_SCHEME[mark],
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
            scoresMap[pos][base] = scoresMap[pos].get(base, {})
            scoresMap[pos][base][isornot] = scoresMap.get(isornot, 0.0) + abs(float(score))

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
                for isornot in ["is", "isnot"]:
                    r.append((c, isornot, scoresMap[i].get(c, {}).get(isornot, 0.0)))
            maxRsum = max(maxRsum, sum(map(itemgetter(2), r)))
            scores.append(sorted(r, key=itemgetter(2)))
        for idx in range(len(scores)):
            for j in range(len(scores[idx])):
                ir = scores[idx][j]
                scores[idx][j] = (ir[0], ir[1], ir[2] / maxRsum)
    return posScores, negScores
