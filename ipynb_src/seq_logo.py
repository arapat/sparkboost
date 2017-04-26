from operator import itemgetter
import seaborn
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
from matplotlib import transforms
import matplotlib.patheffects
import numpy as np

COLOR_SCHEME = {'G': 'orange', 
                'A': 'red', 
                'C': 'blue', 
                'T': 'darkgreen'}

class Scale(matplotlib.patheffects.RendererBase):
    def __init__(self, sx, sy=None):
        self._sx = sx
        self._sy = sy

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        affine = affine.identity().scale(self._sx, self._sy)+affine
        renderer.draw_path(gc, tpath, affine, rgbFace)


def draw_logo(all_scores):
    fig = plt.figure()
    fig.set_size_inches(len(all_scores), 2.5)
    ax = fig.add_subplot(111)
    ax.set_xticks(range(len(all_scores)))

    xshift = 0
    trans_offset = transforms.offset_copy(
        ax.transAxes, 
        fig=fig, 
        x=0, 
        y=0, 
        units='points'
    )

    for scores in all_scores:
        yshift = 0
        for base, score in scores:
            txt = ax.text(0, 
                          0, 
                          base, 
                          transform=trans_offset,
                          fontsize=80, 
                          color=COLOR_SCHEME[base],
                          weight='bold',
                          ha='center',
                          family='sans-serif')
            txt.set_clip_on(False) 
            txt.set_path_effects([Scale(1.0, score)])
            fig.canvas.draw()
            window_ext = txt.get_window_extent(txt._renderer)
            yshift = window_ext.height * score
            trans_offset = transforms.offset_copy(txt._transform, fig=fig, y=yshift, units='points')
        xshift += window_ext.width
        trans_offset = transforms.offset_copy(ax.transAxes, fig=fig, x=xshift, units='points')


    ax.set_yticks(range(0,3))


    seaborn.despine(ax=ax, offset=30, trim=True)
    ax.set_xticklabels(range(1,len(all_scores)+1), rotation=90)
    ax.set_yticklabels(np.arange(0,3,1))

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
