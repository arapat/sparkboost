from per_iter_funcs import *

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
             title="Sample ratio " + str(sampleRatio) + "% auPRC", xlabel="Iteration", ylabel="auPRC")  # auPRC
        plot(zip(range(-3, 0), labels[-3:]), joint, vlines=vlines,
             title="Sample ratio " + str(sampleRatio) + "% effective count",
             xlabel="Iteration", ylabel="effective count")  # auPRC
        plot(zip(range(3, len(labels), 3), labels[3:-3:3]), joint, vlines=vlines,
             title="Sample ratio " + str(sampleRatio) + "% average score",
             xlabel="Iteration", ylabel="average score")  # scores
        # plot(zip(range(4, len(labels), 3), labels[4:-3:3]), joint, vlines=vlines)  # scores (pos)
        # plot(zip(range(5, len(labels), 3), labels[5:-3:3]), joint, vlines=vlines)  # scores (neg)

