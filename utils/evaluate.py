import copy
import torch
from torch import nn
from numpy import sign,ravel
import numpy as np

import models.Fed as Fed


def deltaWeightEvaluate(w,w_old):
    deltas = copy.deepcopy(w)
    for k in w.keys():
        deltas[k] -= w_old[k]
    weights = None
    for k in deltas.keys():
        newlist = deltas[k].numpy().reshape(-1)
        if weights is None:
            weights = newlist
        else:
            weights = np.append(weights,newlist)
    print('Weights size: %d' % len(weights))

    wmax = weights.max()
    wmin = weights.min()
    wavg = weights.mean()
    wstd = weights.std()

    weights = np.sort(weights)
    q1 = weights[int(len(weights)/4)]
    q3 = weights[int(len(weights)*3/4)]

    print("max: %.4f, min: %.4f, avg: %.4f, std: %.4f" % (wmax,wmin,wavg,wstd))
    print("range: %.4f, IQR: %.4f, IQR/range: %.4f, std/range:  %.4f" % (wmax-wmin , q3-q1 , (q3-q1)/(wmax-wmin) , wstd/(wmax-wmin)))


def l2NormEvaluate(w_old,w):
    if w_old is None: # 1st iter
        w_delta = w
    else:
        w_delta = Fed.DeltaWeights(w,w_old)

    w_davg = Fed.FedAvg(w_delta)

    l2norms = [0] * len(w)
    for k in w_davg.keys():
        avlist = w_davg[k].cpu().numpy().reshape(-1)
        for i in range(len(w)):
            ilist = w_delta[i][k].cpu().numpy().reshape(-1)
            diff = avlist - ilist
            l2norms[i] += np.linalg.norm(diff,ord=2)

    return l2norms

def scaledL2NormEvaluate(w_old,w,scale):
    if w_old is None: # 1st iter
        w_delta = w
    else:
        w_delta = Fed.DeltaWeights(w,w_old)

    w_davg = Fed.FedAvg(w_delta)

    l2norms = [0] * len(w)
    for k in w_davg.keys():
        avlist = w_davg[k].cpu().numpy().reshape(-1)
        for i in range(len(w)):
            ilist = w_delta[i][k].cpu().numpy().reshape(-1)
            diff = avlist - ilist
            diff = np.true_divide(diff,scale[k])
            l2norms[i] += np.linalg.norm(diff,ord=2)

    return l2norms