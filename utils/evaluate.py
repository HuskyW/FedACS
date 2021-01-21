import copy
import torch
from torch import nn
from numpy import sign,ravel
import numpy as np
import math

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


def l2NormEvaluate(w_old,w,replace_avg=None):
    if w_old is None: # 1st iter
        w_delta = w
    else:
        w_delta = Fed.DeltaWeights(w,w_old)

    if replace_avg is None:
        w_davg = copy.deepcopy(w_delta[0])
        for k in w_davg.keys():
            for i in range(1, len(w_delta)):
                w_davg[k] += w_delta[i][k]
            w_davg[k] = w_davg[k] / len(w_delta)
    else:
        ravglist = [replace_avg]
        w_davgl = Fed.DeltaWeights(ravglist,w_old)
        w_davg = w_davgl[0]

    l2norms = [0] * len(w)
    for k in w_davg.keys():
        avlist = w_davg[k]
        for i in range(len(w)):
            ilist = w_delta[i][k]
            diff = avlist - ilist
            l2norms[i] += np.linalg.norm(diff,ord=2)

    return l2norms


def FA_round(args,iter):
    if args.faf < 0:
        return False

    if args.faf == 0:
        return True
    
    if args.faf == 1:
        if iter < 200:
            return True
    return False