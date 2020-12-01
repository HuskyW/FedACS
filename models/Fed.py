#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from numpy import sign,ravel


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvgWithCmfl(w,w_old,threshold=0.8,mute=True):
    w_delta = DeltaWeights(w,w_old)
    w_davg = FedAvg(w_delta)
    agreeThres = 0
    for k in w_davg.keys():
        templist = w_davg[k].numpy().reshape(-1)
        agreeThres += len(templist)
    agreeThres *= threshold
    w_agree = []

    maxagree = 0
    maxindex = 0
    
    for i in range(len(w_delta)):
        agreeCount = 0
        for k in w_davg.keys():
            templist1 = w_davg[k].numpy().reshape(-1)
            templist2 = w_delta[i][k].numpy().reshape(-1)
            for j in range(len(templist1)):
                if sign(templist1[j]) == sign(templist2[j]):
                    agreeCount += 1
        if agreeCount >= agreeThres:
            w_agree.append(w[i])
        if maxagree < agreeCount:
            maxagree = agreeCount
            maxindex = i
    if len(w_agree) > 0:
        w_avg = FedAvg(w_agree)
    else:
        w_avg = w[maxindex]

    if mute == False:
        print("CMFL: %d out of %d is accepted" % (len(w_agree),len(w)))
    return w_avg
    
def DeltaWeights(w,w_old):
    deltas = copy.deepcopy(w)
    for k in w[0].keys():
        for i in range(len(w)):
            deltas[i][k] -= w_old[k]
    return deltas