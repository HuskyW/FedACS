#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from numpy import sign,ravel
import numpy as np


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvgWithCmfl(w,w_old,threshold=0.8,mute=True,checking=None):
    w_delta = DeltaWeights(w,w_old)
    w_davg = FedAvg(w_delta)
    agreeThres = 0
    for k in w_davg.keys():
        templist = w_davg[k].cpu().numpy().reshape(-1)
        agreeThres += len(templist)
    agreeThres *= threshold
    w_agree = []

    maxagree = 0
    maxindex = 0

    checklist = [False] * len(w_delta)
    
    for i in range(len(w_delta)):
        agreeCount = 0
        for k in w_davg.keys():
            templist1 = w_davg[k].cpu().numpy().reshape(-1)
            templist2 = w_delta[i][k].cpu().numpy().reshape(-1)
            for j in range(len(templist1)):
                if sign(templist1[j]) == sign(templist2[j]):
                    agreeCount += 1
        if agreeCount >= agreeThres:
            w_agree.append(w[i])
            checklist[i] = True
        if maxagree < agreeCount:
            maxagree = agreeCount
            maxindex = i
    if len(w_agree) > 0:
        w_avg = FedAvg(w_agree)
    else:
        w_avg = w[maxindex]

    if checking is not None:
        clientId = checking[0].copy()
        iidThreshold = checking[1]
        rec = [0] * 4
        for i in range(len(clientId)):
            check = 0
            if clientId[i] >= iidThreshold: # nIID
                check += 2
            if checklist[i] is False: #killed
                check += 1
            rec[check] += 1
        print("Cutting : IID (%2d/%2d), non (%2d/%2d)" %tuple(rec))
        

    if mute == False and checking is None:
        print("CMFL: %d out of %d is accepted" % (len(w_agree),len(w)))
    return w_avg


def FedAvgWithL2(w,w_old,avg_l2,boosting=False,cutting=False,mute=True,checking=None):

    boostingThres = avg_l2 * 1
    cuttingThres = avg_l2 * 1.2
    boostcountThres = 0.8

    if w_old is None: # 1st iter
        w_delta = w
    else:
        w_delta = DeltaWeights(w,w_old)

    w_davg = FedAvg(w_delta)
    l2norms = [0] * len(w)
    for k in w_davg.keys():
        avlist = w_davg[k].cpu().numpy().reshape(-1)
        for i in range(len(w)):
            ilist = w_delta[i][k].cpu().numpy().reshape(-1)
            diff = avlist - ilist
            l2norms[i] += np.linalg.norm(diff,ord=2)

    new_avg_l2 = np.mean(l2norms)

    if w_old is None:
        return w_davg, new_avg_l2

    w_dapproved = []
    if cutting:
        for i in range(len(w)):
            if l2norms[i] < cuttingThres:
                w_dapproved.append(w_delta[i])
        if mute == False:
            print("L2: %d out of %d accepted" % (len(w_dapproved),len(w)))

        if len(w_dapproved) == 0:
            minl2index = l2norms.index(min(l2norms))
            w_dapproved.append(w_delta[minl2index])

    else:
        w_dapproved = w_delta
        
    w_davg = FedAvg(w_dapproved)
    
    if boosting:
        boostCount = 0
        for i in range(len(w)):
            if l2norms[i] < boostingThres:
                boostCount += 1
        
        if boostCount >= int(len(w) * boostcountThres):
            for k in w_davg.keys():
                torch.mul(w_davg[k],3)
            if mute == False:
                print("L2: Boosting activated (%d/%d)" % (boostCount, int(len(w) * boostcountThres)))
        elif mute == False:
            print("L2: Boosting failed (%d/%d)" % (boostCount, int(len(w) * boostcountThres)))
    
    for k in w_davg.keys():
        w_davg[k] += w_old[k]

    if checking is not None:
        clientId = checking[0].copy()
        iidThreshold = checking[1]

        if cutting is True:
            rec = [0] * 4
            for i in range(len(clientId)):
                check = 0
                if clientId[i] >= iidThreshold: # nIID
                    check += 2
                if l2norms[i] > cuttingThres: #killed
                    check += 1
                rec[check] += 1
            print("Cutting : IID (%2d/%2d), non (%2d/%2d)" %tuple(rec))
        if boosting is True:
            rec = [0] * 4
            for i in range(len(clientId)):
                check = 0
                if clientId[i] >= iidThreshold: # nIID
                    check += 2
                if l2norms[i] < boostingThres: #killed
                    check += 1
                rec[check] += 1
            print("Boosting: IID (%2d/%2d), non (%2d/%2d)" %tuple(rec))

    return w_davg, new_avg_l2

    
def DeltaWeights(w,w_old):
    deltas = copy.deepcopy(w)
    for k in w[0].keys():
        for i in range(len(w)):
            deltas[i][k] -= w_old[k]
    return deltas