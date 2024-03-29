#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
import openpyxl
import torchvision
import math

from utils.sampling import mnist_iid, cifar_iid, complex_skewness_mnist, uni_skewness_cifar, pareto_skewness_cifar, dirichlet_skewness_cifar, inversepareto_skewness_cifar, staged_skewness_cifar, fewclass_uni_skewness_cifar
from utils.options import args_parser
from models.Update import LocalUpdate, SingleBgdUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvgWithCmfl
from models.test import test_img

from utils.evaluate import l2NormEvaluate, FA_round
from models.Bound import estimateBounds
from models.Bandit import SelfSparringBandit, Rexp3Bandit, OortBandit
from models.dla_simple import SimpleDLA


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # seed
    np.random.seed(args.seed)

    # restore args
    if args.dataset == 'mnist':
        args.num_channels = 1
        allsamples = 60000
    if args.dataset == 'cifar':
        args.num_channels = 3
        allsamples = 50000
    
    numsamples = int(allsamples/args.num_users)

    if args.num_data > 0:
        numsamples = args.num_data

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.sampling == 'iid':
            dict_users,dominance = mnist_iid(dataset_train, args.num_users, numsamples)
        elif args.sampling == 'uniform':
            dict_users, dominance = complex_skewness_mnist(dataset_train, args.num_users, numsamples)
        else:
            exit("Bad argument: sampling")
    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
        if args.sampling == 'iid':
            dict_users, dominance = cifar_iid(dataset_train, args.num_users, numsamples)
        elif args.sampling == 'nclass':
            print("Dont use this IID")
            exit(0)
            #dict_users, dominance = nclass_skewness_cifar(dataset_train, args.num_users, numsamples)
        elif args.sampling == 'uniform':
            dict_users, dominance = uni_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'ipareto':
            dict_users, dominance = inversepareto_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'dirichlet':
            dict_users, dominance = dirichlet_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'pareto':
            dict_users, dominance = pareto_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'staged':
            dict_users, dominance = staged_skewness_cifar(dataset_train,args.num_users, numsamples)
        elif args.sampling == 'fewclass':
            dict_users, dominance = fewclass_uni_skewness_cifar(dataset_train,args.num_users, numsamples)
        else:
            exit("Bad argument: iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'resnet50' and args.dataset == 'cifar':
        net_glob = torchvision.models.resnet50(pretrained=False)
    elif args.model == 'vgg16' and args.dataset == 'cifar':
        net_glob = torchvision.models.vgg16(pretrained=False)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = torchvision.models.resnet18(pretrained=False)
    elif args.model == 'sdla' and args.dataset == 'cifar':
        net_glob = SimpleDLA().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # client selection scheme
    if args.client_sel == 'random':
        pass
    elif args.client_sel == 'fedacs':
        bandit = SelfSparringBandit(args)
    elif args.client_sel == 'rexp3':
        bandit = Rexp3Bandit(args)
    elif args.client_sel == 'oort':
        bandit = OortBandit(args)
    else:
        print("Bad Argument: client_sel")
        exit(-1)

    # ranking of dominance
    domiranktemp = {}
    domirank = {}
    for i in range(args.num_users):
        domiranktemp[i] = dominance[i]
    if args.sampling != 'dirichlet':
        DRTsort = sorted(domiranktemp.items(),key=lambda x:x[1],reverse=False)
    else:
        DRTsort = sorted(domiranktemp.items(),key=lambda x:x[1],reverse=True)
    for i in range(len(DRTsort)):
        domirank[DRTsort[i][0]] = i

    # copy weights
    w_glob = net_glob.state_dict()
    w_old = copy.deepcopy(w_glob)

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    locallr = args.lr

    if args.testing > 0:
        testacc = []
    
    domilog = []
    rewardlog = []
    hitmaplog = []

    l2eval = np.ones((args.epochs-1,args.num_users))
    l2eval = l2eval*-1

    removed = []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(1,args.epochs+1):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        if args.frac > 1:
            m = int(args.frac)
        else:
            m = max(int(args.frac * args.num_users), 1)
        eva_locals = []
        loss_reward = {}


        # client selection
        if args.client_sel != 'random':
            idxs_users = bandit.requireArms(m)
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],locallr=locallr)
            w, loss, newlr = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            loss_reward[idx] = loss


            if FA_round(args,iter) is True:
                eva = SingleBgdUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                evaw = eva.train(net=copy.deepcopy(net_glob).to(args.device))
                eva_locals.append(copy.deepcopy(evaw))

        # Evaluate L2 norm and update rewards
        if FA_round(args,iter) is True:
            l2n = l2NormEvaluate(copy.deepcopy(w_old),copy.deepcopy(eva_locals))
            l2n = np.array(l2n) * math.sqrt(numsamples) * -1
            
            if args.cut_thres > 0:
                w_cutted = []
                l2nThres = args.cut_thres * np.mean(l2n)
                removed = []
                for i in range(len(l2n)):
                    if l2n[i] < l2nThres:
                        w_cutted.append(w_locals[i])
                    else:
                        killuser = idxs_users[i]
                        removed.append(domirank[killuser])
                w_locals = w_cutted

            rewards = {}
            for i in range(len(l2n)):
                clientidx = idxs_users[i]
                rewards[clientidx] = l2n[i]

                # write log
                l2eval[iter-2][clientidx] = l2n[i]

            if args.sampling == 'staged':
                w_observe = []
                for i in range(60):
                    ob = SingleBgdUpdate(args=args, dataset=dataset_train, idxs=dict_users[i])
                    obs = ob.train(net=copy.deepcopy(net_glob).to(args.device))
                    w_observe.append(copy.deepcopy(obs))
                l2_observe = l2NormEvaluate(copy.deepcopy(w_old),copy.deepcopy(w_observe),replace_avg=FedAvg(eva_locals))
                l2_observe = np.array(l2_observe) * math.sqrt(numsamples) * -1
                for i in range(len(l2_observe)):
                    l2eval[iter-2][i] = l2_observe[i]
                

            
            if args.client_sel == 'fedacs' or args.client_sel == 'rexp3':
                bandit.updateWithRewards(rewards)
            if args.client_sel == 'oort':
                bandit.updateWithRewards(loss_reward)
            
            if iter % args.testing == 0 and args.testing > 0:
                avgl2n = sum(l2n)/len(l2n)
                rewardlog.append(avgl2n)

        # update global weights
        if args.cmfl is False:
            w_glob = FedAvg(w_locals)
        else:
            w_glob = FedAvgWithCmfl(w_locals,copy.deepcopy(w_old))

        w_old = copy.deepcopy(w_glob)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob.cuda()

        domirankrecord = []
        for client in idxs_users:
            domirankrecord.append(domirank[client])
        domirankrecord.sort()

        hitmaplog.append(domirankrecord)


        # Log domi
        if iter % args.testing == 0 and args.testing > 0:
            domi = []
            for client in idxs_users:
                domi.append(dominance[client])
            domilog.append(float(sum(domi)/len(domi)))

            # test and log accuracy
            
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Round {:3d}, Testing accuracy: {:.2f}".format(iter, acc_test))
            net_glob.train()
            print("Dominance ranking: " + str(domirankrecord))
            if len(removed) > 0:
                print("Removed: " + str(removed))

            testacc.append(float(acc_test))

        # Learning rate decay
        locallr = newlr
        
    # =====Rounds terminals=====
    logidx = str(args.log_idx)

    if args.log_idx < 0:
        filepath = "./save/acc.log"
    else:
        filepath = "./save/acc/"+logidx+'.log'
    
    with open(filepath,'w') as fp:
        for i in range(len(testacc)):
            content = str(testacc[i]) + '\n'
            fp.write(content)
        print('Acc log written')

    if args.log_idx < 0:
        filepath = "./save/reward.log"
    else:
        filepath = "./save/reward/"+logidx+'.log'

    with open(filepath,'w') as fp:
        for i in range(len(rewardlog)):
            content = str(rewardlog[i]) + '\n'
            fp.write(content)
        print('Reward log written')

    if args.log_idx < 0:
        filepath = "./save/domi.log"
    else:
        filepath = "./save/domi/"+logidx+'.log'

    with open(filepath,'w') as fp:
        for i in range(len(domilog)):
            content = str(domilog[i]) + '\n'
            fp.write(content)
        print('Dominance log written')

    if args.log_idx < 0:
        filepath = "./save/hitmap.xlsx"
    else:
        filepath = "./save/hitmap/"+logidx+'.xlsx'

    np_hitmaplog = np.array(hitmaplog)
    pd_hitmaplog = pd.DataFrame(np_hitmaplog)
    writer = pd.ExcelWriter(filepath)
    pd_hitmaplog.to_excel(writer,'sheet_1',float_format='%d')
    writer.save()
    writer.close()

    print('Hitmap record written')

    if args.sampling == 'staged':
        dominance = np.expand_dims(dominance,0)
        l2eval = np.concatenate((dominance,l2eval),axis=0)
        writer = pd.ExcelWriter('l2eval.xlsx')
        l2pandas = pd.DataFrame(l2eval)
        l2pandas.to_excel(writer, 'sheet_1', float_format='%f')
        writer.save()
        writer.close()
        print('L2 record written')
    