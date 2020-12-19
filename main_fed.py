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

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_merged_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvgWithCmfl, FedAvgWithL2
from models.test import test_img

import utils.evaluate as evaluate


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # seed
    np.random.seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid == 0:
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.iid == 1:
            dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.iid == 2:
            dict_users = mnist_merged_iid(dataset_train, args.num_users)
        else:
            exit("Bad argument: iid")
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid == 0:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
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

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.testing > 0:
        testacc = []

    l2eval = np.zeros((args.epochs-1,10))

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(1,args.epochs+1):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = [0,1,2,3,4,5,6,75,85,95]
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        if args.mode == 0:
            w_glob = FedAvg(w_locals)

            if iter != 1:
                l2n = evaluate.l2NormEvaluate(w_old,w_locals)
                for i in range(len(idxs_users)):
                    l2eval[iter-2][i] = l2n[i]

        elif args.mode == 1:
            if iter == 1:
                w_glob = FedAvg(w_locals)
            else:
                w_glob = FedAvgWithCmfl(w_locals,w_old,threshold=0.6,mute=False,checking=(idxs_users,70))


        elif args.mode == 2:
            if iter == 1:
                w_glob,avgl2 = FedAvgWithL2(w_locals,None,0,boosting=False,cutting=False,mute=False)
                l2_record = avgl2
            else:
                w_glob,avgl2 = FedAvgWithL2(w_locals,w_old,l2_record,boosting=True,cutting=True,mute=False,checking=(idxs_users,70))
                l2_record = l2_record * 0.8 + avgl2 * 0.2

        else:
            exit('Bad argument: mode')

        w_old = w_glob
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)


        # test the model
        if iter > 0 and iter % args.testing == 0 and args.testing > 0:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Round {:3d}, Testing accuracy: {:.2f}".format(iter, acc_test))
            net_glob.train()

            testacc.append((iter,float(acc_test)))

    

    with open("log.txt",'w') as fp:
        for i in range(len(testacc)):
            content = str(testacc[i][1]) + '\n'
            fp.write(content)
        print('Log written')


    writer = pd.ExcelWriter('l2eval.xlsx')
    l2pandas = pd.DataFrame(l2eval)
    l2pandas.to_excel(writer, 'sheet_1', float_format='%f')
    writer.save()
    writer.close()

    '''
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    
    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    '''
