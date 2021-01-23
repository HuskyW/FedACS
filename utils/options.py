#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=200, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.05, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=300, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    # modified
    parser.add_argument('--sampling', type=str, default='iid', help='iid || uniform || pareto || ipareto || dirichlet')
    parser.add_argument('--testing', type=int, default=1, help="test the model after some rounds, -1: never")
    parser.add_argument('--client_sel', type=str, default='random', help="Client selection, random || fedacs || rexp3 || oort")
    parser.add_argument('--log_idx', type=int, default=-1, help="Index of log file")
    parser.add_argument('--faf', type=int, default=0, help="How offen FA round is used, -1:never, 0:always, 1:early stop")
    parser.add_argument('--lrd', type=float, default=0.9993, help="Learning rate decay")
    parser.add_argument('--extension', type=int, default=10, help="Candidate extension")
    parser.add_argument('--num_data', type=int, default=-1, help="Number of data in each client, -1:auto")
    parser.add_argument('--historical_rounds', type=int, default=0, help="How many rounds are remenbered by bandit for historical comparison? 0: never")
    parser.add_argument('--cut_thres', type=float, default=0, help="Threshold to remove bad updates, 0: never")
    parser.add_argument('--cmfl', action='store_true', help='use CMFL')

    args = parser.parse_args()
    return args
