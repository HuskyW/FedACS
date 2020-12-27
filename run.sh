#!/bin/bash

for trial in {1..5..1}
do
echo "================================="
echo "          Trial $trial"
echo "================================="
python main_fed.py --epochs 5 --local_ep 1 --local_bs 300 --model cnn --dataset mnist --iid 4 --testing 1 --mode 0 --client_sel 1
name="${trial}.log"
accpath="./save/acc/$name"
domipath="./save/domi/$name"
rewardpath="./save/reward/$name"
cp ./acc.log $accpath
cp ./domi.log $domipath
cp ./reward.log $rewardpath
done