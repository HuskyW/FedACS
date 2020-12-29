#!/bin/bash

rm -f ./save/acc/*
rm -f ./save/domi/*
rm -f ./save/reward/*
for trial in {1..100..1}
do
echo -e "\n\n\n"
echo "================================="
echo "           Trial $trial"
echo "================================="
echo -e "\n\n\n"
python main_fed.py --epochs 1000 --local_ep 1 --local_bs 300 --model cnn --dataset mnist --iid 4 --testing 10 --mode 0 --client_sel 0
name="${trial}.log"
accpath="./save/acc/$name"
domipath="./save/domi/$name"
rewardpath="./save/reward/$name"
cp ./acc.log $accpath
cp ./domi.log $domipath
cp ./reward.log $rewardpath
done