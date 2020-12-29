#!/bin/bash

rm -f ./save/acc/*
rm -f ./save/domi/*
rm -f ./save/reward/*
rm -f ./templog/*


for trial in {1..9..1}
do
{
echo "Trial $trial"
python main_fed.py --epochs 1000 --local_ep 1 --local_bs 300 --model cnn --dataset cifar --num_channels 3 --iid 4 --testing 10 --mode 0 --client_sel 0 --log_idx $trial &> ./templog/$trial.log
}&

rem=$(($trial%3))

if [ $rem -eq 0 ];then
wait
fi

done
wait