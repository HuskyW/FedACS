#!/bin/bash

rm -f ./save/acc/*
rm -f ./save/domi/*
rm -f ./save/reward/*
rm -f ./templog/*

time1=$(date)
echo 'Start:' $time1

for trial in {1..9..1}
do
{
echo "Trial $trial"
python main_fed.py --epochs 1000 --local_ep 1 --local_bs 300 --model cnn --dataset mnist --num_channels 1 --iid 0 --testing 10 --client_sel 0 --log_idx $trial > ./templog/$trial.log
}&

rem=$(($trial%5))

if [ $rem -eq 0 ];then
wait
fi

done
wait

time1=$(date)
echo 'End:' $time1