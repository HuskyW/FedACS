#!/bin/bash

rm -f ./save/acc/*
rm -f ./save/domi/*
rm -f ./save/reward/*
rm -f ./templog/*

time1=$(date)
echo 'Start:' $time1

for trial in {1..5..1}
do
{
echo "Trial $trial"
python main_fed.py --epochs 500 --local_ep 5 --local_bs 400 --model cnn --dataset cifar --iid 0 --testing 5  --client_sel 0 --num_data 2000 --num_users 200 --faf 0 --frac 0.05 --lrd 0.9993 --extension 8 --log_idx $trial #&> ./templog/$trial.log
}&

rem=$(($trial%5))

if [ $rem -eq 0 ];then
wait
fi

done
wait

time2=$(date)
echo 'Start:' $time1
echo 'End:' $time2