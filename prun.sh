#!/bin/bash

rm -f ./save/acc/*
rm -f ./save/domi/*
rm -f ./save/reward/*
rm -f ./save/hitmap/*
rm -f ./templog/*

time1=$(date)
echo 'Start:' $time1

for trial in {1..10..1}
do
{
echo "Trial $trial"
python main_fed.py --epochs 1000 --local_ep 5 --local_bs 400 --sampling dirichlet --testing 5  --client_sel fedacs --num_data 2000 --extension 4 --historical_rounds 0 --log_idx $trial #&> ./templog/$trial.log
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