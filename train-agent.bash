#!/bin/bash

#arguments:
if [[ $# != 1 && $# != 3 ]] ;
then
   echo "usage: train-agent.bash output_dir network_init er_init"
   exit
fi
#output_dir - directory to put logs and results into
output_dir=$1
echo $output_dir
#network_init - file to read the first network from
network_init=$2
echo $network_init
#er_init - file to read the first experience replay from
er_init=$3
echo $er_init

#parameters:
#p - location of python script
p=python
#rounds - rounds of training
rounds=2
#processes - only for self-play
processes=2
#games_per_process_round
games_per_process_round=1
#positions_per_train_cycle
positions_per_train_cycle=1024

#create output_dir
mkdir $output_dir
pwd
#variables:
#global_network
if [ $# == 1 ];
then
    global_network=$output_dir"/network_init"
    echo "initializing network at $global_network"
    $p init-network.py $global_network
else
    global_network=$network_init
    echo "using network at $global_network"    
fi
#global_er
global_er=$output_dir"/er_init.npz"
if [ $# == 1 ];
then
    echo "initializing ER at $global_er"
    $p combine-ers.py $global_er #creates an empty er
else
    echo "using ER at $er_init"
    $p combine-ers.py $global_er $er_init
fi

#for _ in rounds of training
for (( round=0; round<$rounds; round++ ));
do
    echo "round $round"
    #run processes to self play
    echo "self-play:"
    seq 1 $processes | parallel $p 'self-play.py' $games_per_process_round $global_network "$output_dir"/round_"$round"_process_{}_save.npz
    #aggregate ers into global_er
    files=""
    for (( i=1; i<=$processes; i++ )); do
	files=$files$output_dir"/round_"$round"_process_"$i"_save.npz "
    done
    echo "aggregating:"$files
    $p combine-ers.py $global_er $global_er $files
    #run training step to form new global_network
    echo "training:"
    new_network=$output_dir"/network_round_"$round
    $p self-train.py $positions_per_train_cycle $global_er \
       $global_network $new_network
    global_network=$new_network
done
