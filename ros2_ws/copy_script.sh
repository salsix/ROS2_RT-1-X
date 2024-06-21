#!/bin/bash

# Set the number of copies you want to create
num_copies=30
num_train=25

# Set the base name of the source file
src_file="./episodes/test_epi.npy"

rm -r "/home/jonathan/Thesis/rlds_dataset_builder/episodes/data/val"
rm -r "/home/jonathan/Thesis/rlds_dataset_builder/episodes/data/train"

mkdir "/home/jonathan/Thesis/rlds_dataset_builder/episodes/data/val"
mkdir "/home/jonathan/Thesis/rlds_dataset_builder/episodes/data/train"

# Loop to create copies
for i in $(seq -f "%02g" 1 $num_copies); do
    if [ "$i" -gt "$num_train" ];
    then
    	cp "$src_file" "/home/jonathan/Thesis/rlds_dataset_builder/episodes/data/val/test_epi_$i.npy"
    else
    	cp "$src_file" "/home/jonathan/Thesis/rlds_dataset_builder/episodes/data/train/test_epi_$i.npy"
    fi
done
