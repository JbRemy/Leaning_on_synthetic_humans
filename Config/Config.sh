#!/usr/bin/env bash

username=${1:-'username'}
password=${2:-'password'}

# This file sets up the machine.
# Prerequisit : Python3 (Anaconda)

#creating environement
conda env create -f environment.yml

apt-get install ffmpeg

# Clonning the surreal repo
#git clone https://github.com/gulvarol/surreal

# Downloading data
#Data/download_data.sh Data ${username} ${password}

# Creating directories
for data_set in 'train' 'test' 'val'; do
    mkdir ./Data/${data_set}
    for set in 'images' 'matrix'; do
        mkdir ./Data/${data_set}/${set}
    done
done
