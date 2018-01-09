#!/usr/bin/env bash

outdir=${1:-'/temp'}
username=${2:-'username'}
password=${3:-'password'}

# This script can be used to download all the data.
# Partial download can be enabled by modifying the loops below.
# Run `chmod u+x download_surreal.sh` and pass the path of the output directory as follows:
# `./download_surreal.sh /path/to/data_set yourusername yourpassword`
# Replace the path with the output folder you want to download, potentially '~/data_sets'.
# Replace username and password with the credentials you received by e-mail upon accepting license terms.
# You can remove -q option to debug.

#modified

for data_set in 'cmu'; do
    for set_name in 'val'; do #'train' 'test'
        for modality in '.mp4' '_info.mat' ; do
        	echo 'Downloading '${data_set}' data_set '${set_name}' set, files with '${modality}

        	#unncomment to create a small subset
        	head -10 ./surreal/download/files/files_${data_set}_${set_name}${modality}.txt > ./surreal/download/files/test_files_${data_set}_${set_name}${modality}.txt
            wget --user=${username} --password=${password} -m -q -i ./surreal/download/files/test_files_${data_set}_${set_name}${modality}.txt --no-host-directories -P ${outdir}

            #wget --user=${username} --password=${password} -m -q -i /surreal/download/files/files_${data_set}_${set_name}${modality}.txt --no-host-directories -P ${outdir}
        done
    done
done