#!/usr/bin/env bash

outdir=${1:-'/temp'}
username=${2:-'username'}
password=${3:-'password'}

# This script can be used to download all the data.
# Partial download can be enabled by modifying the loops below.
# Run `chmod u+x download_surreal.sh` and pass the path of the output directory as follows:
# `./download_surreal.sh /path/to/dataset yourusername yourpassword`
# Replace the path with the output folder you want to download, potentially '~/datasets'.
# Replace username and password with the credentials you received by e-mail upon accepting license terms.
# You can remove -q option to debug.

#modified

for dataset in 'cmu'; do
    for setname in 'val'; do
        for modality in '.mp4' '_info.mat' ; do
        	echo 'Downloading '${dataset}' dataset '${setname}' set, files with '${modality}

        	#unncomment to create a small subset
        	head -10 /home/jbremy/surreal/download/files/files_${dataset}_${setname}${modality}.txt > /home/jbremy/surreal/download/files/test_files_${dataset}_${setname}${modality}.txt
            wget --user=${username} --password=${password} -m -q -i /home/jbremy/surreal/download/files/test_files_${dataset}_${setname}${modality}.txt --no-host-directories -P ${outdir}

            #wget --user=${username} --password=${password} -m -q -i /surreal/download/files/files_${dataset}_${setname}${modality}.txt --no-host-directories -P ${outdir}
        done
    done
done