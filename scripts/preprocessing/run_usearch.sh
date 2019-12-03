#/usr/bash

# $1 is a fasta file
# id 0.2 = 0.2% of missmatches allowed

# check if folder clusters already exists and if doesn't, create it!
[ ! -d "./clusters/" ] && mkdir ./clusters

folder=./clusters/${1:5:-6}
mkdir ${folder}
usearch -cluster_fast $1 -id 0.2 -clusters ${folder}/c

