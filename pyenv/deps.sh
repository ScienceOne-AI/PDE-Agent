#!/bin/bash

source activate PDEA

save_dir="./models"

if [ ! -d $save_dir ]
then
    mkdir -p $save_dir
fi

python ./load_models.py

SRC=$(find ./tmp/models--tbs17--MathBERT/snapshots/ -type d -name '[0-9a-f]*' | head -n1)
DST=./models/MathBERT
mkdir -p "$DST"
cp -r "$SRC"/* "$DST"/
rm -rf ./tmp

# verify
ls "$DST"

