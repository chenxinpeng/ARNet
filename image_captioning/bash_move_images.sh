#!/usr/bin/env bash

DIR="data/images/mscoco/train2014/*.jpg"

for IMG in $DIR
do
    mv $IMG ./data/images/mscoco
done
