#! /bin/bash

cd data/
ffmpeg -i loopy.mp4 -qscale:v 1 -qmin 1 -vf fps=2  input/input_%4d.jpg

cd ..
python convert.py -s data
python train.py -s data -m data/output --eval