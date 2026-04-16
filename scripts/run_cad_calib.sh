#!/bin/bash

RERUN=ON

# cad calib
python src/gmmcalib.py \
 --data_path ../data/single_chair/ \
 --config_file_path ../config/config_single_chair.yaml \
 --model_path ../data/models/chair.obj \
 --method cad_calib 