#!/bin/bash

source activate PDEA

INPUT=${1:-../data/yaml/inputs/Allen-Cahn equation/eval1.yaml}

python ./run_inference.py "$INPUT"

