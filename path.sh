#!/usr/bin/bash
conda activate YOURENVIRONMENT
<< COMMENTS
    python                  3.6
    tensorflow              2.0.0
    numpy                   1.18.1
    python-speech-features  0.6
    librosa                 0.7.2
COMMENTS

## The name of the train, val, test clean datasets
trn_type=train_v1
val_type=val_v1
tst_type=test_v1

## The name of your noise & rir datasets
degrade_type=degrade_v1

## The name of your architecture
Archi_vrs=v1_TDR

## Setting other global variables
curr_opts="--trn_type $trn_type --val_type $val_type --tst_type $tst_type --degrade_type $degrade_type --Archi_vrs $Archi_vrs"
. ./run_GlobalVariables.sh --stage "-1" --data_type $trn_type $curr_opts
