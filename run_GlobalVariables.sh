#!/usr/bin/bash
echo "Doing : $0 $@"

stage= # 1 2 3
data_type=
    trn_type=
    val_type=
    tst_type=
    degrade_type=
Archi_vrs=
    nj=32
    sr=8000

    if [[ $stage == -1 ]]; then
        echo "Here are the global variables"
    fi
. ./utils/parse_options.sh
    if [[ $stage == -1 ]]; then
        echo "  data_type                       : $data_type"
        echo "      trn_type                    : $trn_type"
        echo "      val_type                    : $val_type"
        echo "      tst_type                    : $tst_type"
        echo "      degrade_type                : $degrade_type"
        echo
    fi

## Params
    num_a=${#nj}
    if [[ $stage == -1 ]]; then
        echo "  nj                              : $nj"
        echo "  sr                              : $sr"
        echo "  num_a                           : $num_a"
        echo
    fi

## Logs 
    log_dir_dataprep=./logs/${data_type}
    log_dir_DNN=./logs/${Archi_vrs}
    if [[ $stage == -1 ]]; then
        echo "  log_dir_dataprep                : $log_dir_dataprep"
        echo "  log_dir_DNN                     : $log_dir_DNN"
        echo
    fi
    mkdir -pv $log_dir_dataprep
    mkdir -pv $log_dir_DNN

## Current : Files/Dir
if [ ! -f $data_type ]; then
    data_dir=./data/${data_type}
        wav_scp=${data_dir}/wav_clean.scp
        split_dir=$data_dir/split
    wav_dir=./wav/${data_type}; 
    if [[ $stage == -1 ]]; then
        echo "-- ## Current : Files/Dir"
        echo "  data_dir                        : $data_dir"
        echo "      wav_scp                     : $wav_scp"
        echo "      split_dir                   : $split_dir"
        echo "  wav_dir                         : $wav_dir"
        echo
    fi
fi

## Train : Files/Dir
if [ ! -f $trn_type ]; then
    trn_data_dir=./data/${trn_type}
        trn_wav_scp=${trn_data_dir}/wav_clean.scp
        trn_split_dir=$trn_data_dir/split
    trn_wav_dir=./wav/${trn_type}; 
    if [[ $stage == -1 ]]; then
        echo "-- ## Train : Files/Dir"
        echo "  trn_data_dir                    : $trn_data_dir"
        echo "      trn_wav_scp                 : $trn_wav_scp"
        echo "      trn_split_dir               : $trn_split_dir"
        echo "  trn_wav_dir                     : $trn_wav_dir"
        echo
    fi
fi

## Val : Files/Dir
if [ ! -f $val_type ]; then
    val_data_dir=./data/${val_type}
        val_wav_scp=${val_data_dir}/wav_clean.scp
        val_split_dir=$val_data_dir/split
    val_wav_dir=./wav/${val_type}; 
    if [[ $stage == -1 ]]; then
        echo "-- ## Val : Files/Dir"
        echo "  val_data_dir                    : $val_data_dir"
        echo "      val_wav_scp                 : $val_wav_scp"
        echo "      val_split_dir               : $val_split_dir"
        echo "  val_wav_dir                     : $val_wav_dir"
        echo
    fi
fi

## Test : Files/Dir
if [ ! -f $tst_type ]; then
    tst_data_dir=./data/${tst_type}
        tst_wav_scp=${tst_data_dir}/wav_clean.scp
        tst_split_dir=$tst_data_dir/split
    tst_wav_dir=./wav/${tst_type}; 
    if [[ $stage == -1 ]]; then
        echo "-- ## Test : Files/Dir"
        echo "  tst_data_dir                    : $tst_data_dir"
        echo "      tst_wav_scp                 : $tst_wav_scp"
        echo "      tst_split_dir               : $tst_split_dir"
        echo "  tst_wav_dir                     : $tst_wav_dir"
        echo
    fi
fi

## Degrade : Files/Dir
if [ ! -f $degrade_type ]; then
    degrade_data_dir=./data/${degrade_type}
        degrade_wav_nse=${degrade_data_dir}/wav_nse.scp
        degrade_wav_RIR=${degrade_data_dir}/wav_RIR.scp
        degrade_split_dir=$degrade_data_dir/split
    degrade_wav_dir=./wav/${degrade_type}; 
    if [[ $stage == -1 ]]; then
        echo "-- ## Degrade : Files/Dir"
        echo "  degrade_data_dir                : $degrade_data_dir"
        echo "      degrade_wav_nse             : $degrade_wav_nse"
        echo "      degrade_wav_RIR             : $degrade_wav_RIR"
        echo "      degrade_split_dir           : $degrade_split_dir"
        echo "  degrade_wav_dir                 : $degrade_wav_dir"
        echo
    fi
fi
