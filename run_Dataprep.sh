#!/usr/bin/bash
echo "Doing : $0 $@" && echo 
    # Default variables
    stage=-1
    data_type=
        trn_type=train_v0
        val_type=val_v0
        tst_type=test_v0
        degrade_type=degrade_v0
    Archi_vrs=""
    stage_v= # 1 2 3
        nj=32
        sr=8000

    # Read input Variables :
    . ./utils/parse_options.sh

    # Set global variables :
    . ./run_GlobalVariables.sh \
        --stage $stage \
        --data_type $data_type \
        --trn_type $trn_type \
        --val_type $val_type \
        --tst_type $tst_type \
        --degrade_type $degrade_type \
        --Archi_vrs $Archi_vrs \

    [ -f $stage ]     && echo "stage=$stage not defined" && exit 1
    [ -f $data_type ] && echo "data_type=$data_type not defined" && exit 1
    [ -f $Archi_vrs ] && echo "Archi_vrs=$Archi_vrs not defined" && exit 1

    # Set bash to 'debug' mode, it will exit on :
        # -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
        set -e
        set -u
        set -o pipefail

## Dataprep 
if [[ $stage == 0 ]]; then
    curr_log_dir=$log_dir/step1_dataprep_raw2dict; mkdir -pv $curr_log_dir
    echo && echo "Doing Stage $stage.$stage_v : step1_dataprep_raw2dict : saving raw wave files to dictionaries ..."
    if   [[ $stage_v == 1 ]]; then ## step1_dataprep_raw2dict : saving train/val/test

        echo "python utils/step1_dataprep_raw2dict.py * * * > $curr_log_dir/$data_type.log"
        python utils/step1_dataprep_raw2dict.py \
            -raw_type "clean" \
            -data_type $data_type \
            > $curr_log_dir/$data_type.log

    elif   [[ $stage_v == 2 ]]; then ## step1_dataprep_raw2dict : saving nse
        echo "python utils/step1_dataprep_raw2dict.py * * * > $curr_log_dir/nse.log"
        python utils/step1_dataprep_raw2dict.py \
            -raw_type "nse"   \
            -data_type $degrade_type \
            > $curr_log_dir/nse.log

    elif [[ $stage_v == 3 ]]; then ## step1_dataprep_raw2dict : saving rir
        echo "python utils/step1_dataprep_raw2dict.py * * * > $curr_log_dir/rir.log"
        python utils/step1_dataprep_raw2dict.py \
            -raw_type "rir"   \
            -data_type $degrade_type \
            > $curr_log_dir/rir.log

    fi
    echo "Done. Stage $stage.$stage_v : step1_dataprep_raw2dict : saving raw wave files to dictionaries"
    exit 0
fi
