#!/usr/bin/bash
echo "Doing : $0 $@" && echo 
    # Default variables
    stage=-1
    data_type='.'
        trn_type=
        val_type=
        tst_type=
        degrade_type=
    Archi_vrs=""
    stage_v= # 1 2 3
        nj=32
        sr=8000

    # Read input Variables :
    . ./utils/parse_options.sh

    # Set global variables :
    . ./run_GlobalVariables.sh \
        --data_type $data_type \
        --stage $stage \
        --trn_type $trn_type \
        --val_type $val_type \
        --tst_type $tst_type \
        --degrade_type $degrade_type \
        --Archi_vrs $Archi_vrs \

    [ -f $stage ]     && echo "stage=$stage not defined" && exit 1
    [ -f $Archi_vrs ] && echo "Archi_vrs=$Archi_vrs not defined" && exit 1

    # Set bash to 'debug' mode, it will exit on :
        # -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
        set -e
        set -u
        set -o pipefail

## Set up
if [[ $stage == 1 ]]; then ## Setup Model
    echo && echo "Stage $stage  : Setup Model ..."
    
    curr_log_dir=$log_dir/step2_modelsetup; mkdir -pv $curr_log_dir

    if   [[ $stage_v == 1 ]]; then ## Setup Model (denoise)
        curr_log=$curr_log_dir/step2_modelsetup.${Enh_mode}.log
        echo && echo "python utils/step2_modelsetup.py > $curr_log"
        python -u utils/step2_modelsetup.py \
            -Archi_vrs $Archi_vrs \
            -Enh_mode 'denoise' \
            > $curr_log

    elif   [[ $stage_v == 2 ]]; then ## Setup Model (dereverb)
        curr_log=$curr_log_dir/step2_modelsetup.${Enh_mode}.log
        echo && echo "python utils/step2_modelsetup.py > $curr_log"
        python -u utils/step2_modelsetup.py \
            -Archi_vrs $Archi_vrs \
            -Enh_mode 'dereverb' \
            > $curr_log
    fi

    echo "Stage $stage  : Setup Model done ."
    exit 0
fi

if [[ $stage == 3 ]]; then ## Train Model
    echo && echo "Stage $stage  : Train Model ..."
    if   [[ $stage_v == 1 ]]; then ## Train Model (denoise)

        curr_log_dir=$log_dir/step2_denoise_training; mkdir -pv $curr_log_dir
        curr_log=$curr_log_dir/step2_denoise_training.log
        echo "python utils/step2_denoise_training.py > $curr_log "

        python -u utils/step2_denoise_training.py \
            -Enh_mode 'denoise' \
            -Archi_vrs $Archi_vrs \
            -trn_type $trn_type \
            -val_type $val_type \
            -degrade_type $degrade_type \
            > $curr_log 

    elif [[ $stage_v == 2 ]]; then ## Train Model (dereverb)

        curr_log_dir=$log_dir/step2_dereverb_training; mkdir -pv $curr_log_dir
        curr_log=$curr_log_dir/step2_dereverb_training.log
        echo "python utils/step2_dereverb_training.py > $curr_log "

        python -u utils/step2_dereverb_training.py \
            -Enh_mode 'dereverb' \
            -Archi_vrs $Archi_vrs \
            -trn_type $trn_type \
            -val_type $val_type \
            -degrade_type $degrade_type \
            > $curr_log 

    elif [[ $stage_v == 3 ]]; then ## Train Joint Model (denoise+dereverb)

        curr_log_dir=$log_dir/step4_joint_training; mkdir -pv $curr_log_dir
        curr_log=$curr_log_dir/step4_joint_training.log
        echo "python utils/step4_joint_training.py > $curr_log "

        python -u utils/step4_joint_training.py \
            -Enh_mode 'joint' \
            -Archi_vrs $Archi_vrs \
            -trn_type $trn_type \
            -val_type $val_type \
            -degrade_type $degrade_type \
            > $curr_log 

    fi
    echo "Stage $stage  : Train Model done ."
    exit 0
fi

if [[ $stage == 4 ]]; then ## Inference
    echo && echo "Stage $stage  : Test Model ..."

    if   [[ $stage_v == 1 ]]; then ## Inference (denoise)
        curr_log_dir=$log_dir/step3_denoise_testing; mkdir -pv $curr_log_dir
        curr_log=$curr_log_dir/step3_denoise_testing.log
        echo "python utils/step3_denoise_testing.py > $curr_log &"
        python -u utils/step3_denoise_testing.py \
            -Enh_mode 'denoise' \
            -Archi_vrs $Archi_vrs \
            -data_type $data_type \
            -degrade_type $degrade_type \
            > $curr_log

    elif [[ $stage_v == 2 ]]; then ## Inference (dereverb)
        curr_log_dir=$log_dir/step3_dereverb_testing; mkdir -pv $curr_log_dir
        curr_log=$curr_log_dir/step3_dereverb_testing.log
        echo "python utils/step3_dereverb_testing.py > $curr_log &"
        python -u utils/step3_dereverb_testing.py \
            -Enh_mode 'dereverb' \
            -Archi_vrs $Archi_vrs \
            -data_type $data_type \
            -degrade_type $degrade_type \
            > $curr_log

    fi
    echo "Stage $stage  : Test Model done ."
    exit 0
fi
