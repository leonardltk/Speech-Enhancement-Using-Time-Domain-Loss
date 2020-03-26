# TRAINING SCRIPT

## Init (To set global variables in terminal)
```bash
# See the requirements in this file
. path.sh
```

## Dataprep
```bash
## step1_dataprep_raw2dict : saving train/val/test
nohup bash ./run_Dataprep.sh --stage 0 --stage_v 1 --data_type $trn_type $curr_opts &> $log_dir_dataprep/run_Dataprep.${trn_type}.0.1.log &
nohup bash ./run_Dataprep.sh --stage 0 --stage_v 1 --data_type $val_type $curr_opts &> $log_dir_dataprep/run_Dataprep.${val_type}.0.1.log &
nohup bash ./run_Dataprep.sh --stage 0 --stage_v 1 --data_type $tst_type $curr_opts &> $log_dir_dataprep/run_Dataprep.${tst_type}.0.1.log &

## step1_dataprep_raw2dict : saving nse
nohup bash ./run_Dataprep.sh --stage 0 --stage_v 2 --data_type $degrade_type $curr_opts &> $log_dir_dataprep/run_Dataprep.${degrade_type}.0.2.log &

## step1_dataprep_raw2dict : saving rir
nohup bash ./run_Dataprep.sh --stage 0 --stage_v 3 --data_type $degrade_type $curr_opts &> $log_dir_dataprep/run_Dataprep.${degrade_type}.0.3.log &
```

## Model Setup
```bash
## Setup model
nohup bash ./run_DNN.sh --stage 1 --stage_v 1 $curr_opts &> $log_dir_DNN/run_DNN.1.1.log &
nohup bash ./run_DNN.sh --stage 1 --stage_v 2 $curr_opts &> $log_dir_DNN/run_DNN.1.2.log &
```

## Training
```bash
## Train model (Denoise)
nohup bash ./run_DNN.sh --stage 3 --stage_v 1 $curr_opts &> $log_dir_DNN/run_DNN.3.1.log &

## Train model (Dereverb)
nohup bash ./run_DNN.sh --stage 3 --stage_v 2 $curr_opts &> $log_dir_DNN/run_DNN.3.2.log &

## Train model (Joint)
wait # wait for previous two jobs to finish first, before you run the joint training
nohup bash ./run_DNN.sh --stage 3 --stage_v 3 $curr_opts &> $log_dir_DNN/run_DNN.3.3.log &
```

## Inference
```bash
## Inference (denoise) 
nohup bash ./run_DNN.sh --stage 4 --stage_v 1 --data_type $trn_type $curr_opts &> $log_dir_DNN/run_DNN.4.1.log

# Inference (dereverb) 
nohup bash ./run_DNN.sh --stage 4 --stage_v 2 --data_type $trn_type $curr_opts &> $log_dir_DNN/run_DNN.4.2.log

# Inference (Joint) 
nohup bash ./run_DNN.sh --stage 4 --stage_v 3 --data_type $trn_type $curr_opts &> $log_dir_DNN/run_DNN.4.3.log
```
