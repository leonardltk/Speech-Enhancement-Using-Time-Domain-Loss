from __future__ import print_function # pdb.set_trace() # !import code; code.interact(local=vars())
if True:
    import os,sys,datetime
    sys.path.insert(0, 'utils')
    from _helper_funcs_ import *
    START_TIME=datetime.datetime.now()
    datetime.datetime.now() - START_TIME
    print(f"===========\npython {sys.argv}\n    Start_Time:{START_TIME}\n===========")
if True:
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-raw_type')
    parser.add_argument('-data_type')
    # 
    args=parser.parse_args()
    raw_type=args.raw_type
    conf_sr=conf.SR_conf()
    # 
    pp = pprint.PrettyPrinter(indent=4)
    global_st = time.time()
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.enable_eager_execution()
    print('############ End of Config Params ##############')

print('\n############ Performing Data Simulation ############')
if   raw_type.lower()=='clean':
    conf_dataprep   =   conf.Data_conf(args.data_type)
    wave_dict_path  =   conf_dataprep.Clean_WAVE_raw_dict_path
    wav_scp         =   conf_dataprep.wav_clean_scp
    hash2curr       =   conf_dataprep.hash2clean

elif raw_type.lower()=='nse':
    conf_degrade    =   conf.Degrade_conf(args.data_type)
    wave_dict_path  =   conf_degrade.NSE_WAVE_dict_path
    wav_scp         =   conf_degrade.wav_nse_scp
    hash2curr       =   conf_degrade.hash2nse

elif raw_type.lower()=='rir':
    conf_degrade    =   conf.Degrade_conf(args.data_type)
    wave_dict_path  =   conf_degrade.RIR_WAVE_dict_path
    wav_scp         =   conf_degrade.wav_RIR_scp
    hash2curr       =   conf_degrade.hash2rir


########################################################
with open(wav_scp,'r') as f_r: curr_LIST=f_r.readlines()
print('## Write hash2file')

_, curr2hash_dict=write_hash2file_dict(hash2curr,curr_LIST)
print('len(curr_LIST)',len(curr_LIST))

curr_dict={}
for curr_c in curr_LIST:
    curr_uttid,curr_wavepath = curr_c.strip('\n').split(' ')
    # curr_uttid = curr_c.replace('\n','').split(' ')[0]
    # curr_wavepath = ' '.join( curr_c.replace('\n','').split(' ')[1:] )
    curr_hash = curr2hash_dict[curr_uttid]
    if curr_hash in curr_dict : 
        print(f'\tHash Collision curr_hash={curr_hash} curr_uttid={curr_uttid} curr_wavepath={curr_wavepath}')
        continue
    curr_WAVE_raw, sr_out=read_audio(curr_wavepath, mode="soundfile", sr=conf_sr.sr, mean_norm=True)
    curr_dict[curr_hash]=curr_WAVE_raw

print('len(curr_dict)',len(curr_dict))
dump_load_pickle(wave_dict_path, 'dump', a=curr_dict)
print('Written to ',wave_dict_path)

#################################################################
END_TIME=datetime.datetime.now()
print(f"===========\
Done python {sys.argv}\
    Start_Time  :{START_TIME}\
    End_Time    :{END_TIME}\
    Duration    :{END_TIME-START_TIME}\
===========")

"""
!import code; code.interact(local=vars())
"""
