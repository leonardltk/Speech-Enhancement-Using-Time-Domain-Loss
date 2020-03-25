from __future__ import print_function # pdb.set_trace() # !import code; code.interact(local=vars())
if True:
    import os,sys,datetime
    sys.path.insert(0, 'utils')
    from _helper_funcs_ import *
    START_TIME=datetime.datetime.now()
    datetime.datetime.now() - START_TIME
    print(f"===========\npython {sys.argv}\n    Start_Time:{START_TIME}\n===========")
if True:
    print('############ Printing Config Params ############')
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-Enh_mode')
    parser.add_argument('-Archi_vrs')
    parser.add_argument('-trn_type')
    parser.add_argument('-val_type')
    parser.add_argument('-degrade_type')
    # 
    args=parser.parse_args()
    conf_DNN=conf.DNN_conf(args.Enh_mode, args.Archi_vrs) # joint
    if True: ## Update parameters for joint model
        conf_DNN.context=21 # We load in s21 instead of s11 due to denoising prior to dereverbing
        str(conf_DNN)
    conf_DNN_denoise=conf.DNN_conf('denoise', args.Archi_vrs)
    conf_DNN_dereverb=conf.DNN_conf('dereverb', args.Archi_vrs)
    conf_sr=conf.SR_conf()
    conf_train=conf.Data_conf(args.trn_type)
    conf_val=conf.Data_conf(args.val_type)
    conf_degrade=conf.Degrade_conf(args.degrade_type)
    # 
    pp = pprint.PrettyPrinter(indent=4)
    use_cpu=False
    if use_cpu:
        print('\n\nUSING CPU !!!\n\n')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        print('\n\nUSING GPU !!!\n\n')
        os.environ["CUDA_VISIBLE_DEVICES"]="3"
        # os.environ["CUDA_VISIBLE_DEVICES"]=conf_DNN.CUDA_VISIBLE_DEVICES
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # sess = tf.Session(config=config)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.enable_eager_execution()
    # tf.config.experimental_run_functions_eagerly(True)
    print('############ End of Config Params ##############')

if True : ## versioning
    functions_dict = versioning(conf_DNN_denoise.Archi_vrs)
    build_denoise_model = functions_dict['build_denoise_model']
    build_dereverb_model = functions_dict['build_dereverb_model']
    joint_model_wave2wave = functions_dict['joint_model_wave2wave']
    generate_wavetolps_JointNN = functions_dict['generate_wavetolps_JointNN']
if True : ## (WAVE) Full Data Read
    print("    ## Reading raw dictionary of the data...")
    Clean_WAVE_raw_dict= dump_load_pickle(conf_train.Clean_WAVE_raw_dict_path, 'load')
    NSE_WAVE_dict= dump_load_pickle(conf_degrade.NSE_WAVE_dict_path, 'load')
    RIR_WAVE_dict= dump_load_pickle(conf_degrade.RIR_WAVE_dict_path, 'load')
    print('Clean_WAVE_raw_dict  :',len(Clean_WAVE_raw_dict))
    print('NSE_WAVE_dict        :',len(NSE_WAVE_dict))
    print('RIR_WAVE_dict        :',len(RIR_WAVE_dict))

if True : ## 'denoise'
    ## Loading abs-to-abs Denoising Model
    print("\n## Loading Model")
    load_model_keras_kwargs={
        'model_mode':'path', 
        'custom_objects':{'s11_to_s1':s11_to_s1}, 
        'model_path':conf_DNN.Enhance_model_path, 
        'weights_path':conf_DNN.Enhance_weights_path, 
        'verbose':True
    }
    model_denoise = load_model_keras( **load_model_keras_kwargs)
    model_denoise.summary()
if True : ## 'dereverb'
    ## Loading abs-to-abs Dereverb Model
    print("\n## Loading Model")
    load_model_keras_kwargs={
        'model_mode':'path', 
        'custom_objects':{'s11_to_s1':s11_to_s1}, 
        'model_path':conf_DNN.Enhance_model_path, 
        'weights_path':conf_DNN.Enhance_weights_path, 
        'verbose':True
    }
    model_dereverb = load_model_keras( **load_model_keras_kwargs)
    model_dereverb.summary()
if True : ## 'joint'
    ## Linking it up with up Denoising & Dereverb Models 
    ## Denoise model (wave to abs)
    joint_model_wave2wave_kwargs = {
        'mdl_name':'joint_model_wave2wave',
        'dereverb_model':model_dereverb,
        'denoise_model':model_denoise,
            'context':conf_DNN.context,
            'num_freq_bins':conf_sr.num_freq_bins,
            'n_fft':conf_sr.n_fft,
            'frame_step':conf_sr.hop_length,
                'mdl_mode':'training', 
    }
    joint_model_e2e = joint_model_wave2wave(**joint_model_wave2wave_kwargs)
    joint_model_e2e.summary()
    save_model(joint_model_e2e, 
                    model_path=conf_DNN.Training_model_path, 
                    weights_path=conf_DNN.Training_weights_path,
                    verbose=True)
if True : ## Compiling
    joint_model_e2e = compile_opt(_mod_in=joint_model_e2e, _opt_mode=conf_DNN.opt_mode, _opt_dict=conf_DNN.opt_dict, compile_dict=conf_DNN.compile_dict)

if True : ## Callbacks
    reduce_lr = ReduceLROnPlateau(**conf_DNN.ReduceLROnPlateau_kwargs)
    ckpt = ckpt_saving(conf_DNN.Ckpt_Mod_Weights_fold, conf_DNN.ModelCheckpoint_kwargs, save_all=True)
    csv_log = CSVLogger(conf_DNN.csv_log_path, separator='\t', append=True)
    savemodel = SaveModel_joint(joint_model_e2e, conf_DNN, conf_DNN_dereverb, conf_DNN_denoise)

if True : ## Training for wave-to-lps
    print("\n## Training - joint_model_e2e.fit_generator")
    kwangs_in={ 
        'num_freq_bins':conf_sr.num_freq_bins,
        'n_fft':conf_sr.n_fft,
        'hop_length':conf_sr.hop_length,
        'batch_size':   conf_DNN.batch_size,
        'context':conf_DNN.context,
            'snr_list':conf_degrade.snr_list,
            'desired_power':conf_sr.desired_power,
    }
    history = joint_model_e2e.fit( 
            generate_wavetolps_JointNN('fit',
                Clean_WAVE_raw_dict, NSE_WAVE_dict, RIR_WAVE_dict,
                **kwangs_in),
            steps_per_epoch=conf_DNN.steps_per_epoch,
            epochs=conf_DNN.epochs,
            callbacks=[reduce_lr, ckpt, csv_log, savemodel],
            initial_epoch=conf_DNN.initial_epoch,
            verbose=2,)

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
