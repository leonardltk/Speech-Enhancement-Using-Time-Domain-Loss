from __future__ import print_function # pdb.set_trace() # !import code; code.interact(local=vars())
if True:
    import os,sys,datetime
    sys.path.insert(0, os.path.join('/data2','lootiangkuan','LLutils')) # @ jja178
    # 
    sys.path.insert(0, 'utils')
    from _helper_funcs_ import *
    # 
    START_TIME=datetime.datetime.now()
    datetime.datetime.now() - START_TIME
    print("===========\npython {}\n Start_Time:{}\n===========".format(' '.join(sys.argv),str( START_TIME )))
print('############ Printing Config Params ############')
if True:
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-Enh_mode')
    parser.add_argument('-Archi_vrs')
    parser.add_argument('-data_type')
    parser.add_argument('-degrade_type')
    # 
    args=parser.parse_args()
    conf_DNN=conf.DNN_conf(args.Enh_mode, args.Archi_vrs)
    conf_sr=conf.SR_conf()
    conf_data=conf.Data_conf(args.data_type)
    conf_degrade=conf.Degrade_conf(args.degrade_type)
    # 
    pp = pprint.PrettyPrinter(indent=4)
    global_st = datetime.datetime.now()
    use_cpu=False
    if use_cpu:
        print('\n\nUSING CPU !!!\n\n')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        print('\n\nUSING GPU !!!\n\n')
        # os.environ["CUDA_VISIBLE_DEVICES"]="2"
        os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
    functions_dict = versioning(conf_DNN.Archi_vrs)
    # denoise_model_wave2abs_inf = functions_dict['denoise_model_wave2abs_inf']
    # denoise_reconstruction_test = functions_dict['denoise_reconstruction_test']
    # denoise_uttid_to_inputs_test = functions_dict['denoise_uttid_to_inputs_test']
    dereverb_model_wave2abs_inf = functions_dict['dereverb_model_wave2abs_inf']
    dereverb_reconstruction_test = functions_dict['dereverb_reconstruction_test']
    dereverb_uttid_to_inputs_test = functions_dict['dereverb_uttid_to_inputs_test']

if True : ## (WAVE) Full Data Read
    print("    ## Reading raw dictionary of the data and its cmvn stats...")
    Clean_WAVE_raw_dict= dump_load_pickle(conf_data.Clean_WAVE_raw_dict_path, 'load')
    RIR_WAVE_dict= dump_load_pickle(conf_degrade.RIR_WAVE_dict_path, 'load')
    print('Clean_WAVE_raw_dict  :',len(Clean_WAVE_raw_dict))
    print('RIR_WAVE_dict    :',len(RIR_WAVE_dict))
    # 
    RIR_WAVE_key = list(RIR_WAVE_dict)

if True : ## Loading Trained Models 
    # model_dereverb -> dereverb_model_e2e_inf -> dereverb_reconstruction_mdl
    print("\n## Loading Model")
    if True : ## Load submodel first
        load_model_keras_kwargs={
            'model_mode':'path', 
            'custom_objects':{'s11_to_s1':s11_to_s1}, 
            'model_path':conf_DNN.Enhance_model_path, 
            'weights_path':conf_DNN.Enhance_weights_path, 
            'verbose':True
        }
        model_dereverb = load_model_keras( **load_model_keras_kwargs)
        model_dereverb.summary()
    if True : ## Linking it up with up wave-to-abs Denoising Model 
        dereverb_model_wave2abs_kwargs = {
            'dereverb_model':model_dereverb,
            'context':conf_DNN.context,
            'num_freq_bins':conf_sr.num_freq_bins,
            'n_fft':conf_sr.n_fft,
            'frame_step':conf_sr.hop_length,
        }
        dereverb_model_e2e_inf = dereverb_model_wave2abs_inf('dereverb_model_wave2abs', 
            mdl_mode='inference', 
            **dereverb_model_wave2abs_kwargs)
        dereverb_model_e2e_inf.summary()
    if True : ## get dereverb_reconstruction_mdl
        ## Loading wave-to-abs Dereverb Model
        dereverb_reconstruction_test_kwargs = {
            'num_freq_bins':conf_sr.num_freq_bins,
            'n_fft':conf_sr.n_fft,
            'frame_step':conf_sr.hop_length,}
        dereverb_reconstruction_test_kwargs = {'num_freq_bins':conf_sr.num_freq_bins,'n_fft':conf_sr.n_fft,'frame_step':conf_sr.hop_length,}
        dereverb_reconstruction_mdl = dereverb_reconstruction_test('dereverb_reconstruction_test', 
            **dereverb_reconstruction_test_kwargs)
        dereverb_reconstruction_mdl.summary()

try : ## Testing (wave to lps)
    print("## Predicting ...")
    ## Init
    splice_num=int(conf_DNN.context//2) # 11/2=5
    dereverb_uttid_to_inputs_test_kwargs={ 
        'reverb_LPS_mu':None,
        'reverb_LPS_std':None,
        'reverb_LPS_std_inv':None,
            'clean_LPS_mu':None,
            'clean_LPS_std':None,
            'clean_LPS_std_inv':None,
                'zeros_s6':np.zeros((1,(splice_num+1)*conf_sr.hop_length)),
                'input_s11':conf_sr.n_fft + (2*splice_num*conf_sr.hop_length),
                    'num_freq_bins':conf_sr.num_freq_bins,
                    'n_fft':conf_sr.n_fft,
                    'hop_length':conf_sr.hop_length,
                    'context':conf_DNN.context
    }
    ## Iterate
    for curr_uttid in Clean_WAVE_raw_dict:
        ## Read waves from dict
        clean_WAVE_raw=Clean_WAVE_raw_dict[curr_uttid]
        ## Read degrade from dict
        # nse_uttid=random.choice(NSE_WAVE_key)
        # noise_WAVE=NSE_WAVE_dict[nse_uttid]
        # _snr{des_snr}
        for des_rir in random.sample(RIR_WAVE_key,4):
            RIR_WAVE=RIR_WAVE_dict[des_rir]
            ## Read waves from dict
            clean_wav, reverb_wav, _=clean_to_noisyreverb(
                clean_WAVE_raw, None, RIR_WAVE, 
                None, conf_sr.desired_power)
            reverb_inputs = dereverb_uttid_to_inputs_test(reverb_wav,**dereverb_uttid_to_inputs_test_kwargs)
            # 
            # Dereverbd Abs
            DereverbdAbs_s1,NoisyAbs_s11 = dereverb_model_e2e_inf.predict(reverb_inputs, batch_size=conf_DNN.batch_size) # (time, 1, freq)
            DereverbdAbs_permute = np.transpose(DereverbdAbs_s1,(1,0,2))
            NoisyAbs_permute = np.transpose(NoisyAbs_s11,(1,0,2))
            # Reconstructed wave
            reverb_inf = add_frames(reverb_wav, conf_sr.hop_length)
            y_hat_tmp=dereverb_reconstruction_mdl.predict([reverb_inf,DereverbdAbs_permute])
            y_hat_matched = y_hat_tmp[0,conf_sr.hop_length:conf_sr.hop_length+len(reverb_wav)]
            # Clean Abs
            CleanAbs_permute, x_stft_pha, x_stft, x_LPS = wav2LPS_v2(clean_wav, **conf_sr.kwargs_STFT)
            # 
            ## Writing to file 
            wavbn=f'{curr_uttid}_rir{des_rir}.wav'
            enh_wavpath = os.path.join(conf_DNN.expinf_wav_dir,wavbn)
            y_hat_matched[y_hat_matched>1]=1.
            y_hat_matched[y_hat_matched<-1]=-1.
            # 
            reverb_wavbn=f'{curr_uttid}_rir{des_rir}_reverb.wav'
            reverb_wavpath = os.path.join(conf_DNN.expinf_wav_dir,reverb_wavbn)
            write_audio(reverb_wavpath,reverb_wav,conf_sr.sr,mode="soundfile")
            # 
            write_audio(enh_wavpath,y_hat_matched,conf_sr.sr,mode="soundfile")
            print('Saved to :',enh_wavpath)
            if 1 : ## Debugger mode
                min_num=min(np.min( DereverbdAbs_permute[0] ),
                            np.min( NoisyAbs_permute[0] ),
                            np.min( CleanAbs_permute[0] ),)
                ## Rescale to see if after minusing them will be better
                # reverb_wav_renorm=reverb_wav/np.max(np.abs(reverb_wav))
                # y_hat_matched_renorm=y_hat_matched/np.max(np.abs(y_hat_matched))
                ## Plot Tested result
                print('\ncurr_uttid           :',curr_uttid)
                k=3;col=2;l=1; curr_fig=plt.figure(figsize=(6*col,3*k)); 
                kwargs_plot={'colour_to_set':'black','hop_length':128,'sr':8000,'curr_fig':curr_fig,}
                display_audio(clean_wav, 'clean_wav',       'wave', kcoll=[k,col,l], **kwargs_plot); l+=1
                display_audio(x_LPS , 'CleanAbs_permute' ,'spec', kcoll=[k,col,l], **kwargs_plot); l+=1
                display_audio(reverb_wav, 'reverb_wav',       'wave', kcoll=[k,col,l], **kwargs_plot); l+=1
                display_audio(NoisyAbs_permute[5].T-min_num , 'NoisyAbs_permute' ,'spec', kcoll=[k,col,l], **kwargs_plot); l+=1
                display_audio(y_hat_matched, 'y_hat_matched',       'wave', kcoll=[k,col,l], **kwargs_plot); l+=1
                display_audio(DereverbdAbs_permute[0].T-min_num , 'DereverbdAbs_permute' ,'spec', kcoll=[k,col,l], **kwargs_plot); l+=1
                plt.tight_layout(); 
                plt.savefig(f'plot_path_{des_rir}.png')
            # 
        pdb.set_trace()
except:
    traceback.print_exc()
    pdb.set_trace()

"""
!import code; code.interact(local=vars())
"""
