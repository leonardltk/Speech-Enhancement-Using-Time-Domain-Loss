import os,sys
sys.path.insert(0, os.path.join('utils'))
from _helper_basics_ import *
from _helper_DNN_ import *
import conf

if True : ## Step 1 : Data
    def write_hash2file_dict(file_in, det_LIST):
        f_hash2file=open( file_in,'w')
        file2hash_DICT={}
        hash2file_DICT={}
        for wav_det in det_LIST:
            ## Break up the lines to uttid, path
            det_uttid = wav_det.replace('\n','').split(' ')[0]
            det_raw_wavepath = wav_det.replace('\n','').split(' ')[1:]
            det_raw_wavepath = ' '.join(det_raw_wavepath)

            ## Hash them
            det_hash=hash(det_uttid)
            hsh_det_c='{} {}\n'.format(det_hash,det_uttid)

            ## Write them, Save to dict
            f_hash2file.write(hsh_det_c)
            hash2file_DICT[det_hash]=det_uttid
            file2hash_DICT[det_uttid]=det_hash
        f_hash2file.close()
        print('    Finished writing to',file_in)
        return hash2file_DICT, file2hash_DICT
    def count_duration(wave_in,mode, sr):
        dur = len(wave_in)/sr
        if mode=='sec':     return dur
        elif mode=='min':   return dur/60
        elif mode=='hrs':   return dur/3600
    def clean_to_noisyreverb(
        clean_WAVE_in=None, noise_WAVE_in=None, RIR_WAVE_in=None, 
        des_snr=None, des_power=None):
        # Power normalization
        clean_WAVE_out = power_norm(clean_WAVE_in, des_power, _prevent_clipping=True)
        ## Add reverb
        if RIR_WAVE_in is not None:
            Reverb_WAVE_out = augment_reverb_wave( clean_WAVE_out,RIR_WAVE_in)
        else:
            Reverb_WAVE_out = clean_WAVE_out
        ## Add noise
        if noise_WAVE_in is not None:
            NoisyReverb_WAVE_out = augment_noisy_wave( Reverb_WAVE_out, noise_WAVE_in, des_snr)
        else:
            NoisyReverb_WAVE_out = None
        ## return
        return clean_WAVE_out, Reverb_WAVE_out, NoisyReverb_WAVE_out

if True : ## Step 2 : Model - Denoise Model
    ## Build Model
    class s11_to_s1(Layer): ## Pick out the center frame to perform masking on
        def __init__(self, **kwargs):
            super(s11_to_s1, self).__init__(**kwargs)
        def call(self, inputs, training=False):
            curr_spliced_frame = tf.slice(inputs, 
                        begin=[0, 5*129],
                        size =[-1, 129])
            return curr_spliced_frame
    def build_denoise_model_v3(mdl_name, _regularizer, 
        input_shape, output_shape):
        kwangs_dense={}
        if 1 :
            kwangs_dense['kernel_initializer']     ='glorot_normal'
            kwangs_dense['bias_initializer']       ='zeros'
            kwangs_dense['kernel_regularizer']     =_regularizer
            kwangs_dense['bias_regularizer']       =_regularizer
        kwangs_mask={}
        if 1 :
            kwangs_mask['activation']             ='linear'
            kwangs_mask['kernel_initializer']     ='glorot_normal'
            kwangs_mask['bias_initializer']       ='zeros'
            kwangs_mask['kernel_regularizer']     =_regularizer
            kwangs_mask['bias_regularizer']       =_regularizer
        ## Input
        Inp_NoisyReverb_s11 = Input(shape=input_shape)
        NoisyReverb_s11=Flatten()(Inp_NoisyReverb_s11)
        l_out=Dropout(0.2)(NoisyReverb_s11)
        ## Denoising mask
        for _ in range(3):
            l_out = Dense(1024,**kwangs_dense)(l_out);
            l_out = BatchNormalization()(l_out)
            l_out = Activation('relu')(l_out)
            l_out = Dropout(0.2)(l_out)
        Denoising_Mask=Dense(output_shape,**kwangs_mask)(l_out)
        ## Applying denoising mask
        NoisyReverb_s1  = s11_to_s1()(NoisyReverb_s11)
        Out_Denoised = Add()([NoisyReverb_s1,Denoising_Mask])
        ## Initialising the model
        model_denoise = Model(Inp_NoisyReverb_s11, Out_Denoised)
        model_denoise._name=mdl_name
        model_denoise.summary()
        return model_denoise

    ## Link Model
    def denoise_model_wave2abs_wocmvn_v4(mdl_name, 
        denoise_model,context,
        num_freq_bins,n_fft,frame_step, 
        mdl_mode='training'):
        if 1: ## Noisy Input
            Inp_NoisyWave_s11 = Input(shape=(n_fft+10*frame_step,), name='NoisyWave_s11')
            Inp_CleanWave_s1 = Input(shape=(n_fft,), name='CleanWave_s1')
        if 1: ## Wave -> Abs -> cmvn
            ## Noisy
            NoisyLPS_s11 = wave2LPS_Layer(
                n_fft=n_fft, 
                frame_step=frame_step, 
                name='NoisyLPS_s11')(Inp_NoisyWave_s11)
            ## Clean
            CleanLPS_s1, CleanPHA_s1 = wave2LPSpha_Layer(
                n_fft=n_fft, 
                frame_step=frame_step, 
                name='CleanLPSPHA_s1')(Inp_CleanWave_s1)
        if 1: ## Denoising
            DenoisedLPS_s1 = denoise_model(NoisyLPS_s11)
            DenoisedLPS_s1 = Reshape(
                target_shape=(1,num_freq_bins), 
                name='DenoisedLPS_s1')(DenoisedLPS_s1)
        if 1: ## TDR loss here
            DenoisedWave_s1 = LPSphase2istft_nowin_Layer(
                n_fft=n_fft, 
                frame_step=frame_step, 
                name='DenoisedWave_s1')([DenoisedLPS_s1,CleanPHA_s1])
        if 1: ## LPS loss here
            Zero_layer = Subtract(
                name='Zero_layer')([DenoisedLPS_s1,CleanLPS_s1])
        # 
        if mdl_mode=='training':
            model_joint=Model(
                [Inp_NoisyWave_s11, Inp_CleanWave_s1],
                [Zero_layer, DenoisedWave_s1])
        elif mdl_mode=='inference':
            model_joint=Model(
                [Inp_NoisyWave_s11],
                [DenoisedLPS_s1,NoisyLPS_s11]
            )
        # 
        model_joint._name = mdl_name
        return model_joint

    ## Training 
    def generate_wavetoabs_DenoisingNN_v6(gen_mode,
        Clean_WAVE_dict, NSE_WAVE_dict, RIR_WAVE_dict,
            num_freq_bins, n_fft, hop_length, batch_size, context,
            snr_list, desired_power,
                noisy_MAG_mu=None,noisy_MAG_std=None,noisy_MAG_std_inv=None,
                reverb_MAG_mu=None,reverb_MAG_std=None,reverb_MAG_std_inv=None,
        ):
        """
            Input : Clean_WAVE_dict
                    NSE_WAVE_dict
                    RIR_WAVE_dict
            Return: noisy wave s11
                    reverb wave s1
        """
        if True : ## Init
            splice_num=int(context//2) # 11/2=5
            c_wave_idx=splice_num*hop_length
            # 
            input_s11=n_fft + (2*splice_num*hop_length)
            output_s1=n_fft
            noisy_WAVE_s11=np.empty( (batch_size, input_s11) )
            reverb_WAVE_s1=np.empty( (batch_size, output_s1) )
            output_ZERO_s1=np.zeros( (batch_size, 1, num_freq_bins) )
            # 
            Clean_WAVE_key = list(Clean_WAVE_dict)
            NSE_WAVE_key = list(NSE_WAVE_dict)
            RIR_WAVE_key = list(RIR_WAVE_dict)
            # 
            sess = tf.compat.v1.Session()
            with sess.as_default(): tf_hann=tf.signal.hann_window(256, dtype=tf.float32).eval()
        while True: ## Iterate
            bs_idx=0
            while bs_idx < batch_size:
                ## Init
                des_snr=random.choice(snr_list)
                ## Choose utt
                clean_uttid=random.choice(Clean_WAVE_key)
                nse_uttid=random.choice(NSE_WAVE_key)
                rir_uttid=random.choice(RIR_WAVE_key)
                ## Read waves from dict
                clean_WAVE_raw=Clean_WAVE_dict[clean_uttid]
                noise_WAVE=NSE_WAVE_dict[nse_uttid]
                RIR_WAVE=RIR_WAVE_dict[rir_uttid]
                ## Augmentation
                # _, reverb_wav, noisy_wav=clean_to_noisyreverb(clean_WAVE_raw, noise_WAVE, RIR_WAVE, des_snr, desired_power)
                _, reverb_wav, noisy_wav=clean_to_noisyreverb(
                    clean_WAVE_raw, noise_WAVE, RIR_WAVE, 
                    des_snr, desired_power)
                ## Choose a random starting point of the wave
                wave_idx=np.random.randint( input_s11 , len(noisy_wav)-input_s11 )
                ## Appending (s11 of input wave, s1 of output wave) to batch
                noisy_WAVE_s11[bs_idx,:]=noisy_wav[ wave_idx            : wave_idx+input_s11           ]
                reverb_WAVE_s1[bs_idx,:]=reverb_wav[wave_idx+c_wave_idx : wave_idx+c_wave_idx+output_s1]
                bs_idx += 1
            if   gen_mode=='predict':
                yield [noisy_WAVE_s11, reverb_WAVE_s1]
            elif gen_mode=='fit':
                yield {
                        'NoisyWave_s11':noisy_WAVE_s11,
                        'CleanWave_s1':reverb_WAVE_s1,
                      },  {
                        'DenoisedWave_s1':reverb_WAVE_s1*tf_hann,
                        'Zero_layer':output_ZERO_s1,
                      }
            else:
                raise Exception('Name Error : gen_mode = {predict,fit}')
    class SaveModel_denoise(Callback):
        def __init__(self, denoise_model_e2e, conf_DNN):
            self.denoise_model_e2e = denoise_model_e2e
            self.conf_DNN = conf_DNN
        def on_epoch_end(self, epoch, logs):
            print()
            if True : ## Save training model
                # print('\tSaving self.denoise_model_e2e to {}\n'.format(self.conf_DNN.Training_weights_path))
                save_model(self.denoise_model_e2e, 
                    model_path=self.conf_DNN.Training_model_path, 
                    weights_path=self.conf_DNN.Training_weights_path,
                    verbose=False)
            if True : ## After finish training, extract submodel and save it
                # print('\tSaving model_denoise to {}\n'.format(self.conf_DNN.Enhance_weights_path))
                layer_name = self.conf_DNN.mdl_name
                model_denoise_trained = Model(
                    inputs=self.denoise_model_e2e.get_layer(layer_name).input, 
                    outputs=self.denoise_model_e2e.get_layer(layer_name).output)
                model_denoise_trained._name = layer_name
                # print('model_denoise_trained._name : ',model_denoise_trained._name)
                save_model(model_denoise_trained, 
                    model_path=self.conf_DNN.Enhance_model_path, 
                    weights_path=self.conf_DNN.Enhance_weights_path,
                    verbose=False)
                print()

    ## Testing
    def denoise_reconstruction_test_v2(mdl_name, 
        num_freq_bins,n_fft,frame_step):
        """ Generates a model for denoising, 
        creates LPS (instead of abs)
        followed by reconstruction to enhanced wave.
            Usage:
            -------
                denoise_reconstruction_test_kwargs = {'num_freq_bins':num_freq_bins,'n_fft':n_fft,'frame_step':hop_length,}
                denoise_reconstruction_mdl = denoise_reconstruction_test('denoise_reconstruction_test', **denoise_reconstruction_test_kwargs)
                denoise_reconstruction_mdl.summary()

            Inputs:
            -------
                mdl_name : The name of this model

                num_freq_bins : Used to specify the frequency bins for Enhanced spect.

                n_fft,frame_step : Used to compute the noisy phase, and reconstruction of enhanced wave.

            Returns: 
            -------
                model_reconstruction : Model that does end to end denoising.
        """
        if 1: ## Input
            Inp_DegradedWave   = Input(shape=(None,),       name='Inp_DegradedWave')
            Inp_EnhLPS         = Input(shape=(None,num_freq_bins),  name='Inp_EnhLPS')
        _,phase_out = wave2abspha_Layer(
            n_fft=n_fft, 
            frame_step=frame_step,name='phase_out')(Inp_DegradedWave)
        enh_wave = LPSphase2istft_Layer(
            n_fft=n_fft, 
            frame_step=frame_step,name='enh_wave')([Inp_EnhLPS,phase_out])
        model_reconstruction=Model(
            [Inp_DegradedWave, Inp_EnhLPS],
            enh_wave
            )
        model_reconstruction._name=mdl_name
        return model_reconstruction
    def denoise_uttid_to_inputs_test_v3(noisy_wav, 
        zeros_s6,input_s11,
        num_freq_bins, n_fft, hop_length, context,
            noisy_MAG_mu=None,noisy_MAG_std=None,
            reverb_MAG_mu=None,reverb_MAG_std=None,
        ):
        """ Converts noisy_wave into a list of noisy_inputs for the denoising network.
            Usage:
            -------
                splice_num=int(context//2)                                  # 11/2=5
                denoise_uttid_to_inputs_test_kwargs={ 
                    'noisy_MAG_mu':xxx_MAG_mu,                              # shape=(1, 129)
                    'noisy_MAG_std':xxx_MAG_std,                            # shape=(1, 129)
                    'noisy_MAG_std_inv':xxx_MAG_std,                        # shape=(1, 129)
                        'zeros_s6':np.zeros((1,(splice_num+1)*hop_length)), # shape=(1, 768)
                        'input_s11':n_fft + (2*splice_num*hop_length),      # 1536
                            'num_freq_bins':num_freq_bins,
                            'n_fft':n_fft,
                            'hop_length':hop_length,
                            'context':context}
                for curr_uttid in NoisyReverb_WAVE_dict:
                    ## Read waves from dict
                    noisy_wav=NoisyReverb_WAVE_dict[curr_uttid]
                    noisy_inputs = denoise_uttid_to_inputs_test(noisy_wav,**denoise_uttid_to_inputs_test_kwargs)
            
            Inputs:
            -------
                noisy_wave : 
                    original loaded audio
                    shape=(num_of_samples,)

                noisy_MAG_mu , noisy_MAG_std , noisy_MAG_std_inv : 
                    Precalculated cmvn stats, each with shape=(1,129)

                zeros_s6 : 
                    6 frames of zeros, appended to both ends of noisy_wav.
                    So that we can do denoising on denoising (s11->s1) without 
                    loss of the boundary frames.

                input_s11 : int
                    Number of samples required to generate s11 of spectrogram.

                ** kwargs : Typical parameters to do stft.
            
            Returns: 
            -------
                noisy_inputs : A list of noisy_inputs
                    noisy_inputs[0] : spliced audio
                        noisy_inputs[0].shape=(None, 1536)    # Frames of audio, s.t. aft stft will have 11 frames, so s11 of audio
                    noisy_inputs[1:] : Padded cmvn stats, depending on the number of frames this audio have
                        noisy_inputs[1].shape=(None, 1, 129)  # mu
                        noisy_inputs[2].shape=(None, 1, 129)  # std
                        noisy_inputs[3].shape=(None, 1, 129)  # std_inv

                    where None refers to the total number of frames for the whole spectrogram.
        """
        ## Pad the wave front and back
        noisy_wav=np.expand_dims(noisy_wav, axis=0)
        num_frames=int((noisy_wav.shape[1]-n_fft)/hop_length)+1
        noisy_wav=np.append(zeros_s6,noisy_wav, axis=1) # +5+1 frames
        noisy_wav=np.append(noisy_wav,zeros_s6, axis=1) # +5+1 frames
        ## Count how many frames, and initialise it
        noisy_WAVE_s11=np.empty( (num_frames+2, input_s11)     )
        ## Get s11 of input wave, s1 of output wave
        for curr_frame in range(num_frames):
            wave_idx = curr_frame*hop_length
            ## Appending to batch
            noisy_WAVE_s11[curr_frame,:]=noisy_wav[0,wave_idx : wave_idx+input_s11]
        # pdb.set_trace()
        return {'NoisyWave_s11':noisy_WAVE_s11,
                }

if True : ## Step 2 : Model - Dereverb Model
    ## Build Model
    def build_dereverb_model_v1(mdl_name, _regularizer, input_shape, output_shape):
        kwangs_dense={}
        if 1 :
            kwangs_dense['kernel_initializer']     ='glorot_normal'
            kwangs_dense['bias_initializer']       ='zeros'
            kwangs_dense['kernel_regularizer']     =_regularizer
            kwangs_dense['bias_regularizer']       =_regularizer
        kwangs_output={}
        if 1 :
            kwangs_output['activation']             ='linear'
            kwangs_output['kernel_initializer']     ='glorot_normal'
            kwangs_output['bias_initializer']       ='zeros'
            kwangs_output['kernel_regularizer']     =_regularizer
            kwangs_output['bias_regularizer']       =_regularizer
        ## Input
        Inp_Reverb_s11  = Input(shape=input_shape ); 
        l_out=Flatten()(Inp_Reverb_s11)
        l_out=Dropout(0.2)(l_out)
        ## Dereverb
        for _ in range(3):
            l_out = Dense(1024,**kwangs_dense)(l_out);
            l_out = BatchNormalization()(l_out)
            l_out = Activation('relu')(l_out)
            l_out = Dropout(0.2)(l_out)
        ## Estimating the output
        Out_Dereverb_LPS=Dense(output_shape,**kwangs_output)(l_out,)
        ## Initialising the model
        model_dereverb = Model(Inp_Reverb_s11, Out_Dereverb_LPS)
        model_dereverb._name=mdl_name
        model_dereverb.summary()
        return model_dereverb

    ## Link Model
    def dereverb_model_wave2abs_wocmvn_v4(mdl_name, 
        dereverb_model,context,
        num_freq_bins,n_fft,frame_step, 
        mdl_mode='training'):
        if 1: ## Reverb Input
            Inp_ReverbWave_s11 = Input(shape=(n_fft+10*frame_step,), name='ReverbWave_s11')
            Inp_CleanWave_s1 = Input(shape=(n_fft,), name='CleanWave_s1')
        if 1: ## Wave -> Abs -> cmvn
            ## Reverb
            ReverbLPS_s11 = wave2LPS_Layer(
                n_fft=n_fft, 
                frame_step=frame_step, 
                name='ReverbLPS_s11')(Inp_ReverbWave_s11)
            ## Clean
            CleanLPS_s1, CleanPHA_s1 = wave2LPSpha_Layer(
                n_fft=n_fft, 
                frame_step=frame_step, 
                name='CleanLPSPHA_s1')(Inp_CleanWave_s1)
        if 1: ## Denoising
            DereverbLPS_s1 = dereverb_model(ReverbLPS_s11)
            DereverbLPS_s1 = Reshape(
                target_shape=(1,num_freq_bins), 
                name='DereverbLPS_s1')(DereverbLPS_s1)
        if 1: ## TDR loss here
            DereverbWave_s1 = LPSphase2istft_nowin_Layer(
                n_fft=n_fft, 
                frame_step=frame_step, 
                name='DereverbWave_s1')([DereverbLPS_s1,CleanPHA_s1])
        if 1: ## LPS loss here
            Zero_layer = Subtract(
                name='Zero_layer')([DereverbLPS_s1,CleanLPS_s1])
        # 
        if mdl_mode=='training':
            model_joint=Model(
                [Inp_ReverbWave_s11, Inp_CleanWave_s1],
                [Zero_layer, DereverbWave_s1])
        elif mdl_mode=='inference':
            model_joint=Model(
                [Inp_ReverbWave_s11],
                [DereverbLPS_s1,ReverbLPS_s11]
            )
        # 
        model_joint._name = mdl_name
        return model_joint

    ## Training 
    def generate_wavetolps_DereverbNN_v4(gen_mode,
        Clean_WAVE_dict, RIR_WAVE_dict,
        reverb_LPS_mu,reverb_LPS_std,reverb_LPS_std_inv,
        clean_LPS_mu,clean_LPS_std,clean_LPS_std_inv,
            num_freq_bins, n_fft, hop_length, batch_size, context, desired_power):
        """
            Input : Clean_WAVE_dict
                    RIR_WAVE_dict
            Return: noisy wave s11
                    reverb wave s1
        """
        if True : ## Init
            splice_num=int(context//2) # 11/2=5
            c_wave_idx=splice_num*hop_length
            # 
            input_s11=n_fft + (2*splice_num*hop_length)
            output_s1=n_fft
            reverb_WAVE_s11=np.empty( (batch_size, input_s11) )
            clean_WAVE_s1=np.empty( (batch_size, output_s1) )
            output_ZERO_s1=np.zeros( (batch_size, 1, num_freq_bins) )
            # 
            Clean_WAVE_key = list(Clean_WAVE_dict)
            # NSE_WAVE_key = list(NSE_WAVE_dict)
            RIR_WAVE_key = list(RIR_WAVE_dict)
            # 
            sess = tf.compat.v1.Session()
            with sess.as_default(): tf_hann=tf.signal.hann_window(256, dtype=tf.float32).eval()
        ## Iterate
        while True:
            bs_idx=0
            while bs_idx < batch_size:
                ## Init
                # des_snr=random.choice(snr_list)
                ## Choose utt
                clean_uttid=random.choice(Clean_WAVE_key)
                # nse_uttid=random.choice(NSE_WAVE_key)
                rir_uttid=random.choice(RIR_WAVE_key)
                ## Read waves from dict
                clean_WAVE_raw=Clean_WAVE_dict[clean_uttid]
                # noise_WAVE=NSE_WAVE_dict[nse_uttid]
                RIR_WAVE=RIR_WAVE_dict[rir_uttid]
                ## Augmentation
                clean_wav, reverb_wav, _=clean_to_noisyreverb(
                    clean_WAVE_raw, None, RIR_WAVE, 
                    None, desired_power)
                ## Choose a random starting point of the wave
                wave_idx=np.random.randint( input_s11 , len(reverb_wav)-input_s11 )
                ## Get s11 of input wave, s1 of output wave, Appending to batch
                reverb_WAVE_s11[bs_idx,:]=reverb_wav[ wave_idx            : wave_idx+input_s11           ]
                clean_WAVE_s1[bs_idx,:]=clean_wav[    wave_idx+c_wave_idx : wave_idx+c_wave_idx+output_s1]
                bs_idx += 1
            if   gen_mode=='predict':
                yield [reverb_WAVE_s11, clean_WAVE_s1]
            elif gen_mode=='fit':
                yield {
                        'ReverbWave_s11':reverb_WAVE_s11,
                        'CleanWave_s1':clean_WAVE_s1,
                    },  {
                        'DereverbWave_s1':clean_WAVE_s1*tf_hann,
                        'Zero_layer':output_ZERO_s1,
                      }
            else:
                raise Exception('Name Error : gen_mode = {predict,fit}')
    class SaveModel_dereverb(Callback):
        def __init__(self, dereverb_model_e2e, conf_DNN):
            self.dereverb_model_e2e = dereverb_model_e2e
            self.conf_DNN = conf_DNN
        def on_epoch_end(self, epoch, logs):
            print()
            if True : ## Save training model
                save_model(self.dereverb_model_e2e, 
                    model_path=self.conf_DNN.Training_model_path, 
                    weights_path=self.conf_DNN.Training_weights_path,
                    verbose=False)
            if True : ## After finish training, extract submodel and save it
                layer_name = self.conf_DNN.mdl_name
                model_dereverb_trained = Model(
                    inputs = self.dereverb_model_e2e.get_layer(layer_name).input, 
                    outputs = self.dereverb_model_e2e.get_layer(layer_name).output)
                model_dereverb_trained._name = layer_name
                save_model(model_dereverb_trained, 
                    model_path = self.conf_DNN.Enhance_model_path, 
                    weights_path = self.conf_DNN.Enhance_weights_path,
                    verbose=False)

    ## Testing
    dereverb_reconstruction_test_v2 = denoise_reconstruction_test_v2 # For now, they are the same
    def dereverb_uttid_to_inputs_test_v2(reverb_wav, 
        zeros_s6,input_s11,
        num_freq_bins, n_fft, hop_length, context,
            reverb_LPS_mu:None,
            reverb_LPS_std:None,
            reverb_LPS_std_inv:None,
            clean_LPS_mu:None,
            clean_LPS_std:None,
            clean_LPS_std_inv:None,
        ):
        ## Pad the wave front and back
        reverb_wav=np.expand_dims(reverb_wav, axis=0)
        num_frames=int((reverb_wav.shape[1]-n_fft)/hop_length)+1
        reverb_wav=np.append(zeros_s6,reverb_wav, axis=1) # +5+1 frames
        reverb_wav=np.append(reverb_wav,zeros_s6, axis=1) # +5+1 frames
        ## Count how many frames, and initialise it
        ReverbWAVE_s11=np.empty( (num_frames+2, input_s11)     )
        ## Get s11 of input wave, s1 of output wave
        for curr_frame in range(num_frames):
            wave_idx = curr_frame*hop_length
            ## Appending to batch
            ReverbWAVE_s11[curr_frame,:]=reverb_wav[0,wave_idx : wave_idx+input_s11]
        # pdb.set_trace()
        return {'ReverbWave_s11':ReverbWAVE_s11,}

if True : ## Step 3 : Model - Joint Model
    class splicing_s11(Layer):
        def __init__(self, 
            _context,
            _num_feats,
            **kwargs):
            super(splicing_s11, self).__init__(**kwargs)
            self._context = _context
            self._num_feats = _num_feats
        def get_config(self):
            config = super().get_config().copy()
            return config
        def call(self, inputs, training=True):
            spliced_frames_list=[] # Will be filled 
            ## Given a variable number of timesteps, t,
            #      splice 11 frames,
            # 
            # Since cannot deal with varying timesteps:
            # We just take in 21 input time step
            # To output 11 denoised timesteps
            # To output  1 dereverbed timestep
            for frame_start_idx in range(11):
                # 
                curr_spliced_frame = tf.slice(inputs, 
                            begin=[0, frame_start_idx, 0],
                            size =[-1, self._context, self._num_feats])
                # print('curr_spliced_frame.shape',curr_spliced_frame.shape)
                # curr_spliced_frame.shape (?, 11, 169)
                # 
                curr_spliced_frame=Reshape(target_shape=(self._context*self._num_feats,))(curr_spliced_frame)
                # print('curr_spliced_frame.shape',curr_spliced_frame.shape)
                # curr_spliced_frame.shape (?, 1859)
                spliced_frames_list.append(curr_spliced_frame)
            return spliced_frames_list

    ## Link Model
    def joint_model_wave2wave_v4(mdl_name, 
        dereverb_model, denoise_model,
        context, num_freq_bins, n_fft, frame_step, 
        mdl_mode='training'):
        if 1: ## Input
            Inp_ReverbNoisyWave_s21 = Input(shape=(n_fft+(context-1)*frame_step,), name='ReverbNoisyWave_s21')
            Inp_CleanWave_s1 = Input(shape=(n_fft+0*frame_step,), name='CleanWave_s1')
        if 1: ## Wave -> LPS
            ## ReverbNoisy
            ReverbNoisyLPS_s21 = wave2LPS_Layer(
                n_fft=n_fft, 
                frame_step=frame_step, 
                name='ReverbNoisyAbs_s11')(Inp_ReverbNoisyWave_s21)
            ## Clean
            CleanLPS_s1, CleanPHA_s1 = wave2LPSpha_Layer(
                n_fft=n_fft, 
                frame_step=frame_step, 
                name='CleanLPSPHA_s1')(Inp_CleanWave_s1)
            ## Perform Splicing, and obtain "Denoised_s11" from "ReverbNoisyLPS_s21"
            Splicing_list = splicing_s11(
                _context=context//2+1,
                _num_feats=num_freq_bins,
                name='Splicing_list',
                )(ReverbNoisyLPS_s21)
            denoised_lps_list=[]
            for inp_nsy in Splicing_list:
                denoised_lps_list.append( denoise_model(inp_nsy) )
            ## Reshape to s11
            DenoisedLPS_s11=Concatenate(axis=1, 
                name='concat')(denoised_lps_list)
            DenoisedLPS_s11=Reshape(
                target_shape=(11,num_freq_bins), 
                name='Denoised_s11')(DenoisedLPS_s11)
        if 1: ## Dereverb
            DereverbLPS_s1=dereverb_model(DenoisedLPS_s11)
            DereverbLPS_s1 = Reshape(
                target_shape=(1,num_freq_bins), 
                name='DereverbLPS_s1')(DereverbLPS_s1)
            if 1: ## TDR loss here
                # DereverbWave_s1 = LPSphase2istft_Layer(
                DereverbWave_s1 = LPSphase2istft_nowin_Layer(
                    n_fft=n_fft, 
                    frame_step=frame_step, 
                    name='DereverbWave_s1')([DereverbLPS_s1,CleanPHA_s1])
            if 1: ## LPS loss here
                Zero_layer = Subtract(
                    name='Zero_layer')([DereverbLPS_s1,CleanLPS_s1])
        if 1: ## Return
            if mdl_mode=='training':
                model_joint=Model(
                    [Inp_ReverbNoisyWave_s21,Inp_CleanWave_s1],
                    [Zero_layer, DereverbWave_s1])
            elif mdl_mode=='inference':
                model_joint=Model(
                    [Inp_ReverbNoisyWave_s21],
                    [DereverbLPS_s1,ReverbNoisyLPS_s21]
                )
            # 
        model_joint._name = mdl_name
        return model_joint

    ## Training 
    def generate_wavetoabs_JointNN_v3(gen_mode,
        Clean_WAVE_dict, NSE_WAVE_dict, RIR_WAVE_dict,
        num_freq_bins, n_fft, hop_length, batch_size, context,
        snr_list, desired_power,):
        """
            Input : Clean_WAVE_dict
                    NSE_WAVE_dict
                    RIR_WAVE_dict
            Return: ReverbNoisyWave_s21
                    CleanWave_s1
        """
        ## Init
        s21=context//2  # 21//2=10
        s11=s21//2      # 10//2=5
        c_wave_idx=s21*hop_length
        input_s21=n_fft + (2*s21*hop_length)
        input_s11=n_fft + (2*s11*hop_length)
        output_s1=n_fft
        c_idx_s21=s21*hop_length # This is the shift from noisy vs clean
        c_idx_s11=s11*hop_length # This is the shift from noisy vs reverb
        # 
        NoisyReverb_WAVE_s21 = np.empty( (batch_size, input_s21) ) # NoisyReverb_WAVE_s21
        Reverb_WAVE_s11      = np.empty( (batch_size, input_s11) ) # Reverb_WAVE_s11
        Clean_WAVE_s1        = np.empty( (batch_size, output_s1) )
        Output_ZERO_s1=np.zeros( (batch_size, 1, num_freq_bins) )
        # 
        Clean_WAVE_key = list(Clean_WAVE_dict)
        NSE_WAVE_key = list(NSE_WAVE_dict)
        RIR_WAVE_key = list(RIR_WAVE_dict)
        # 
        sess = tf.compat.v1.Session()
        with sess.as_default(): tf_hann=tf.signal.hann_window(256, dtype=tf.float32).eval()
        ## Iterate
        while True:
            bs_idx=0
            while bs_idx < batch_size:
                ## Init
                des_snr=random.choice(snr_list)
                ## Choose utt
                clean_uttid=random.choice(Clean_WAVE_key)
                nse_uttid=random.choice(NSE_WAVE_key)
                rir_uttid=random.choice(RIR_WAVE_key)
                ## Read waves from dict
                clean_WAVE_raw=Clean_WAVE_dict[clean_uttid]
                while len(clean_WAVE_raw)<input_s21: continue
                noise_WAVE=NSE_WAVE_dict[nse_uttid]
                RIR_WAVE=RIR_WAVE_dict[rir_uttid]
                ## Augmentation
                clean_wav, reverb_wav, noisy_wav=clean_to_noisyreverb(
                    clean_WAVE_raw, noise_WAVE, RIR_WAVE, 
                    des_snr, desired_power)
                ## Choose a random starting point of the wave
                wave_idx=np.random.randint( input_s21 , len(noisy_wav)-input_s21 )
                ## Appending to batch
                #  Get s21 of noisy wave
                #  Get s11 of reverb wave
                NoisyReverb_WAVE_s21[bs_idx,:] =noisy_wav[ wave_idx           : wave_idx+input_s21          ]
                Reverb_WAVE_s11[bs_idx,:]     =reverb_wav[wave_idx+c_idx_s11 : wave_idx+c_idx_s11+input_s11]
                Clean_WAVE_s1[bs_idx,:]       = clean_wav[wave_idx+c_idx_s21 : wave_idx+c_idx_s21+output_s1]
                bs_idx += 1
            if   gen_mode=='predict':
                yield [NoisyReverb_WAVE_s21, Clean_WAVE_s1]
            elif gen_mode=='fit':
                yield {
                        'ReverbNoisyWave_s21':NoisyReverb_WAVE_s21,
                        'CleanWave_s1':Clean_WAVE_s1,
                      },  {
                        'DereverbWave_s1':Clean_WAVE_s1*tf_hann,
                        'Zero_layer':Output_ZERO_s1,
                      }
            else:
                raise Exception('Name Error : gen_mode = {predict,fit}')
    class SaveModel_joint(Callback):
        def __init__(self, joint_model_e2e, conf_DNN, conf_DNN_dereverb, conf_DNN_denoise):
            self.joint_model_e2e = joint_model_e2e
            self.conf_DNN = conf_DNN
            self.conf_DNN_dereverb = conf_DNN_dereverb
            self.conf_DNN_denoise = conf_DNN_denoise
        def on_epoch_end(self, epoch, logs):
            print()
            if True : ## Save training model
                save_model(self.joint_model_e2e, 
                    model_path=self.conf_DNN.Training_model_path, 
                    weights_path=self.conf_DNN.Training_weights_path,
                    verbose=False)
            if True : ## Save dereverb submodel
                layername = self.conf_DNN_dereverb.mdl_name+'_dereverb'
                model_joint_dereverb = Model(
                    inputs = self.joint_model_e2e.get_layer(layername).input,
                    outputs = self.joint_model_e2e.get_layer(layername).output,
                    )
                model_joint_dereverb._name = layername
                save_model(model_joint_dereverb, 
                    model_path   = self.conf_DNN.Enhance_model_path.replace(   '.json','_dereverb.json'),
                    weights_path = self.conf_DNN.Enhance_weights_path.replace( '.h5','_dereverb.h5'),
                    verbose=False)
            if True : ## Save denoise submodel
                layername = self.conf_DNN_denoise.mdl_name+'_denoise'
                model_joint_denoise = Model(
                    inputs = self.joint_model_e2e.get_layer(layername).input,
                    outputs = self.joint_model_e2e.get_layer(layername).output,
                    )
                model_joint_denoise._name = layername
                save_model(model_joint_denoise, 
                    model_path   = self.conf_DNN.Enhance_model_path.replace(   '.json','_denoise.json'),
                    weights_path = self.conf_DNN.Enhance_weights_path.replace( '.h5','_denoise.h5'),
                    verbose=False)

    ## Testing
    def joint_uttid_to_inputs_test(): 1

def versioning(vrs):
    if vrs == 'v1_TDR': # _beta
        return {
        'build_denoise_model': build_denoise_model_v3,
            'denoise_model_wave2abs': denoise_model_wave2abs_wocmvn_v4,
                'generate_wavetoabs_DenoisingNN':generate_wavetoabs_DenoisingNN_v6,
                    'denoise_reconstruction_test': denoise_reconstruction_test_v2,
                    'denoise_uttid_to_inputs_test': denoise_uttid_to_inputs_test_v3,
        'build_dereverb_model': build_dereverb_model_v1,
            'dereverb_model_wave2abs': dereverb_model_wave2abs_wocmvn_v4,
                'generate_wavetolps_DereverbNN': generate_wavetolps_DereverbNN_v4,
                    'dereverb_reconstruction_test': dereverb_reconstruction_test_v2,
                    'dereverb_uttid_to_inputs_test': dereverb_uttid_to_inputs_test_v2,
        'build_joint_model': None,
            'joint_model_wave2wave': joint_model_wave2wave_v4,
                'generate_wavetolps_JointNN': generate_wavetoabs_JointNN_v3,
                    'joint_reconstruction_test': None,
                    'joint_uttid_to_inputs_test': None,
            }
    else:
        print('Not in the selection')

##############################################
