from __future__ import print_function
import os
import pprint
pp = pprint.PrettyPrinter()

class Data_conf():
    def __init__(self, datatype):
        self.datatype=datatype
        ############ Data Dir/Path ############
        self.data_dir=os.path.join('./data',self.datatype); os.makedirs( self.data_dir ,exist_ok=True)
        self.wavscp=os.path.join(self.data_dir,'wav.scp')
        self.wav_clean_scp=os.path.join(self.data_dir,'wav_clean.scp')
        self.hash2clean=os.path.join(self.data_dir,'hash2clean')
        ############ Exp Dir/Path ############
        self.expfeats_dir=os.path.join('./exp_feats', self.datatype); os.makedirs( self.expfeats_dir ,exist_ok=True)
        self.exp_dict_dir=os.path.join(self.expfeats_dir, 'dict'); os.makedirs( self.exp_dict_dir ,exist_ok=True)
        if 1: ## dict
            self.Clean_WAVE_raw_dict_path=os.path.join(self.exp_dict_dir,'Clean_WAVE_raw.dict')
            self.NSE_WAVE_dict_path=os.path.join(self.exp_dict_dir,'NSE_WAVE.dict')
            self.RIR_WAVE_dict_path=os.path.join(self.exp_dict_dir,'RIR_WAVE.dict')

            self.Clean_WAVE_dict_path=os.path.join(self.exp_dict_dir,'Clean_WAVE.dict')
            self.Reverb_WAVE_dict_path=os.path.join(self.exp_dict_dir,'Reverb_WAVE.dict')
            self.NoisyReverb_WAVE_dict_path=os.path.join(self.exp_dict_dir,'NoisyReverb_WAVE.dict')
        ############ Wav Dir/Path ############
        self.wav_dir=os.path.join('./wav', self.datatype); os.makedirs( self.wav_dir ,exist_ok=True)
        ############ Results Dir/Path ############
        self.results_dir=os.path.join('./results', self.datatype); os.makedirs( self.results_dir ,exist_ok=True)

    def __str__(self):
        print(f'\n############ Data_conf ############')
        print(f'\t\t datatype               : {self.datatype}')
        print(f'\t############ Data Dir/Path ############')
        print(f'\t\t data_dir               : {self.data_dir}')
        print(f'\t\t wavscp                 : {self.wavscp}')
        print(f'\t\t wav_clean_scp          : {self.wav_clean_scp}')
        print(f'\t\t hash2clean             : {self.hash2clean}')
        print(f'\t############ Exp Dir/Path ############')
        print(f'\t\t expfeats_dir                   : {self.expfeats_dir}')
        print(f'\t\t   exp_dict_dir                 : {self.exp_dict_dir}')
        print(f'\t\t     Clean_WAVE_raw_dict_path   : {self.Clean_WAVE_raw_dict_path}')
        print(f'\t\t     NSE_WAVE_dict_path         : {self.NSE_WAVE_dict_path}')
        print(f'\t\t     RIR_WAVE_dict_path         : {self.RIR_WAVE_dict_path}')
        print(f'\t\t     Clean_WAVE_dict_path       : {self.Clean_WAVE_dict_path}')
        print(f'\t\t     Reverb_WAVE_dict_path      : {self.Reverb_WAVE_dict_path}')
        print(f'\t\t     NoisyReverb_WAVE_dict_path : {self.NoisyReverb_WAVE_dict_path}')
        print(f'\t############ Wav Dir/Path ############')
        print(f'\t\t wav_dir                : {self.wav_dir}')
        print(f'\t############ Results Dir/Path ############')
        print(f'\t\t results_dir            : {self.results_dir}')
        return ""


class Degrade_conf():
    def __init__(self, datatype):
        self.datatype=datatype
        ############ Data Dir/Path ############
        self.data_dir=os.path.join('./data',self.datatype); os.makedirs( self.data_dir ,exist_ok=True)
        self.wav_nse_scp=os.path.join(self.data_dir,'wav_nse.scp')
        self.wav_RIR_scp=os.path.join(self.data_dir,'wav_RIR.scp')
        self.hash2rir=os.path.join(self.data_dir,'hash2rir')
        self.hash2nse=os.path.join(self.data_dir,'hash2nse')
        ############ Exp Dir/Path ############
        self.expfeats_dir=os.path.join('./exp_feats', self.datatype); os.makedirs( self.expfeats_dir ,exist_ok=True)
        self.NSE_WAVE_dict_path=os.path.join(self.expfeats_dir, 'NSE_WAVE_dict')
        self.RIR_WAVE_dict_path=os.path.join(self.expfeats_dir, 'RIR_WAVE_dict')
        ############ Parameters ############
        self.snr_list=[0,5,10,30]

        # self.hrs_list=[0.5/3,0.5/3,0.5/3,0.5]
        self.hrs_list=[0.005/3,0.005/3,0.005/3,0.005]

    def __str__(self):
        print(f'\n############ Data_conf ############')
        print(f'\t\t datatype               : {self.datatype}')
        print(f'\t############ Data Dir/Path ############')
        print(f'\t\t data_dir               : {self.data_dir}')
        print(f'\t\t wav_nse_scp            : {self.wav_nse_scp}')
        print(f'\t\t hash2rir               : {self.hash2rir}')
        print(f'\t\t hash2nse               : {self.hash2nse}')
        print(f'\t############ Data Dir/Path ############')
        print(f'\t\t snr_list               : {self.snr_list}')
        print(f'\t\t hrs_list               : {self.hrs_list}')
        return ""


class SR_conf():
    def __init__(self):
        ############ Audio Wave Parameters ############
        if 1:
            self.sr=8000
            self.n_fft=256
            self.win_length=256
            self.hop_length=self.n_fft//2
            self.winlen=1.*self.win_length/self.sr
            self.winstep=1.*self.hop_length/self.sr
            self.num_freq_bins=self.n_fft//2 +1

            self.desired_power=6.392e-05
            self.des_len = 81536
        ############ Audio Feat Parameters ############
        if 1:
            self.kwargs_STFT={
                'pad_mode':True,
                'mode':'librosa',
                'n_fft':self.n_fft,
                'win_length':self.win_length,
                'hop_length':self.hop_length,
                'nfft':self.n_fft,
                'winstep':self.winstep,
                'winlen':self.winlen,
                'fs':self.sr,
            }
            self.kwargs_MFCC={
                'mode':"librosa",
                'sr_in':self.sr,
                'pad_mode':True,
                'preemph':0.97,
                'ceplifter':22,
                'n_mels':32,
            }
            self.kwargs_GF={}
            if 1 :
                self.kwargs_GF['GF_bins']=64

    def __str__(self):
        print('\n############ SR_conf ############')
        print('\t############ Audio Wave Parameters ############')
        print(f'\t\t sr                 :{self.sr}')
        print(f'\t\t n_fft              :{self.n_fft}')
        print(f'\t\t win_length         :{self.win_length}')
        print(f'\t\t hop_length         :{self.hop_length}')
        print(f'\t\t winlen             :{self.winlen}')
        print(f'\t\t winstep            :{self.winstep}')
        print(f'\t\t num_freq_bins      :{self.num_freq_bins}')

        print(f'\t\t desired_power      :{self.desired_power}')
        print(f'\t\t des_len            :{self.des_len}')
        print('\t############ Audio Feat Parameters ############')
        print(f'\t\t kwargs_STFT        :{kwargs_STFT}')
        print(f'\t\t kwargs_MFCC        :{kwargs_MFCC}')
        print(f'\t\t kwargs_GF        :{kwargs_GF}')
        return ""


class DNN_conf():
    def __init__(self, Enh_mode='denoise', Archi_vrs="v1_TDR", ):
        self.Enh_mode=Enh_mode
        self.Archi_vrs=Archi_vrs
        ############ Dir ############
        if True:
            self.Archi_dir = os.path.join('model', self.Enh_mode, self.Archi_vrs)
            self.Weights_path            = os.path.join(self.Archi_dir, "Logs");                       os.makedirs(self.Weights_path,exist_ok=True)
            self.Ckpt_Mod_Weights_fold   = os.path.join(self.Archi_dir, "Checkpoint_Model_Weights");   os.makedirs(self.Ckpt_Mod_Weights_fold,exist_ok=True)
            # self.LR_gridsearch           = os.path.join(self.Archi_dir, "LR_gridsearch")
            # if 1:
            #     self.LR_Log                  = os.path.join(self.LR_gridsearch, "Log");                os.makedirs(self.LR_Log,exist_ok=True)
            #     self.LR_Plot                 = os.path.join(self.LR_gridsearch, "Plot");               os.makedirs(self.LR_Plot,exist_ok=True)
            #     self.LR_History              = os.path.join(self.LR_gridsearch, "History");            os.makedirs(self.LR_History,exist_ok=True)
            self.Final_Weights_fold      = os.path.join(self.Archi_dir, "Final_Model_Weights");        os.makedirs(self.Final_Weights_fold,exist_ok=True)
            if True:
                # Enhance Model
                self.Enhance_model_path   = os.path.join(self.Final_Weights_fold, self.Enh_mode+'.json')
                self.Enhance_weights_path = os.path.join(self.Final_Weights_fold, self.Enh_mode+'.h5')
                # Training Model
                self.Training_mode=self.Enh_mode+'_Train'
                self.Training_model_path   = os.path.join(self.Final_Weights_fold, self.Training_mode+'.json')
                self.Training_weights_path = os.path.join(self.Final_Weights_fold, self.Training_mode+'.h5')
            self.Plot_path_dir           = os.path.join(self.Archi_dir,'Plots');                      os.makedirs(self.Plot_path_dir,exist_ok=True)
            self.Plot_path = os.path.join(self.Plot_path_dir,'{}.png'.format(Archi_vrs))
        ############ Model Parameters ############
        if True: ## Model input/output shapes
            self.context=11
            self.pred_frame=5 # of the context
            self.input_shape=( self.context*(129), )
            self.output_shape=129
            self.mdl_name=f"{self.Archi_vrs}_{self.Enh_mode}"
        if True: ## Regularization
            self.reg_mode="l1_l2"
            self.reg_l1=1e-5
            self.reg_l2=1e-5
            self.kwargs_reg={}
            self.kwargs_reg['reg_l1']=self.reg_l1
            self.kwargs_reg['reg_l2']=self.reg_l2
        ######################## Hyper Parameters ###############################
        if True:
            self.initial_epoch = 1-1;
            # self.epochs = 10;
            self.epochs = 100;
            self.batch_size = 32;
            # self.steps_per_epoch = 10;
            self.steps_per_epoch = 1000;
            self.validation_steps = 0;
        ######################## Optimizer, Metrics Parameters ###############################
        if True: ## Optimizer
            self.opt_mode='Adam'
            self.opt_dict={'lr':1e-3}
            if self.opt_mode=='SGD':
                self.opt_dict['momentum']=.99
                self.opt_dict['decay']=0
            elif self.opt_mode=='Adam':
                self.opt_dict['decay']=0
                self.opt_dict['beta_1']=0.9
                self.opt_dict['beta_2']=0.999
                self.opt_dict['amsgrad']=False
        if True: ## Loss, Metrics,
            self.compile_dict = {}
            self.loss_type='huber_loss' # 'mse' 'mae'
            self.compile_dict['loss'] = self.loss_type
            self.metrics=None # 'categorical_accuracy' 'acc'
            self.compile_dict['metrics'] = self.metrics
            #
            ## We give TDR more weight as loss in waveform is small
            self.compile_dict['loss_weights']=[1,int(1e5)]
            #
            # self.compile_dict['sample_weight_mode']=None
            # self.compile_dict['weighted_metrics']=None
            # self.compile_dict['target_tensors']=None
        ######################## Callbacks ###############################
        if True: 
            self.Callbacks_list=['ReduceLROnPlateau','ModelCheckpoint','CSVLogger']
            
            ## ReduceLROnPlateau
            self.ReduceLROnPlateau_kwargs={
                "monitor":'loss',
                "factor":0.5,
                "patience":5,
                "verbose":1,
                "mode":'auto',
                "min_delta":1e-6,
                "cooldown":3,
                "min_lr":1e-6,
                }
            
            ## ModelCheckpoint
            self.CkptFold_det = [self.Archi_vrs, self.Ckpt_Mod_Weights_fold]
            self.ModelCheckpoint_kwargs={
                "monitor" : 'loss', 
                "verbose" : 1, 
                "save_best_only":True, 
                "save_weights_only":False,
                "mode":'auto',
                }
            # "period" : 1 ## (deprecated) how many epochs.
            """ "save_freq" : 1  ## This refers to how many samples seen, not how many epochs. But it is still very glitchy because of this
                Epoch 00001: loss improved from inf to 428.80817, saving model to ./model/AlexNet/Checkpoint_Model_Weights/01-428.81.hdf5
                 1/20 [>.............................] - ETA: 9s - loss: 428.8082
                 2/20 [==>...........................] - ETA: 8s - loss: 408.6893del to ./model/AlexNet/Checkpoint_Model_Weights/01-388.57.hdf5
                 3/20 [===>..........................] - ETA: 8s - loss: 379.0225del to ./model/AlexNet/Checkpoint_Model_Weights/01-319.69.hdf5
                 4/20 [=====>........................] - ETA: 7s - loss: 367.5825
                 5/20 [======>.......................] - ETA: 7s - loss: 361.3616
                 6/20 [========>.....................] - ETA: 6s - loss: 347.2880del to ./model/AlexNet/Checkpoint_Model_Weights/01-276.92.hdf5
                 7/20 [=========>....................] - ETA: 6s - loss: 339.0638
                 8/20 [===========>..................] - ETA: 5s - loss: 335.5200
                 9/20 [============>.................] - ETA: 5s - loss: 332.0451
                10/20 [==============>...............] - ETA: 4s - loss: 328.5721
                11/20 [===============>..............] - ETA: 4s - loss: 328.9381
                12/20 [=================>............] - ETA: 3s - loss: 327.5210
                13/20 [==================>...........] - ETA: 3s - loss: 327.7996
                14/20 [====================>.........] - ETA: 2s - loss: 328.1067
                15/20 [=====================>........] - ETA: 2s - loss: 326.4287
                16/20 [=======================>......] - ETA: 1s - loss: 324.9698
                17/20 [========================>.....] - ETA: 1s - loss: 323.9119
                18/20 [==========================>...] - ETA: 0s - loss: 321.6416
                19/20 [===========================>..] - ETA: 0s - loss: 322.6315
                20/20 [==============================] - 10s 483ms/step - loss: 321.5176
            """

            ## CSVLogger
            self.csv_log_path = os.path.join(self.Weights_path,'{}_Trglog.txt'.format(self.Archi_vrs))
            
            ## Save_Live_Plot
        ######################## Inference Details ###############################
        if True: 
            self.expinf_dir = os.path.join('exp_inf', self.Enh_mode)
            self.expinf_wav_dir = os.path.join(self.expinf_dir, self.Archi_vrs); os.makedirs(self.expinf_wav_dir,exist_ok=True)
        ######################## Results Details ###############################
        if True: 
            self.results_dir = os.path.join('results',self.Enh_mode)
            self.results_vrs_dir = os.path.join(self.results_dir,self.Archi_vrs); os.makedirs(self.results_vrs_dir,exist_ok=True)

    def __str__(self):
        print(f'\n############ {self.Enh_mode}_{self.Archi_vrs}_conf ############')
        print('\t############ Dir ############')
        if True:
            print(f'\t\t Archi_dir                  :{self.Archi_dir}')
            print(f'\t\t Weights_path               :{self.Weights_path}')
            print(f'\t\t Ckpt_Mod_Weights_fold      :{self.Ckpt_Mod_Weights_fold}')
            print(f'\t\t Final_Weights_fold         :{self.Final_Weights_fold}')
            print(f'\t\t     Enhance_model_path     :{self.Enhance_model_path}')
            print(f'\t\t     Enhance_weights_path   :{self.Enhance_weights_path}')
            print(f'\t\t     Training_mode          :{self.Training_mode}')
            print(f'\t\t     Training_model_path    :{self.Training_model_path}')
            print(f'\t\t     Training_weights_path  :{self.Training_weights_path}')
            print(f'\t\t Plot_path_dir              :{self.Plot_path_dir}')
            print(f'\t\t     Plot_path              :{self.Plot_path}')
        print('\t############ Model Parameters ############')
        if True:
            print(f'\t\t context                    :{self.context}')
            print(f'\t\t pred_frame                 :{self.pred_frame}')
            print(f'\t\t input_shape                :{self.input_shape}')
            print(f'\t\t output_shape               :{self.output_shape}')
            print(f'\t\t mdl_name                   :{self.mdl_name}')
            print(f'\t\t reg_mode                   :{self.reg_mode}')
            print(f'\t\t kwargs_reg                 :{self.kwargs_reg}')
        print('\t############ Optimizer, Metrics Parameters ############')
        if True:
            print(f'\t\t opt_mode                   :{self.opt_mode}')
            print(f'\t\t opt_dict                   :{self.opt_dict}')
            print(f'\t\t compile_dict               :{self.compile_dict}')
        print('\t############ Callbacks ############')
        if True:
            print(f'\t\t Callbacks_list             :{self.Callbacks_list}')
            print(f'\t\t ReduceLROnPlateau_kwargs   :{self.ReduceLROnPlateau_kwargs}')
            print(f'\t\t CkptFold_det               :{self.CkptFold_det}')
            print(f'\t\t ModelCheckpoint_kwargs     :{self.ModelCheckpoint_kwargs}')
            print(f'\t\t csv_log_path               :{self.csv_log_path}')
        print('\t############ Inference Details ############')
        if True:
            print(f'\t\t expinf_dir                 :{self.expinf_dir}')
            print(f'\t\t expinf_wav_dir             :{self.expinf_wav_dir}')
        print('\t############ Results Details ############')
        if True:
            print(f'\t\t results_dir                :{self.results_dir}')
            print(f'\t\t results_vrs_dir            :{self.results_vrs_dir}')
        return ""
