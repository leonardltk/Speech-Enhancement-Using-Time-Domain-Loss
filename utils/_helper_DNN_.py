from __future__ import print_function # pdb.set_trace()
from _helper_basics_ import *
if True: ## imports
    import tensorflow as tf
    import tensorflow.keras

    ## Architecture
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import *

    ## Training
    from tensorflow.keras.losses import *
    from tensorflow.keras.metrics import *
    from tensorflow.keras.optimizers import *
    from tensorflow.keras.callbacks import *
    from tensorflow.keras.applications import *
    from tensorflow.keras.regularizers import *
    # ## Misc
    from tensorflow.keras.models import model_from_json
    from tensorflow.keras.utils import multi_gpu_model, to_categorical
    #   from keras.utils.np_utils import to_categorical
    #   keras.utils.multi_gpu_model(model, gpus=None, cpu_merge=True, cpu_relocation=False) -> tf.distribute.MirroredStrategy

    ## Keras Customization
    import tensorflow.keras.backend as K
    ## Keras PreProcessing
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

################################# Signal Processing Layers #################################
## wave to feat
class wave2LPS_Layer(Layer):
    def __init__(self, n_fft, frame_step, **kwargs):
        super(wave2LPS_Layer, self).__init__(**kwargs)
        self.n_fft = n_fft
        self.frame_step = frame_step
    def get_config(self):
        config = super().get_config().copy()
        return config
    def call(self, inputs, training=True):
        eps1e6=tf.constant(1e-6)
        curr_window=tf.signal.hann_window
        stfts = tf.signal.stft(
            inputs, frame_length=self.n_fft, frame_step=self.frame_step,
            window_fn=curr_window,
            pad_end=False)
        abs_out=tf.math.abs(stfts)
        lps_out = tf.math.log(abs_out + eps1e6)
        # lps_out = tf.math.log1p(abs_out)
        return lps_out
class wave2LPSpha_Layer(Layer):
    def __init__(self, n_fft, frame_step, **kwargs):
        super(wave2LPSpha_Layer, self).__init__(**kwargs)
        self.n_fft = n_fft
        self.frame_step = frame_step
    def get_config(self):
        config = super().get_config().copy()
        return config
    def call(self, inputs, training=True):
        eps1e6=tf.constant(1e-6)
        curr_window=tf.signal.hann_window
        stfts = tf.signal.stft(
            inputs, frame_length=self.n_fft, frame_step=self.frame_step,
            window_fn=curr_window,
            pad_end=False)
        abs_out=tf.math.abs(stfts)
        lps_out = tf.math.log(abs_out + eps1e6)
        # lps_out = tf.math.log1p(abs_out)
        phase_out=tf.math.angle(stfts)
        return lps_out, phase_out
## feat to wave
class LPSphase2istft_Layer(Layer):
    def __init__(self, n_fft, frame_step, **kwargs):
        super(LPSphase2istft_Layer, self).__init__(**kwargs)
        self.n_fft = n_fft
        self.frame_step = frame_step
    def get_config(self):
        config = super().get_config().copy()
        return config
    def call(self, inputs, training=True):
        eps1e6=tf.constant(1e-6)
        curr_window=tf.signal.hann_window
        ## LPS -> ABS
        lps_in,phase_in = inputs
        abs_in  = tf.math.exp(lps_in) - eps1e6
        # abs_in = tf.math.expm1(lps_in)
        ## Compute stft
        abs_out_complex = tf.dtypes.complex(abs_in,0*abs_in,name=None)
        phase_out_complex = tf.dtypes.complex(0*phase_in,phase_in,name=None)
        stft = abs_out_complex * tf.math.exp(phase_out_complex)
        ## Compute istft
        istft=tf.signal.inverse_stft(
            stft, frame_length=self.n_fft, frame_step=self.frame_step,
            window_fn=tf.signal.inverse_stft_window_fn(frame_step=self.frame_step,forward_window_fn=curr_window),)
        return istft
class LPSphase2istft_nowin_Layer(Layer):
    def __init__(self, n_fft, frame_step, **kwargs):
        super(LPSphase2istft_nowin_Layer, self).__init__(**kwargs)
        self.n_fft = n_fft
        self.frame_step = frame_step
        self.eps1e6=tf.constant(1e-6)
        self.curr_window=tf.signal.hann_window
    def get_config(self):
        config = super().get_config().copy()
        return config
    def call(self, inputs, training=True):
        ## LPS -> ABS
        lps_in,phase_in = inputs
        abs_in  = tf.math.exp(lps_in) - self.eps1e6
        # abs_in = tf.math.expm1(lps_in)
        ## Compute stft
        abs_out_complex = tf.dtypes.complex(abs_in,0*abs_in,name=None)
        phase_out_complex = tf.dtypes.complex(0*phase_in,phase_in,name=None)
        stft = abs_out_complex * tf.math.exp(phase_out_complex)
        ## Compute istft
        istft=tf.signal.inverse_stft(
            stft, frame_length=self.n_fft, frame_step=self.frame_step,
            window_fn=None,)
        return istft

################################# Compilation #################################
def compile_opt(_mod_in, _opt_mode, _opt_dict, compile_dict=None,
    _loss=None, _metrics=None, loss_weights=None):
    ## Choosing Optimizer
    if _opt_mode=='Adam':   _curr_opt = Adam(**_opt_dict)
    elif _opt_mode=='SGD':  _curr_opt = SGD(**_opt_dict)
    else:                   error('Undefined optimizer')
    ## Compiling them
    if compile_dict is not None: _mod_in.compile(optimizer=_curr_opt, **compile_dict)
    elif _metrics:
        _mod_in.compile(loss=_loss, optimizer=_curr_opt, metrics=_metrics)
    else:
        _mod_in.compile(loss=_loss, optimizer=_curr_opt)
    return _mod_in

################################# Callbacks #################################
def ckpt_saving(_Ckpt_dir, _ModelCheckpoint_kwargs, save_all=False):
    if save_all:
        _ckpt_path = os.path.join(_Ckpt_dir , "Epoch-{epoch:04d}")
        _monitor_modes=_ModelCheckpoint_kwargs['monitor']
        
        if   _monitor_modes == 'loss': _ckpt_path     += "_L-{"+_monitor_modes+":.5f}"
        elif _monitor_modes == 'acc': _ckpt_path      += "_A-{"+_monitor_modes+":.5f}"
        elif _monitor_modes == 'categorical_accuracy': _ckpt_path += "_CA-{"+_monitor_modes+":.5f}"

        elif _monitor_modes == 'val_loss': _ckpt_path += "_VL-{"+_monitor_modes+":.5f}"
        elif _monitor_modes == 'val_acc': _ckpt_path  += "_VA-{"+_monitor_modes+":.5f}"
        elif _monitor_modes == 'val_categorical_accuracy': _ckpt_path += "_VCA-{"+_monitor_modes+":.5f}"

        _ckpt_path += ".hdf5"
    else:  
        _ckpt_path = os.path.join(_Ckpt_dir , "best_only.hdf5")
    print('ckpt_saving() : _ckpt_path : ',_ckpt_path,'\n')
    _ModelCheckpoint_kwargs['filepath']=_ckpt_path

    _checkpoint = ModelCheckpoint(**_ModelCheckpoint_kwargs)
    return _checkpoint
def lrtrack_cb(): # Learning Rate Tracker
    # model.fit(..., callbacks=[lrtrack], ...)
    class LRTracker(Callback):
        def on_epoch_end(self, epoch, logs={}):
            optimizer = self.model.optimizer
            lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
            ratio = K.eval(1. / (1. + optimizer.decay * optimizer.iterations))
            itrs = K.eval(optimizer.iterations)
            print('LR:{:.6f}'.format(lr))
    return LRTracker()
def Save_Live_Plot(fig_path_in): # Save plot of loss every epoch
    class PlotLosses(Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []
            
            self.fig = plt.figure()
            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1
            
            clear_output(wait=True)
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()
            
            # plt.show();
            plt.savefig(fig_path_in);

    return PlotLosses()

################################# Admin #################################
def layer_output(model, layer_name, x=None, siamese=False, verbose=True):
    """
    model        : Non siamese model
    layer_name   : desired layer_name for model
    x            : input to extract bottleneck features from

    curr_layer_model : Model that gives the bottleneck outputs
    bottle_feature   : bottleneck features from x
    """
    bottle_feature = -1
    if not siamese: ## Redefine the model by using layer_name as output
        Inp = model.input
        Outp = model.get_layer(layer_name).output
        curr_layer_model = Model(Inp, Outp)
        if verbose: curr_layer_model.summary()
    else: ## If Siamese, do:
        curr_layer_model = model.get_layer(layer_name)
        if verbose: curr_layer_model.summary()

    if x is not None: bottle_feature = curr_layer_model.predict(x)

    return curr_layer_model, bottle_feature
def save_model(model, model_path=None, weights_path=None, state_path=None, verbose=True):
    """
    model        : compiled neural network model
    model_path   : dir/path.json    to save the model to
    weights_path : dir/path.h5      to save the model weights to
    state_path   : dir/path.h5      to save the model state to
    """

    if model_path: 
        if not model_path[-5:]=='.json': model_path=model_path+'.json'
        model_json = model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        if verbose : print('Saved model      : '+model_path)
    ## Save weights to HDF5
    if weights_path: 
        if not weights_path[-3:]=='.h5': weights_path=weights_path+'.h5'
        model.save_weights(weights_path)
        if verbose : print('Saved weights    : '+weights_path)
    ## Save state (model+weights+optimizer) to H5
    if state_path: 
        if not state_path[-3:]=='.h5': state_path=state_path+'.h5'
        model.save(state_path)
        if verbose : print('Saved state      : '+state_path)
    return 
def load_model_keras(model_mode, built_model=None, custom_objects=None,
    model_path=None, weights_path=None, state_path=None, verbose=True):
    if model_mode == 'path':
        """
            model_path = Weights_path+"model_v8_1.json"
            weights_path = Ckpt_Mod_Weights_fold+"v8_2/weights_v8_2_Epoch-0005_L-147.42.hdf5"
            
            v8_3 = load_model_keras('path', built_model=None, model_path=model_path, weights_path=weights_path, state_path=None)
            v8_3.summary()

            adam_opt = Adam(lr=1e-4, decay=decay)
            v8_3.compile(loss=['mse'] ,optimizer=adam_opt)
            v8_3.optimizer.get_config()
        """
        ## json -> model
        if not model_path.endswith('.json'): model_path=model_path+'.json'
        with open(model_path, 'r') as f_json: loaded_model_json = f_json.read()

        if custom_objects : 
            loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
        else :
            loaded_model = model_from_json(loaded_model_json)

        if verbose : print('Loaded model from path : '+model_path)

        ## model -> weights
        if weights_path is not None:
            if (not weights_path.endswith('.hdf5')) and (not weights_path.endswith('.h5')): 
                assert False , '\n\n{} must end with either .hdf5 or .h5\n\n'.format(weights_path)
            loaded_model.load_weights(weights_path, by_name=False)
            if verbose : print('Loaded weights         : '+weights_path)
    elif model_mode == 'compiled':
        """
            weights_path = Ckpt_Mod_Weights_fold+"v8_2/weights_v8_2_Epoch-0005_L-147.42.hdf5"
            v8_3 = build_v8(...)
            v8_3 = load_model_keras('compiled', built_model=v8_3, model_path=None, weights_path=weights_path, state_path=None)
            v8_3.summary()
        """
        loaded_model = built_model
        loaded_model.load_weights(weights_path, by_name=False)
        if verbose : print('Loaded weights on model: '+weights_path)
    elif model_mode == 'state':
        """
            state_path = Weights_path+"v8_2_state.h5"
            v8_2.save(state_path)

            v8_3 = load_model_keras('state', built_model=None, model_path=None, weights_path=None, state_path=state_path)
            v8_3.summary()

            v8_3.optimizer.get_config()
        """
        if not state_path.endswith('.h5'): state_path=state_path+'.h5'
        loaded_model = load_model(state_path)
        if verbose : 
            print('Loaded model state from: '+state_path)
            print('NO NEED to recompile, can just use model.fit right away')
    elif model_mode == 'Subclass':
        model_test = MyModel_MyLayer(num_classes=10)
        _ = model_test.predict(x_tst) # To initalise this subclassed model
        model_test = load_model_keras(  model_mode='compiled', 
                                        built_model=model_test, 
                                        weights_path=conf_DNN.weights_path, 
                                        verbose=True)
        y_hat_tst = model_test.predict(x_tst)
        print('y_hat_tst : ',y_hat_tst)
    
    return loaded_model
