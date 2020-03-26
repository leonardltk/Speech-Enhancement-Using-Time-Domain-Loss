#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
if 1 : ## Imports
    import sys, os, datetime, traceback, pprint, pdb # pdb.set_trace()
    import subprocess, itertools, importlib , math, glob, time, random, shutil, csv, statistics
    from operator import itemgetter
    import numpy as np
    import pickle
    ## Plots
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    # import seaborn 
    # import pandas as pd
    ## Audio
    import wave, python_speech_features#, pyaudio
    import librosa, librosa.display
    import scipy, scipy.io, scipy.io.wavfile, scipy.signal
    import soundfile as sf
    import resampy
    import audiofile as af

##########################################################################################################################################################
## Admin
## Saving/Loading 
def dump_load_pickle(file_Name, mode, a=None):
    if mode == 'dump':
        # open the file for writing
        fileObject = open(file_Name,'wb') 
        
        # this writes the object a to the file named 'testfile'
        pickle.dump(a,fileObject, protocol=2)   
        # cPickle.dump(a,fileObject, protocol=2)   
        
        # here we close the fileObject
        fileObject.close()
        b = 'dumped '+file_Name
    elif mode == 'load':
        # we open the file for reading
        fileObject = open(file_Name,'rb')  
        
        # load the object from the file into var b
        b = pickle.load(fileObject)  
        # b = cPickle.load(fileObject)  
        
        # here we close the fileObject
        fileObject.close()
    return b

##########################################################################################################################################################
## Audio
def read_audio(filename_in, mode="audiofile", sr=None, mean_norm=False):
    """
        Input : file name
        Return : waveform in np.float16, range [-1,1]

        mode=="scipy" : will read in int16, which we will convert to float and divide by 2**15
        
        mean_norm=True if want to return the mean_normalised waveform
        
        matlab reads in the same way as librosa do.
        
        If read audio is in different channels : use def make_single_channel()
    """

    if mode=="wave":
        with wave.open(filename_in) as w:
            data = w.readframes(w.getnframes())
        sig = np.frombuffer(data, dtype='<i2').reshape(-1, channels)
        normalized = utility.pcm2float(sig, np.float32)
        sound = normalized
        # 
        # sound_wav = wave.open(filename_in)
        # n = sound_wav.getnframes()
        # sound = sound_wav.readframes(n)
        # debug_cannot_extract_array_values
        # 
        sound_fs = sound_wav.getframerate()
    elif mode=="scipy":
        [sound_fs, sound] = scipy.io.wavfile.read(filename_in)
        if sr and sr!=sound_fs: 
            sound = resampy.resample(sound, sound_fs, sr, axis=0)
            sound_fs=sr
        assert sound.dtype=='int16'
        sound = 1.*sound
    elif mode=="librosa":
        # must define sr=None to get native sampling rate
        sound, sound_fs = librosa.load(filename_in,sr=sr)
        # sound *= 2**15
    elif mode=="soundfile":
        sound, sound_fs = sf.read(filename_in)
        if sr and sr!=sound_fs: 
            sound = resampy.resample(sound, sound_fs, sr, axis=0)
            sound_fs=sr
    elif mode=="audiofile":
        sound, sound_fs = af.read(filename_in)
        if sr and sr!=sound_fs: 
            sound = resampy.resample(sound, sound_fs, sr, axis=0)
            sound_fs=sr
    
    if mean_norm: sound -= sound.mean()
    
    return sound, sound_fs
def write_audio(filename_out,x_in,sr,mode="soundfile"):
    """
        Assume input is in the form np.int16, with range [-2**15,2**15]
    """
    curr_x_in_dtype=x_in.dtype
    if mode == "librosa":
        print('\nThis is now deprecated, use mode==soundfile instead\n')
        # assert (curr_x_in_dtype==np.float16)       , '{} is wrong, save in np.float16'.format(curr_x_in_dtype)
        assert np.max(np.abs(x_in))<=1      , '{} is out of range'.format(filename_out)
        librosa.output.write_wav(filename_out, x_in, sr)
    elif mode == "scipy":
        assert curr_x_in_dtype==np.int16         , 'curr_x_in_dtype={} is wrong, save in np.int16'.format(curr_x_in_dtype)
        assert (not np.max(np.abs(x_in))>2**15)  , 'max is {} .\n {} is out of range'.format(np.max(np.abs(x_in)),filename_out)
        assert (not np.max(np.abs(x_in))==0) , 'min is {} .\n{} is either double in [-1,1] or 0Hz, please check, skipping...'.format(np.min(np.abs(x_in)),filename)
        scipy.io.wavfile.write(filename_out, sr, x_in)
    elif mode == "soundfile":
        assert np.max(np.abs(x_in))<=1      , '{} is out of range'.format(filename_out)
        sf.write(filename_out,x_in,sr)
    else:
        print('mode:{} is incorrect should be librosa/scipy/soundfile'.format(mode))
def display_audio(sig, title, 
    mode='wave', sr=None, hop_length=None, 
        fmin=None, fmax=None,
        x_axis='time', y_axis='hz', 
        num_bins=None,
    xlims=None,ylims=None,clims=None,
    autoplay=False, colorbar=None, 
    colour_to_set='white', curr_fig=None, kcoll=None):

    if curr_fig and kcoll: 
        ax = curr_fig.add_subplot(kcoll[0],kcoll[1],kcoll[2])

    ## Modes ----------------------------------------------
    if mode=='wave':
        librosa.display.waveplot(sig, sr=sr)
        if not title==None: plt.title(title) 
    if mode=='plot':
        plt.plot(sig)
        if not title==None: plt.title(title) 
    elif mode=='spec':
        librosa.display.specshow(sig, 
            sr=sr, hop_length=hop_length,
            fmin=fmin, fmax=fmax,
            x_axis=x_axis, y_axis=y_axis)
        if not title==None: plt.title(title) 
    elif mode=='matplot':
        plt.imshow(sig)
        if not title==None: plt.title(title) 
    elif mode=='audio':
        
        import IPython.display as ipd
        # ipd.display( ipd.Audio(""+"hello.wav") )
        # ipd.display( ipd.Audio(spkr, rate=sr) )

        if not title==None: print(title)
        ipd.display( ipd.Audio(sig, rate=sr, autoplay=autoplay) )
    elif mode=='audio_terminal':
        if not title==None: print(title)
        play_audio(sig,CHUNK=1024)
    elif mode=='image':
        import IPython.display as ipd
        # ipd.display( ipd.Image("LSTM.png") )
        # ipd.display( ipd.Image(x, format='png') )

        ipd.display( ipd.Image(sig, format='png') )
        if not title==None: plt.title(title) 
    elif mode=='constellation_points':
        """ Usage : 

            ## Read Audio
            x,sr_out=read_audio(song_path, mode="librosa", sr=sr, mean_norm=False)
            
            ## Get feats
            kwargs_STFT={
                'pad_mode':True,
                'mode':'librosa',
                    'n_fft':conf_sr.n_fft,
                    'win_length':conf_sr.win_length,
                    'hop_length':conf_sr.hop_length,
                    'nfft':conf_sr.nfft,
                    'winstep':conf_sr.winstep,
                    'winlen':conf_sr.winlen,
                    'fs':conf_sr.sr,
            }
            x_MAG, _,_,x_LPS=wav2LPS(x, **kwargs_STFT)
            
            ## ...Dropbox/Work/BIGO/2_Projects/002_MusicHashing/jja178/Combined/v1_baseline/utils/_Shazam_.py
            import _Shazam_ as Shazam
            kwargs_hashPeaks={
                'num_tgt_pts':3,
                "delay_time" : seconds_to_frameidx(1, conf_sr.hop_length, conf_sr.n_fft, conf_sr.sr),
                "delta_time" : seconds_to_frameidx(5, conf_sr.hop_length, conf_sr.n_fft, conf_sr.sr),
                "delta_freq" : Hz_to_freqidx(1500, conf_sr.num_freq_bins, conf_sr.sr),
                }
            raw_peaks = Shazam.get_raw_constellation_pts(x_MAG) 
            filt_peaks = Shazam.filter_peaks(raw_peaks, conf_sr.n_fft, high_peak_percentile=75,low_peak_percentile=60)
            filt_peaks_large = [(curr_peak[0],curr_peak[1],10) for curr_peak in filt_peaks]
            # hashMatrix = Shazam.hashPeaks(filt_peaks, conf_sr, **kwargs_hashPeaks)
            [(curr_peak[0],curr_peak[1],10) for curr_peak in filt_peaks]
            
            ## Plot
            k=3;col=1;l=1; curr_fig=plt.figure(figsize=(6*col,3*k)); 
            kwargs_plot={'colour_to_set':'black','hop_length':conf_sr.hop_length,'sr':conf_sr.sr,'curr_fig':curr_fig,}
            display_audio(x,     'x',       'wave', kcoll=[k,col,l], **kwargs_plot); l+=1
            display_audio(x_LPS, 'x_LPS',   'spec', kcoll=[k,col,l], **kwargs_plot); l+=1
            display_audio([x_LPS,filt_peaks_large], 'Peaks', 'constellation_points', kcoll=[k,col,l], **kwargs_plot); l+=1
            plt.tight_layout();plt.show()
            plt.savefig('plot_path.png')
        """
        curr_lps,curr_peaks = sig
        librosa.display.specshow(curr_lps, sr=sr, hop_length=hop_length)
        plt.scatter(*zip(*curr_peaks), marker='.', color='blue', alpha=0.5)
        if not title==None: plt.title(title) 
    elif mode=='histogram':
        """ Usage : 
            x = [i for i in range(100)]
            
            ## Plot
            k=3;col=1;l=1; curr_fig=plt.figure(figsize=(6*col,3*k)); 
            kwargs_plot={'colour_to_set':'black', 'mode':histogram, 'curr_fig':curr_fig,}
            display_audio(x,     'x',       'histogram', kcoll=[k,col,l], **kwargs_plot); l+=1
            plt.tight_layout();plt.show()
            plt.savefig('plot_path.png')
        """
        plt.hist(sig, bins=num_bins)
        if not title==None: plt.title(title) 
    ## Modes ----------------------------------------------
    
    if not colorbar==None: plt.colorbar() 
    if not xlims==None: plt.xlim(xlims) 
    if not ylims==None: plt.ylim(ylims) 
    if not clims==None: plt.clim(clims) 

    if colour_to_set and curr_fig and kcoll:
        ax.spines['bottom'].set_color(colour_to_set)
        ax.spines['top'].set_color(colour_to_set)
        ax.yaxis.label.set_color(colour_to_set)
        ax.xaxis.label.set_color(colour_to_set)
        ax.tick_params(axis='both', colors=colour_to_set)
        ax.title.set_color(colour_to_set)

##########################################################################################################################################################
## Augmentation
def calc_power(_x_in):
    assert len(_x_in.shape)==1, 'it should be waveform i.e. vector '
    power_out = np.linalg.norm(_x_in)**2/len(_x_in)

    return power_out
def power_norm(_wav_in, _des_power, _prevent_clipping=True):

    wav_Power   = calc_power(_wav_in)

    ## Calc output power
    wav_pnorm = _wav_in*(_des_power/wav_Power)**.5
    renorm_Pow=calc_power(wav_pnorm)

    ## To prevent clipping
    if _prevent_clipping:
        max_val=np.max(np.abs(wav_pnorm))
        if max_val>1:
            print(max_val)
            wav_pnorm /= max_val
    
    return wav_pnorm
def get_alpha_SNR(clean_P,noise_P, _snr, _mode):
    ## Calc alpha, assuming they are of the same power
    if _mode=="CP":
        k=noise_P/clean_P
        k*=10**(_snr/10)
        k**=.5
        alpha_out=k/(1+k)
        beta_out=1-alpha_out
    if _mode=="LL":
        k = 10 ** (_snr/10)
        alpha_out = (k/(k+1))**0.5
        beta_out = 1/(k+1)**0.5
    return alpha_out,beta_out

def augment_noisy_wave(  _clean_WAV, _noise_WAV, _snr,  allow_clipping=True): 
    ## y = a*clean + b*noise

    num_samples_clean=float(np.max(_clean_WAV.shape))
    num_samples_noise=float(np.max(_noise_WAV.shape))
    ## Repeat the noise until it is of same length as _clean_WAV
    factor_tile=np.ceil(num_samples_clean/num_samples_noise)
    factor_tile=int(factor_tile)
    _noise_WAV=np.tile(_noise_WAV,factor_tile)
    _noise_WAV=_noise_WAV[:len(_clean_WAV)]

    ## Power 
    clean_P=calc_power(_clean_WAV)
    _noise_WAV_pnorm=power_norm(_noise_WAV, clean_P, allow_clipping)
    noise_P=calc_power(_noise_WAV_pnorm)

    ## Make Noisy
    __alpha,__beta=get_alpha_SNR(clean_P,noise_P, _snr, _mode="LL")
    _noisy_WAV_pnorm=__alpha*_clean_WAV + __beta*_noise_WAV_pnorm

    if not allow_clipping:
        num_clipped=np.sum(_noisy_WAV_pnorm>1)
        num_clipped+=np.sum(_noisy_WAV_pnorm<-1)
        _noisy_WAV_pnorm[_noisy_WAV_pnorm>1]=1.
        _noisy_WAV_pnorm[_noisy_WAV_pnorm<-1]=-1.
        # print('Clipped {} samples'.format(num_clipped))
    return _noisy_WAV_pnorm
def augment_reverb_wave( _clean_WAV, _RIR_WAV,   pow_eq=True): 
    def shift(xs,n):
        e = np.empty_like(xs)
        if n>= 0:
            e[:n]=0.
            e[n:]=xs[:-n]
        else:
            e[n:]=0.
            e[:n]=xs[-n:]
        return e
    ## Make Noisy
    _reverb_WAV=scipy.signal.fftconvolve(_clean_WAV, _RIR_WAV, mode='full')
    # pdb.set_trace() # !import code; code.interact(local=vars())
    p_max=np.argmax(_RIR_WAV)
    _reverb_WAV=shift(_reverb_WAV, -p_max)
    _reverb_WAV = _reverb_WAV[:len(_clean_WAV)]
    ## Power 
    if pow_eq:
        clean_P=calc_power(_clean_WAV)
        reverb_P=calc_power(_reverb_WAV)
        _reverb_WAV = power_norm(_reverb_WAV, clean_P)
    return _reverb_WAV
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

