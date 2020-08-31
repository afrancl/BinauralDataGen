import math
import json
import pdb
from hanning_window import hanning_window
from glob import glob
import tensorflow as tf
import numpy as np
import sys
from pycochleagram import cochleagram as cgm
from pycochleagram import utils as utl
from os.path import basename
import os
import json
from nnresample import resample



version = int(sys.argv[1])
train_filename = '/nobackup/scratch/Wed/francl/noiseRecords_logspaced_fluctuating_0.5octv_anechoic_convolvedHRIR140_stackedCH_upsampled/testset/train{}.tfrecords'.format(version) 
#file_path = './stimuli_out_full_set/testset/*.wav'
#file_path = '/om/user/francl/recorded_binaural_audio_4078_main_2_mic_array_rescaled/*.wav' 
#file_path = '/om/scratch/Wed/francl/broadband_whitenoise/*.wav' 
file_path ='/nobackup/scratch/Wed/francl/logspaced_bandpassed_fluctuating_noise_0.5octv_out_convolved_anechoic_oldHRIRdist140_no_hanning/testset/*.wav' 
#file_path = '/om/scratch/Wed/francl/precedence_effect_pinknoise_jittered_start_jitteredPt5ms_lag_multiAzimLead_0deg_5degStep_rescaled/*.wav' 
#version = 2000
# address to save the TFRecords file
signal_rate =48000
remove_ILD = False


##REVERTS TO OLD RECORD FORMAT###
revert_record_format = False


##STIMULI CONDITIONS##
#flag for background stimuli to remove label flags
background = False
#flag to parse metadata dictionaries for sparse bkgd textures
background_textures = False
#adds a frequency label to the record
freq_label = False
#True if running fixed bandwidth spatialized noise bursts
noise_bursts = False
#True if running for SAM tones
sam_tones = False
#True if running transposed_tons
transposed_tones = False
#True if running with precedence effect clicks
precedence_effect = False
#True if running with precedence effect clicks with multiple lead postions
precedence_effect_multi_azim = False
#True if running with gaussian noise for wood azimuth experiment
wood_noise=False
#True if running spatialized noise with varying freqs/bandwidths
narrowband_noise = False
#changes string parsing to get manually added labels
man_added = False
#changes string parsing to get manually added labels for both ITD and ILD
joint_man_added = False
#CIPIC dataset flag, parses subject number as well as az/elev
CIPIC = False
#Use for smooth HRTFs with broadband noise to extract smoothing factor
smoothed_hrtf = False
#Use for ITD/ILD bias added to bandpass nois
middlebrooks_wrightman = False
#True if processing recorded binaural sounds
binaural_recorded = False
#BRIR version has different labels and postions in string
BRIR_ver = False
#Make cochleagram without any labels
no_labels = False

##PREPROCESSING FLAGS##
#apply seond hanning window before cochleagram calculation
hanning_windowed = False
#zero hanning windowed sound to avoid onset clickes
zero_padded = False
#stacks L/R channels in last dimension isntead of interleaving data
channel_stack = True
#slicing directly from cochleagram to avoid pop onsets
sliced=True
no_back_limit=False
minimum_padding=round(.35*signal_rate)
final_stim_length = round(1.0*signal_rate)

#Change minimum padding if using tranposed or sam_tones to ensure stimuli is in
#sampled portion
minimum_padding = 21000 if transposed_tones or sam_tones else minimum_padding
minimum_padding = 16000 if precedence_effect or wood_noise else minimum_padding

hi_lim =20000
if revert_record_format:
    hi_lim = 15000
    channel_stack = False
    signal_rate = 44100
    final_stim_length = 30000

#Cochleagram gen signal cutoff defies how much of the waveform is proceesed
#to subbands. This is important becasue it both removes the reverberation after
#simulated sound source stops making noise but must still be large enough to accomadate the
#varyig signal lengths of the input audio
coch_gen_sig_cutoff = 2*signal_rate

#Split filenames into train and test set
source_files = glob(file_path)

#get center frequencites of subbands
center_freqs = np.load("/om/user/francl/cochleagram39CH_center_freqs.npy")

#Type conversion functions
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature_numpy(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def calculate_ILD(subbands_l,subbands_r,freq_range,center_freqs):
    valid_indicies = [i for i,x in enumerate(center_freqs) if
                      freq_range[0]<x<freq_range[1]]
    subbands_l_sum = sum([np.sqrt(np.square(x.mean())) for i,x in enumerate(subbands_l) if i in
                          valid_indicies])
    subbands_r_sum = sum([np.sqrt(np.square(x.mean())) for i,x in enumerate(subbands_r) if i in
                          valid_indicies])
    ratio = 20*math.log10(subbands_l_sum/subbands_r_sum)
    return ratio

def cochleagram_wrapper(stim):
    stim_wav, stim_freq = utl.wav_to_array(stim, rescale=None)
    stim_basename = basename(stim)
    stim_wav = stim_wav.T
    if stim_freq != signal_rate:
        stim_wav_empty = np.empty_like(stim_wav)
        print("resampling")
        stim_wav_l = resample(stim_wav[0],signal_rate,stim_freq,As=75,N=64001)
        stim_wav_r = resample(stim_wav[1],signal_rate,stim_freq,As=75,N=64001)
        stim_freq = signal_rate
        stim_wav = np.vstack([stim_wav_l,stim_wav_r])

    #transpose to split channels to speed up calculating subbands
    stim_wav = stim_wav[:,:coch_gen_sig_cutoff]
    delay = 15000
    #first dimesnion due to transpose
    total_singal = stim_wav.shape[1]
    sample_factor = 1   

    #Apply a hanning window to the stimulus
    if hanning_windowed:
        hann_r = hanning_window(stim_wav[1],20,SAMPLERATE=44100)
        hann_l = hanning_window(stim_wav[0],20,SAMPLERATE=44100)
        r_channel = hann_r
        l_channel = hann_l
    else:
        r_channel = stim_wav[1]
        l_channel = stim_wav[0]

    
    #calculate subbands
    subbands_r = cgm.human_cochleagram(r_channel,
                          stim_freq,low_lim=30,hi_lim=hi_lim,sample_factor=sample_factor,
                          padding_size=10000,ret_mode='subband').astype(np.float32)
    
    subbands_l = cgm.human_cochleagram(l_channel,
                          stim_freq,low_lim=30,hi_lim=hi_lim,sample_factor=sample_factor,
                          padding_size=10000,ret_mode='subband').astype(np.float32)

    if sliced:
        front_limit = minimum_padding
        if no_back_limit:
            back_limit= total_singal-final_stim_length
        else:
            back_limit= total_singal-minimum_padding-final_stim_length
        jitter = np.random.randint(round(front_limit),round(back_limit))
        front_slice = jitter
        back_slice = jitter+final_stim_length
        #44100*300ms = 13000
        subbands_l = subbands_l[:,front_slice:back_slice]
        subbands_r = subbands_r[:,front_slice:back_slice]

    if remove_ILD:
        rms_l = np.sqrt(np.square(subbands_l).mean(axis=1)) 
        rms_r = np.sqrt(np.square(subbands_r).mean(axis=1)) 
        max_rms = rms_l if rms_l.mean() > rms_r.mean() else rms_r
        norm_subbands_l = subbands_l / rms_l[:,None]
        norm_subbands_r = subbands_r / rms_r[:,None]
        subbands_r = norm_subbands_r*max_rms[:,None]
        subbands_l = norm_subbands_l*max_rms[:,None]

    if channel_stack:
        num_channels = subbands_l.shape[0] - 2*sample_factor
        subbands = np.empty([num_channels,final_stim_length,2],dtype=subbands_l.dtype)
        #not taking first and last filters because we don't want the low and
        #highpass filters
        subbands[:,:,0] = subbands_l[sample_factor:-sample_factor]
        subbands[:,:,1] = subbands_r[sample_factor:-sample_factor]
    else:
        #Interleaving subbands,so local filters can access both channels
        num_channels = subbands_l.shape[0] - 2*sample_factor
        subbands = np.empty([(2*num_channels),final_stim_length],dtype=subbands_l.dtype)
        subbands[0::2] = subbands_l[sample_factor:-sample_factor]
        subbands[1::2] = subbands_r[sample_factor:-sample_factor]

    #Cut anything -60 dB below peak
    max_val = subbands.max() if subbands.max() > abs(subbands.min()) else abs(subbands.min())
    cutoff = max_val/1000
    subbands[np.abs(subbands) < cutoff] = 0
    #text input as bytes so bytes objects necessary for comparison
    return subbands

def parse_labels_filename(stim_path, metadata_dict):
    stim = basename(stim_path)
    if background:
        return None
    elif background_textures:
        metadata = metadata_dict[stim]        
        azim = [int(x[0]) for x in metadata]
        elev = [int(x[1]) for x in metadata]
        labels = [azim,elev]
    elif man_added:
        azim = np.array(int(stim.split('_')[-1].split('I')[0]),dtype=np.int32)
        elev=0
        carrier_freq = np.array(int(stim.split('_')[1]),dtype=np.int32)
        labels = [azim,elev,carrier_freq]
    elif joint_man_added:
        ITD = np.array(int(stim.split('_')[-2].split('I')[0]),dtype=np.int32)
        ILD = np.array(int(stim.split('_')[-1].split('I')[0]),dtype=np.int32)
        carrier_freq = np.array(int(stim.split('_')[1]),dtype=np.int32)
        labels = [ITD,ILD,carrier_freq]
    elif BRIR_ver:
        azim = np.array(int(stim_basename.split('_')[-3].split('a')[0]),dtype=np.int32)
        elev = 0
        labels = [azim,elev]
    elif transposed_tones:
        modulation_freq = np.array(int(stim.split('_')[2].split('m')[0]),
                                   dtype=np.int32)
        carrier_freq = np.array(int(stim.split('_')[3].split('c')[0]),
                                dtype=np.int32)
        delay = np.array(float(stim.split('_')[4].split('d')[0]),
                                    dtype=np.float32)
        flipped = np.array(1 if len(stim.split('_')) == 6 else 0
                           ,dtype=np.int32)
        labels = [carrier_freq, modulation_freq, delay, flipped]
    elif sam_tones:
        carrier_freq = np.array(int(stim.split('_')[2].split('c')[0]),
                                dtype=np.int32)
        modulation_freq = np.array(int(stim.split('_')[3].split('m')[0]),
                                   dtype=np.int32)
        carrier_delay = np.array(float(stim.split('_')[4].split('c')[0]),
                                 dtype=np.float32)
        modulation_delay = np.array(float(stim.split('_')[5].split('m')[0]),
                                    dtype=np.float32)
        flipped = np.array(1 if len(stim.split('_')) == 7 else 0
                           ,dtype=np.int32)
        labels = [carrier_freq, modulation_freq, carrier_delay, modulation_delay,flipped]
    elif precedence_effect_multi_azim:
        delay = np.array(float(stim.split('_')[2]),dtype=np.float32)
        start_sample = np.array(int(stim.split('_')[5].split('t')[2]),
                                dtype=np.int32)
        lead_level = np.array(float(stim.split('_')[6].split('l')[0]),
                                    dtype=np.float32)
        lag_level = np.array(float(stim.split('_')[7].split('l')[0]),
                                    dtype=np.float32)
        azim = np.array(int(stim.split('_')[4].split('a')[0]),dtype=np.int32)
        flipped = np.array(1 if len(stim.split('_')) == 9 else 0
                           ,dtype=np.int32)
        labels = [delay, azim, start_sample, lead_level, lag_level, flipped]
    elif precedence_effect:
        delay = np.array(float(stim.split('_')[2]),dtype=np.float32)
        start_sample = np.array(int(stim.split('_')[4].split('t')[2]),
                                dtype=np.int32)
        lead_level = np.array(float(stim.split('_')[5].split('l')[0]),
                                    dtype=np.float32)
        lag_level = np.array(float(stim.split('_')[6].split('l')[0]),
                                    dtype=np.float32)
        flipped = np.array(1 if len(stim.split('_')) == 8 else 0
                           ,dtype=np.int32)
        labels = [delay, start_sample, lead_level, lag_level, flipped]
    elif narrowband_noise:
        bandwidth = np.array(float(stim.split('_')[2].split('b')[0]),
                                    dtype=np.float32)
        carrier_freq = np.array(int(stim.split('_')[3].split('c')[0]),
                                dtype=np.int32)
        azim = np.array(int(stim.split('_')[-4].split('a')[0]),dtype=np.int32)
        elev = np.array(int(stim.split('_')[-5].split('e')[0]),dtype=np.int32)
        labels = [azim,elev,bandwidth,carrier_freq]
    elif CIPIC:
        azim = np.array(int(stim.split('_')[-2].split('a')[0]),dtype=np.int32)
        elev = np.array(int(stim.split('_')[-3].split('e')[0]),dtype=np.int32)
        noise_idx = np.array(int(stim.split('_')[-4].split('i')[0]),dtype=np.int32)
        subject_num = np.array(int(stim.split('_')[-1].split('t')[1].split('.')[0]),dtype=np.int32)
        labels = [azim,elev,noise_idx,subject_num]
    elif middlebrooks_wrightman:
        azim = np.array(int(stim.split('_')[-4].split('a')[0]),dtype=np.int32)
        elev = np.array(int(stim.split('_')[-5].split('e')[0]),dtype=np.int32)
        ITD = np.array(int(stim.split('_')[-3].split('I')[0]),dtype=np.int32)
        ILD = np.array(int(stim.split('_')[-2].split('I')[0]),dtype=np.int32)
        noise_idx = np.array(int(stim.split('_')[-6].split('i')[0]),dtype=np.int32)
        subject_num = np.array(int(stim.split('_')[-1].split('t')[1].split('.')[0]),dtype=np.int32)
        low_cutoff = np.array(int(stim.split('_')[-8].split('l')[0]),dtype=np.int32)
        high_cutoff = np.array(int(stim.split('_')[-7].split('h')[0]),dtype=np.int32)
        labels = [azim,elev,ITD,ILD,noise_idx,low_cutoff,high_cutoff,subject_num]
    elif smoothed_hrtf:
        azim = np.array(int(stim.split('_')[-2].split('a')[0]),dtype=np.int32)
        elev = np.array(int(stim.split('_')[-3].split('e')[0]),dtype=np.int32)
        noise_idx = np.array(int(stim.split('_')[-4].split('i')[0]),dtype=np.int32)
        smooth_factor = np.array(int(stim.split('_')[-1].split('s')[0]),dtype=np.int32)
        labels = [azim,elev,noise_idx,smooth_factor]
    elif noise_bursts:
        azim = np.array(int(stim.split('_')[-4].split('a')[0]),dtype=np.int32)
        elev = np.array(int(stim.split('_')[-5].split('e')[0]),dtype=np.int32)
        carrier_freq = np.array(int(stim.split('_')[1]),dtype=np.int32)
        labels = [azim,elev,carrier_freq]
    elif binaural_recorded:
        azim = np.array(int(stim.split('_')[-2]),dtype=np.int32)
        elev = 0
        speech_flag = 0 if "Class" in stim.split('_')[0] else 1
        speech = np.array(speech_flag,dtype=np.int32) 
        labels = [azim, elev, speech_flag]
    elif freq_label:
        azim = np.array(int(stim.split('_')[-4].split('a')[0]),dtype=np.int32)
        elev = np.array(int(stim.split('_')[-5].split('e')[0]),dtype=np.int32)
        freq = np.array(int(stim.split('_')[1]),dtype=np.int32)
        labels = [azim,elev,freq]
    elif no_labels:
        labels = [0,0]
    else:
        azim = np.array(int(stim.split('_')[-4].split('a')[0]),dtype=np.int32)
        elev = np.array(int(stim.split('_')[-5].split('e')[0]),dtype=np.int32)
        labels = [azim,elev]
    return labels

def create_feature(subbands,labels=None):
    if background:
        feature = {'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                  }
    elif background_textures:
        feature = {'train/azim': _int64_feature_numpy(labels[0]),
                   'train/elev': _int64_feature_numpy(labels[1]),
                   #'train/class_num': _int64_feature(labels[2]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                  }
    elif joint_man_added:
        feature = {'train/ITD': _int64_feature(labels[0]),
                   'train/ILD': _int64_feature(labels[1]),
                   'train/azim': _int64_feature(0),
                   'train/elev': _int64_feature(0),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                   'train/freq': _int64_feature(labels[2])
                  }
    elif man_added:
        feature = {'train/azim': _int64_feature(labels[0]),
                   'train/elev': _int64_feature(labels[1]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                   'train/freq': _int64_feature(labels[2])
                  }
    elif freq_label:
        feature = {'train/azim': _int64_feature(labels[0]),
                   'train/elev': _int64_feature(labels[1]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                   'train/freq': _int64_feature(labels[2])
                  }
    elif noise_bursts:
        feature = {'train/azim': _int64_feature(labels[0]),
                   'train/elev': _int64_feature(labels[1]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                   'train/freq': _int64_feature(labels[2])
                  }
    elif sam_tones:
        feature = {'train/carrier_freq': _int64_feature(labels[0]),
                   'train/modulation_freq': _int64_feature(labels[1]),
                   'train/carrier_delay': _float_feature(labels[2]),
                   'train/modulation_delay': _float_feature(labels[3]),
                   'train/flipped': _int64_feature(labels[4]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                  }
        
    elif transposed_tones:
        feature = {'train/carrier_freq': _int64_feature(labels[0]),
                   'train/modulation_freq': _int64_feature(labels[1]),
                   'train/delay': _float_feature(labels[2]),
                   'train/flipped': _int64_feature(labels[3]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                  }
    elif precedence_effect_multi_azim:
        feature = {'train/delay': _float_feature(labels[0]),
                   'train/azim': _int64_feature(labels[1]),
                   'train/start_sample': _int64_feature(labels[2]),
                   'train/lead_level': _float_feature(labels[3]),
                   'train/lag_level': _float_feature(labels[4]),
                   'train/flipped': _int64_feature(labels[5]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                  }
    elif precedence_effect:
        feature = {'train/delay': _float_feature(labels[0]),
                   'train/start_sample': _int64_feature(labels[1]),
                   'train/lead_level': _float_feature(labels[2]),
                   'train/lag_level': _float_feature(labels[3]),
                   'train/flipped': _int64_feature(labels[4]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                  }
    elif narrowband_noise:
        feature = {'train/azim': _int64_feature(labels[0]),
                   'train/elev': _int64_feature(labels[1]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                   'train/bandwidth': _float_feature(labels[2]),
                   'train/center_freq': _int64_feature(labels[3])
                  }
    elif CIPIC:
        feature = {'train/azim': _int64_feature(labels[0]),
                   'train/elev': _int64_feature(labels[1]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                   'train/noise_idx': _int64_feature(labels[2]),
                   'train/subject_num': _int64_feature(labels[3])
                  }
    elif middlebrooks_wrightman:
        feature = {'train/azim': _int64_feature(labels[0]),
                   'train/elev': _int64_feature(labels[1]),
                   'train/ITD': _int64_feature(labels[2]),
                   'train/ILD': _int64_feature(labels[3]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                   'train/noise_idx': _int64_feature(labels[4]),
                   'train/low_cutoff': _int64_feature(labels[5]),
                   'train/high_cutoff': _int64_feature(labels[6]),
                   'train/subject_num': _int64_feature(labels[7])
                  }
    elif smoothed_hrtf:
        feature = {'train/azim': _int64_feature(labels[0]),
                   'train/elev': _int64_feature(labels[1]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                   'train/noise_idx': _int64_feature(labels[2]),
                   'train/smooth_factor': _int64_feature(labels[3])
                  }
    elif binaural_recorded:
        feature = {'train/azim': _int64_feature(labels[0]),
                   'train/elev': _int64_feature(labels[1]),
                   'train/speech': _int64_feature(labels[2]),
                   #'train/class_num': _int64_feature(labels[2]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                  }
    else:
        feature = {'train/azim': _int64_feature(labels[0]),
                   'train/elev': _int64_feature(labels[1]),
                   #'train/speech': _int64_feature(labels[2]),
                   #'train/class_num': _int64_feature(labels[2]),
                   'train/image': _bytes_feature(tf.compat.as_bytes(subbands.tostring())),
                   'train/image_height': _int64_feature(subbands.shape[0]),
                   'train/image_width': _int64_feature(subbands.shape[1]),
                  }
    return feature


def create_record():
    # open the TFRecords file
    # This "version" variable is used to paralleliz the data on a cluster
    # You should run the cieling of len(training_addrs)/5 processes with the arg
    # being the int corresponding to the process
    metadata_dict = {}
    if background_textures:
        json_fnames=glob(file_path.split('*')[0] + '*.json')
        for fname in json_fnames:
            with open(fname, 'r') as f:
                temp_dict = json.load(f)
                metadata_dict.update(temp_dict)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(train_filename,options=options)
    for i in range(200*version,min(200+200*version,len(source_files))):
        subbands = cochleagram_wrapper(source_files[i])
        labels = parse_labels_filename(source_files[i], metadata_dict)
        #split images from labels
        # print how many images are saved every 1000 images
        #if not i % 100:
        #    print("Train data: {0}/ {1}".format(i, len(source_files)))
        #    sys.stdout.flush()
        # Load the image
        # Create a feature
        # Create an example protocol buffer
        feature = create_feature(subbands,labels)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()


def check_record():
    reader_opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    record_iterator = tf.python_io.tf_record_iterator(path=train_filename,
                                                      options=reader_opts)
    #check for corrupted records
    global i
    i=0
    try:
        for _ in record_iterator:
            i += 1
    except Exception as e:
        print('Error in {} at record {}'.format(train_filename, i))
        print(e)
        return False
    return True

print('running')
create_record()
correct = check_record()
if not correct:
    create_record()
    correct = check_record()
    if not correct:
        os.remove(train_filename)
        raise TypeError("Data corrupted in {} at record {} and is unrecoverable. File Deleted.".format(train_filename,i))
