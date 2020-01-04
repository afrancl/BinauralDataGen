import numpy as np
from hanning_window import hanning_window
import scipy.io
from glob import glob
from scipy.io.wavfile import read, write
from scipy import signal
from os.path import basename
import sys
import pdb
import json

DEUBUG = False
out_path="/scratch/Wed/francl/nsynth-test-spatialized"
#out_path = "./pure_tones_out/"
rate_out=44100  #Desired sample rate of "spatialized" WAV
rate_in=44100   #Sample Rate for Stimuli

#appends metadata to preexisting json dicts
metadata_dict_mode = True
#stim_files = glob("/om/user/francl/SoundLocalization/noise_samples_1octvs_jittered/*.wav")
stim_paths = "/scratch/Wed/francl/nsynth-test"
stim_files = glob(stim_paths+"/audio/*.wav") if metadata_dict_mode else glob(stim_paths+"/*.wav")
#stim_files = glob("/nobackup/scratch/Sat/francl/bandpassed_noise_0.3octv_fullspec_jittered_phase_logspace_new_2019_11_19/*.wav")
#stim_files = glob("/om/user/gahlm/sorted_sounds_dataset/sorted_stimuli_specfilt/testset/*.wav")    #File path to stimuli
env_paths = sorted(glob("/om/user/francl/Room_Simulator_20181115_Rebuild/HRIRdist140-5deg_elev_az_room*"))
#env_paths = sorted(glob("/om/user/francl/Room_Simulator_20181115_Rebuild_2_mic_version/2MicIRdist140-5deg_elev_az_room*"))
#env_paths = sorted(glob("/om/user/francl/Room_Simulator_20181115_Rebuild/Anechoic_HRIRdist140-5deg_elev_az_room5x5y5z_materials26wall26floor26ciel"))
ramp_dur_ms = 10
filter_str = ''

#zero padding options
zero_pad = True
padding_samples = 4000 

#jitters ITD if true
vary_itd_flag = False

#stim_files = glob("./pure_tones/*.wav")    #File path to stimuliu
#scales loudness of sounds DO NOT CHANGE
scaling = 0.1

#scales probability of rendering any given position for a sound
#Nsynth
prob_gen = 0.01
#Use for 1oct white noise
#prob_gen =0.017
#boradband whitenoise
#prob_gen = 0.2
#Natural Stim case
#prob_gen =0.05
#Use for anechoic pure tones
#prob_gen = 0.25
#Use for natural stim anechoic testset
#prob_gen = 0.125
#I think this was used in a previous anechoic case
#prob_gen =32.00
version = int(sys.argv[1])

if metadata_dict_mode:
    json_filename = glob(stim_paths+"/*.json")
    assert len(json_filename) == 1, "Only one JSON file supported"
    with open(json_filename[0],'r') as f:
        json_dict = json.load(f)

#slice array to parrallelize in slurm
low_idx = 110*version
high_idx= min(110*(version+1), len(stim_files))
stim_files_slice = stim_files[low_idx:high_idx]


def vary_itd(left_stim,right_stim,diff):
    left_roll = np.roll(left_stim,diff)
    return left_roll, right_stim

for s in stim_files_slice:
    stim_name = basename(s).split(".wav")[0]
    stim_rate, stim=read(s)

    msg = ("The sampling rate {}kHz does not match"
           "the declared value {}kHz".format(stim_rate,rate_in))
    assert stim_rate == rate_in, msg
    if len(stim.shape) > 1:
        print("{} is stereo audio".format(stim_name))
        continue

    #Zeros pad stimulus to avoid sound onset always being at the start of wave
    #files
    hann_stim =  hanning_window(stim,ramp_dur_ms,stim_rate)
    stim = hann_stim.astype(np.float32)

    #Changes stim sampling rate to match HRIR
    #nroutsamples = round(len(stim) * rate_out/rate_in)
    #stim_resampled = signal.resample(stim,nroutsamples)
    #gets filesnames for left and right channel HRIR
    for env in env_paths:
        hrirs_files = sorted(glob('{}/'.format(env)+filter_str+'*.wav'))
        #This should be 72 for 5 degree bins
        num_postitions = len(hrirs_files)/(36*7*2)
        class_balancing_factor = 1 if num_postitions < 1 else 4.0/num_postitions
        env_name_list = basename(env).split("_")
        room_geometry = env_name_list[3]
        room_materials = env_name_list[4]
        for i,(l, r) in enumerate(zip(hrirs_files[::2],hrirs_files[1::2])):
            if np.random.random() > prob_gen*class_balancing_factor:
                continue
            name_list = basename(r).split("_")
            elev = name_list[0].split("e")[0]
            azim = name_list[1].split("a")[0]
            head_location = name_list[2]
            channel = name_list[3].split(".")[0]
            #Reads in HRIRs
            hrir_r =read(r)[1].astype(np.float32)
            hrir_l =read(l)[1].astype(np.float32)
            #"spatializes" the sound, float64 return value. VERY loud. Do not play.
            conv_stim_r = signal.fftconvolve(stim,hrir_r)
            conv_stim_l = signal.fftconvolve(stim,hrir_l) 
            if vary_itd_flag:
                if not zero_pad:
                    raise NotImplementedError("Vary ITD only supported with zero padding")
                diff = np.random.randint(-25,25)
                conv_stim_l,conv_stim_r = vary_itd(conv_stim_l,conv_stim_r,diff)
            #Testing code
            if DEUBUG:
                name = "{}_{}elev_{}azim_convolved.mat".format(stim_name,elev,azim)
                name_with_path = out_path+name
                scipy.io.savemat(name_with_path,mdict={'arr': conv_stim_r})
            #Rescale to not blow out headphones/eardrums
            max_val = max(np.max(conv_stim_r),np.max(conv_stim_l))
            rescaled_conv_stim_r =conv_stim_r/max_val*scaling
            rescaled_conv_stim_l =conv_stim_l/max_val*scaling
            #converts to proper format for scipy wavwriter
            out_stim = np.array([rescaled_conv_stim_l, rescaled_conv_stim_r], dtype=np.float32).T
            if metadata_dict_mode:
                spatial_dict = {'elev':elev,'azim':azim,
                                'head_location':head_location,
                                'room_geometry': room_geometry,
                                'room_materials':room_materials}
                json_dict[stim_name].update(spatial_dict)
                name = "/audio/{}_{}elev_{}ax_{}_{}_{}.wav".format(stim_name,elev,azim,head_location,room_geometry,room_materials)
            else:
                name = "{}_{}elev_{}ax_{}_{}_{}.wav".format(stim_name,elev,azim,head_location,room_geometry,room_materials)
            if not i%1000:
                print(name)
            name_with_path = out_path+name
            write(name_with_path,rate_out,out_stim)

if metadata_dict_mode:
    json_filename = out_path+"/examples.json"
    with open(json_filename,'w') as f:
        json.dump(json_dict,f)
