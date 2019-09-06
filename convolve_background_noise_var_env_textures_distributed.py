import numpy as np
import scipy.io
from glob import glob
from scipy.io.wavfile import read, write
from scipy import signal
from os.path import basename
import nnresample
import sys
import pdb
import json
from collections import defaultdict

DEUBUG = False
out_path = "/nobackup/scratch/Wed/francl/bkgd_textures_out_sparse_sampled_same_texture_expanded_set_44.1kHz/"
env_paths = sorted(glob("/om/user/francl/Room_Simulator_20181115_Rebuild/HRIRdist140-5deg_elev_az_*"))
version = int(sys.argv[1])
hrir_path = env_paths[version]
hrir_name = basename(hrir_path)
walls = hrir_name.split("_")[3]
materials = hrir_name.split("_")[4]
#out_path = "./pure_tones_out/"
rate_out=44100  #Desired sample rate of "spatialized" WAV
rate_in=44100   #Sample Rate for Stimuli

background_files = glob("/nobackup/scratch/Wed/francl/STSstep_v0p2/_synths/*/*.wav")   #File path to background sounds
#hrirs_files = sorted(glob("./JayDesloge/HRIR_elev_az_44100Hz_vary_loc/*{}x{}y{}z*.wav".format(x,y,z)))
MAXVAL = 32767.0
scaling = 0.1
NUM_BKGDS = 10000
NUM_POSITIONS =504

array_write = False


resampled_stim_dict = defaultdict(list)
for i,s in enumerate(background_files):
    stim_rate, stim=read(s)
    
    if stim.dtype != np.float32:
        raise ValueError("Stim not float32 format! Normalization incorrect.")
    stim_name = basename(s).split(".")[0]
    stim_key = "".join(stim_name.split('_')[:3])
    #Changes stim sampling rate to match HRIR
    #nroutsamples = round(len(stim) * rate_out/rate_in)
    #stim_resampled = signal.resample(stim,
    #                                 nroutsamples).astype(np.float32,casting='unsafe',)
    if rate_out != stim_rate:
        stim_resampled = nnresample.resample(stim,rate_out, stim_rate).astype(np.float32)
        pdb.set_trace()
    else:
        stim_resampled = stim
    resampled_stim_dict[stim_key].append((stim_resampled,background_files[i]))
    if not i%100:
        print("Resampling...{} complete".format(i))

hrirs_files = sorted(glob("{}/*.wav".format(hrir_path)))
locs = list(set([filename.split("_")[-2] for filename in hrirs_files]))
resampled_stim_array_keys = list(resampled_stim_dict.keys())
noise_index_array = []
#assuming a left and right channel for every source
for i in range(NUM_BKGDS):
    num_sampled_pos = np.random.randint(3,8)
    source_class = np.random.randint(0,len(resampled_stim_dict.keys()))
    num_sources_in_class = len(resampled_stim_dict[resampled_stim_array_keys[source_class]])
    source_idxs = np.random.choice(num_sources_in_class,NUM_POSITIONS,replace=False)
    pos_idx = np.random.choice(NUM_POSITIONS,num_sampled_pos,replace=False)
    selected_vals = np.full(pos_idx.shape+(2,) , source_class)
    selected_vals[:,1] = source_idxs[pos_idx]
    mask = np.full(source_idxs.shape+(2,),np.nan)
    mask[pos_idx] = selected_vals 
    noise_index_array.append(mask)

#gets filesnames for left and right channel HRIR
metadata_dict = {}
for count,noise_idxs in enumerate(noise_index_array):
    total_waveform = []
    total_waveform_labels = []
    hrirs_files = sorted(glob("{}/*{}*.wav".format(hrir_path,locs[count%len(locs)])))
    for i,(l, r,idx) in enumerate(zip(hrirs_files[::2],hrirs_files[1::2],noise_idxs)):
        if np.isnan(idx[0]):
            continue
        else:
            source_class = resampled_stim_array_keys[int(idx[0])]
            idx = int(idx[1])
        name_list = basename(r).split("_")
        elev = name_list[0].split("e")[0]
        azim = name_list[1].split("a")[0]
        channel = name_list[3].split(".")[0]
        #Reads in HRIRs
        hrir_r =read(r)[1].astype(np.float32)
        hrir_l =read(l)[1].astype(np.float32)
        #Grab correct noise sample
        noise,stim_name = resampled_stim_dict[source_class][idx]
        #"spatializes" the sound, float64 return value. VERY loud. Do not play.
        conv_stim_r = signal.convolve(noise,hrir_r).astype(np.float32)
        conv_stim_l = signal.convolve(noise,hrir_l).astype(np.float32) 
        #Testing code
        if DEUBUG:
            name = "{}_{}elev_{}azim_convolved.mat".format(stim_name,elev,azim)
            name_with_path = out_path+name
            scipy.io.savemat(name_with_path,mdict={'arr': conv_stim_r})
        total_waveform.append([conv_stim_l,conv_stim_r])
        total_waveform_labels.append([azim,elev,stim_name])
    summed_waveform = np.sum(np.array(total_waveform),axis=0)
    ###This is where you stopped### Need to test array summation and listen to waveforms
    #Rescale to not blow out headphones/eardrums
    max_val = np.max(summed_waveform)
    rescaled_summed_waveform = summed_waveform/max_val*scaling
    if array_write:
        name = "noise_{}_spatialized_{}sr.npy".format(count,rate_out)
        name_with_path = out_path+name
        print(name)
        np.save(name_with_path,resampled_stim_array)
    else:
        #converts to proper format for scipy wavwriter
        name = "noise_{}_spatialized_{}_{}_{}.wav".format(count,locs[count%len(locs)],walls,materials)
        metadata_dict[name] = total_waveform_labels
        name_with_path = out_path+name
        out_stim = np.array(rescaled_summed_waveform, dtype=np.float32).T
        print(name)
        write(name_with_path,rate_out,out_stim)
        
json_name = out_path + "label_meatadata_{}_{}_{}.json".format(locs[count%len(locs)],walls,materials)
with open(json_name, 'w') as fp:
        json.dump(metadata_dict,fp)


