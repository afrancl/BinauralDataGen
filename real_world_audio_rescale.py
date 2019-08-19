import numpy as np
from hanning_window import hanning_window
import scipy.io
from glob import glob
from scipy.io.wavfile import read, write
from scipy import signal
from os.path import basename
import sys
import pdb

out_path="/om/user/francl/recorded_binaural_audio_office_0elev_rescaled/"

#stim_files = glob("/om/user/gahlm/sorted_sounds_dataset/sorted_stimuli_specfilt/*.wav")    #File path to stimuli
stim_files = glob("/om/user/francl/recorded_binaural_audio_office_0elev/*/*.wav")    #File path to stimuli

#scales loudness of sounds DO NOT CHANGE
scaling = 0.1


##slice array to parrallelize in slurm
#version = int(sys.argv[1])
#low_idx = 110*version
#high_idx= min(110*(version+1), len(stim_files))
#stim_files_slice = stim_files[low_idx:high_idx]

for i,s in enumerate(stim_files):
    azim_loc = s.split("/")[-2]
    stim_name = basename(s).split(".wav")[0]
    stim_rate, stim=read(s)

    #transpose to split L/R channels
    stim = stim.T
    stim_l = stim[0]
    stim_r = stim[1]

    #Rescale to not blow out headphones/eardrums
    max_val = max(np.max(stim_r),np.max(stim_l))
    rescaled_stim_r =stim_r/max_val*scaling
    rescaled_stim_l =stim_l/max_val*scaling

    #converts to proper format for scipy wavwriter
    out_stim = np.array([rescaled_stim_l, rescaled_stim_r], dtype=np.float32).T
    name = "{}_{}.wav".format(stim_name,azim_loc)
    if not i%1000:
        print(name)
    name_with_path = out_path+name
    write(name_with_path,stim_rate,out_stim)

