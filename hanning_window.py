import numpy as np
from math import *


def hanning_window(stim, ramp_duration_ms, SAMPLERATE=48000):
    stim_np = np.array(stim)
    stim_dur_smp = stim_np.shape[0]
    ramp_dur_smp = floor(ramp_duration_ms*SAMPLERATE/1000)
    hanning_window =  np.hanning(ramp_dur_smp*2)
    onset_win = stim_np[:ramp_dur_smp]*hanning_window[:ramp_dur_smp]
    middle = stim_np[ramp_dur_smp:stim_dur_smp-ramp_dur_smp]
    end_win = stim_np[stim_dur_smp-ramp_dur_smp:]*hanning_window[ramp_dur_smp:]
    windowed_stim = np.concatenate((onset_win,middle,end_win))
    return windowed_stim
