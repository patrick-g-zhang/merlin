"""
This is program written for normalization loudness or RMS of each wav file
mainly for storytelling corpus which we assume each story should be normalized to the same rms level
"""
import librosa
import numpy as np
import glob
wav_file_path = "/home/patrick/corpous/cantonese audiobook/children/WAV/三只羊.wav"
fs = 16000
# for wav_file_path in glob.glob("/home/patrick/corpous/BZNSYP/Wave/*.wav"):
y,sr = librosa.core.load(wav_file_path,sr=fs)
y = y.astype(np.float64)
root_mean_square = np.sqrt(np.sum(np.square(y))/len(y))
root_mean_square_db = 20*np.log10(root_mean_square)
print("{0},{1}".format(root_mean_square,root_mean_square_db))