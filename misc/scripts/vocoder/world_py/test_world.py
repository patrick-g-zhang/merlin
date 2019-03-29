# test difference between python and bash generated features
import os
import sys
import shutil
import glob
import time
import multiprocessing as mp
import numpy as np
import pyworld
import pysptk
import librosa
# initializations
fs = 16000
if fs == 16000:
    nFFTHalf = 1024
    alpha = 0.41

elif fs == 22050:
    nFFTHalf = 1024
    alpha = 0.455

elif fs == 44100:
    nFFTHalf = 2048
    alpha = 0.76

elif fs == 48000:
    nFFTHalf = 2048
    alpha = 0.77

else:
    print("As of now, we don't support %d Hz sampling rate." %(fs))
    print("Please consider either downsampling to 16000 Hz or upsampling to 48000 Hz")
    sys.exit(1)

#bap order depends on sampling rate.
mcsize=59

def read_binfile(filename, dim=60, dtype=np.float64):
    '''
    Reads binary file into numpy array.
    '''
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=dtype)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return  m_data

def write_binfile(m_data, filename, dtype=np.float64):
    '''
    Writes numpy array into binary file.
    '''
    m_data = np.array(m_data, dtype)
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return

def load_binary_file_frame(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    frame_number = features.size // dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return  features, frame_number

def process(filename):
    '''
    The function decomposes a wav file into F0, mel-cepstral coefficients, and aperiodicity
    :param filename: path to wav file
    :return: .lf0, .mgc and .bap files
    '''

    file_id = os.path.basename(filename).split(".")[0]
    print('\n' + file_id)

    ### WORLD ANALYSIS -- extract vocoder parameters ###
    ### extract sp, ap ###
    x, fs = librosa.core.load(filename, sr=16000)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.harvest(x, fs, frame_period=5, f0_floor=71.0, f0_ceil=700)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
    f0 = f0[:, None]
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    mc = pysptk.sp2mc(spectrogram, mcsize, alpha=alpha)
    bap = pyworld.code_aperiodicity(aperiodicity, fs)
    bin_lf0 = read_binfile("1.lf0",dim=1)
    bin_bap = read_binfile("1.bap",dim=1)
    bin_mc = read_binfile("1.mgc", dim=60)
    bin_lf0_2,_ = load_binary_file_frame("1.lf0", dimension=1)
    bin_bap_2,_ = load_binary_file_frame("1.bap", dimension=1)
    bin_mc_2,_ = load_binary_file_frame("1.mgc", dimension=60)
    pass
    ### convert bapd to bap ###

process("1.wav")

read_binfile("lc.f0")