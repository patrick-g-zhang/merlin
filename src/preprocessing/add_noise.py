from scipy.io.wavfile import read,write
import numpy as np
import glob
import os
def wgn(x, snr):
    x = x.astype(np.float64)
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)
def add_noise(input_file, output_file,snr):
    audio_path=input_file
    fs,audio = read(audio_path)
    noise = wgn(audio,snr)
    audio_with_noise=noise+audio
    write(output_file,fs,audio_with_noise.astype(np.int16))

if __name__ == '__main__':
    clean_wav_path = "/home/gyzhang/merlin/egs/kingtts/s3/database/wav/"
    noise_wav_path = "/home/gyzhang/merlin/egs/kingtts/s3/database/noise_wav/"
    snr=0
    os.makedirs(noise_wav_path,exist_ok=True)
    for wav_path in glob.glob(clean_wav_path+'*.wav'):
        basename = os.path.basename(wav_path)
        out_path = os.path.join(noise_wav_path,basename)
        add_noise(wav_path,out_path,snr)