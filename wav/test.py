import librosa as lbr
import pdb

"""
    D:np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
    STFT matrix
"""

if __name__ == "__main__":
    y, sr = lbr.load("TM3_100162.wav", sr=None)

    n_fft = 400
    stft = lbr.core.stft(y, n_fft=n_fft, hop_length=int(n_fft/5), window='hann')

    y_reconstruct = lbr.core.istft(stft, hop_length=int(n_fft/5), window='hann')
    lbr.output.write_wav("istft.wav", y_reconstruct, sr=sr)

    print(stft)
    pdb.set_trace()
