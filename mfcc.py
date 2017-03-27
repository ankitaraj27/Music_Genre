import numpy
import scipy
import math
import scipy.io.wavfile as wavread
from scipy.fftpack import dct

def filterbank(nfilt=26,nfft=661,samplerate=22050,lowfreq=0,highfreq=None):
    highfreq=samplerate/2
    lowmel= 2595 * numpy.log10(1 + lowfreq/700.0)
    highmel= 2595 * numpy.log10(1 + highfreq/700.0)
    points = numpy.linspace(lowmel,highmel,nfilt+2)

    bin = numpy.floor((nfft+1)*(700*(10**(points/2595.0)-1))/samplerate)

    fbank = numpy.zeros([nfilt,nfft//2+1])

    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i- bin[j])/(bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra, L=22):
    if L>0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.0)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        return cepstra

def my_mfcc(fn):
    sample_rate,X = wavread.read(fn)
    N = len(X)
    window_size = 0.03 #30ms
    samples = math.floor(sample_rate*window_size)    #size of each frame
    samples = int(samples)
    start=0
    end=samples
    w = []
    for i in range (0,samples):
        w.append(0.54-(0.46*math.cos((2*math.pi*i)/(samples-1))))

    energy = []
    feat_all = []
    fb = filterbank()
    power_all = []

    for i in range(0,N,2*samples/3):
        temp= X[start:end]
        if len(w) != len(temp) :
            temp1 = numpy.zeros((len(w)-len(temp)),)
            temp = numpy.concatenate([numpy.array(temp),numpy.array(temp1)])
        
        temp = numpy.array(temp)*numpy.array(w)
        temp_fft = numpy.fft.fft(temp)
        power_old = numpy.abs(temp_fft)**2
        power_old = power_old/samples
        power = power_old[0:331]
        start=end-samples/3
        end=start+samples
        power_all.append(power)
        #energy.append(numpy.sum(power))
        #print temp_fft
        #print power
        #print len(power)
        #print temp
        feat = numpy.dot(power,fb.T)
        feat = numpy.where(feat == 0, numpy.finfo(float).eps,feat)
        feat = numpy.log(feat)
        feat_all.append(feat)
    energy = numpy.sum(power_all,1)
    energy = numpy.where(energy == 0.0, numpy.finfo(float).eps,energy)
    feat_all = dct(feat_all, type=2, axis=1, norm='ortho')[:,:13]
    feat_all=lifter(feat_all,22)
    feat_all[:,0] = numpy.log(energy)
    #print numpy.mean(feat_all,axis=0)
    return feat_all



  
#fn="C:\Users\RAJ\Desktop\matlab\genres_wav\jazz\jazz.00087.wav"
