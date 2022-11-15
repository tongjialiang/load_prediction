import os
import sys
import platform
import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as mp

y=[0.1,0.5,0.7,-0.2,-0.6,0.9,0.2,0.5,0.7,0.2]
N=len(y)
x=np.arange(len(y))
y_fft=nf.fft(y)
freqs = nf.fftfreq(len(y), d=1)
print(y)
print(y_fft)
print(freqs)
print(np.fft.ifft(y_fft).real)

y_fft_my=[]
for k in range(N):
    res=0
    for n in range(N):
        res+=((np.exp(complex(0,-2*np.pi*n*k/N)))*y[n])
    y_fft_my.append(res)
print(y_fft_my)

y_ifft=[]
for n in range(N):
    res=0
    for k in range(N):
        res += ((np.exp(complex(0, 2 * np.pi * n * k / N))) * y_fft_my[k])/N
    y_ifft.append(res)
print(y_ifft)

