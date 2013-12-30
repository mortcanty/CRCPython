#!/usr/bin/env python
#  Name:     ex4_1.py
from numpy import *
import matplotlib.pyplot as plt

def chirp(t,t0):
    result = 0.0*t
    idx = array(range(2000))+t0 
    tt = t[idx] - t0
    result[idx] = sin(2*math.pi*2e-3*(tt+1e-3*tt**2))
    return result

def main(): 
    t = array(range(5000))
    plt.plot(t,chirp(t,400)+9)
    plt.plot(t,chirp(t,800)+6)
    plt.plot(t,chirp(t,1400)+3)
    signal = chirp(t,400)+chirp(t,800)+chirp(t,1400)
    kernel = chirp(t,0)[:2000]
    kernel = kernel[::-1]
    plt.plot(t,signal)
    plt.plot(0.003*convolve(signal,kernel,\
                                     mode='same')-5)
    plt.xlabel('Time')
    plt.ylim((-8,12))
    plt.show() 

if __name__ == '__main__':
    main()     