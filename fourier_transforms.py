#!/usr/bin/env python
# coding: utf-8

# ## FFT 1

# #### The arapaly.txt file contains data about the fluctuations of the sea level at the nuclear power plant in Haysham, UK.
# 
# #### Is there a dominant frequency? If yes, what kind of dominant frequency do we expect?
# #### Plot the Fourier-transformed signal! Is the dominant frequency visible in the frequency spectrum?

# In[15]:


# Importing required environments
get_ipython().run_line_magic('pylab', 'inline')
from numpy.fft import *

ds = loadtxt("arapaly.txt")
plot(ds[:288,0], ds[:288,1])
xlabel("Days", size=12)
ylabel("Height difference from average sea level [m]", size=10)
title("Height difference from average sea level with respect to days", size=18, y=1.05) # Title and labels added to plot


# In[14]:


# FFT signal
Fsignal = fft(ds[:,1] - mean(ds[:, 1])) # The mean is subtracted, so the 0 Hz ffrequency does not disturb
freq = fftfreq(len(ds[:,0]), d=ds[1,0] - ds[0,0])

print("Dominant frequency: {} 1/day".format(freq[abs(Fsignal).argmax()])) # Maxima is extracted from transformed dataset
plot(abs(freq), abs(Fsignal), label="Transformed signal")
plot(freq[abs(Fsignal).argmax()], max(abs(Fsignal)), "ro", label="The maximum is at 1,93 1/day")
xlabel("Frequency", size=12)
ylabel("Transformed signal strength", size=12)
title("FFTed signal", size=18, y=1.05)
legend()


# Due to the gravitational attraction of celestial bodies and the orbit of the Earth, the equipotential surfaces in the vicinity of the Earth are not perfectly spherical (minor perturbation). The large water bodies attempt to stay on these equipotential surfaces, but since the Earth is rotating an observer on Earth will see the sea level going up and down periodically, so we expect a dominant frequency.
# 
# This is the basic mechanism of high and low tide observed at different times in the day. Therefore, according to this the dominant frequency is around 12 hours (most dominant frequency spectrum).
# 
# The dominant frequency from FFT is  $1,93 \frac{1}{day}$ , which corresponds to a periodicity of $12$ hours and $25$ minutes. The main reason for high/low tides is the rotation of the Earth, however the Moon perturbes this process and contributes to the difference from exact $12$ hours.
# .
# 
# 

# ## FFT 2

# ### Using the ifft (inverse FFT) function we can obtain the inverse FT of a function. By using this, filter the components higher than 20Hz from the audio file

# In[9]:


ds = loadtxt("zenebona.txt")
plot(ds[:,0], ds[:,1])
xlabel("Time", size=12)
ylabel("Signal strength", size=12)
title("Signal strength with respect to time", size=18, y=1.05)


# In[11]:


Fsignal = fft(ds[:,1]) # Applied FFT to dataset
freq = fftfreq(len(ds[:,0]), d=ds[1,0] - ds[0,0])
plot(abs(freq), abs(Fsignal), label="before")

for i in range(len(Fsignal)):
    if abs(freq[i]) > 20:
        Fsignal[i] = 0 # Creating our virtual filter

        
plot(abs(freq), abs(Fsignal), label="after")
title("The Fourier-transformed signal before and after applying the 20Hz filter", size=18, y=1.05)
xlabel("Frequency", size=12)
ylabel("Transformed signal-strength", size=12)
legend()


# In[12]:


filtered = ifft(Fsignal)
plot(ds[:,0], filtered)
xlabel("Time", size=12)
ylabel("Signal Strenght", size=12)
title("Signal strength with respect to time in filtered signal")


# 
