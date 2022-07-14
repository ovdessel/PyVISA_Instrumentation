
#initial source
#https://siglentna.com/application-note/16-bit-step-arb-sdgx/

#setting up ethernet 
#https://knowledge.ni.com/KnowledgeArticleDetails?id=kA03q000000x3gXCAQ&l=en-US


#import libraries 

import pyvisa as visa #instrument interface

#math
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fftpack import fft, fftfreq
import scipy.optimize

#logging
import logging
from datetime import date
import progressbar as pb
import pickle

logging.basicConfig(filename='Impedance_%s.log'%(date.today().strftime("%b-%d-%Y %H.%M.%S")),
                    format='%(levelname)s:%(message)s',
                    encoding='utf-8', level=logging.INFO)
#xvisa.log_to_screen()

rm = visa.ResourceManager()
fungen = rm.open_resource('TCPIP0::192.168.86.214::inst0::INSTR') #.214 is the function generator
oscope = rm.open_resource('TCPIP0::192.168.86.213::inst0::INSTR') #.213 is the oscope generator

oscope.chunk_size = 10000*1024 #Siglent wants to send all the data at once - aka giant ass chunk sizes
oscope.timeout = 5000 #ms long timeout period to help deal with network latency


#%% Subroutines


def tune_vdiv(channel):
    condition = True
    prev_max = 0
    divider = 3.75
    oscope.write(r"C%s:VDIV %fMV"%(channel,1500.0))
    while(condition):
        #find the division
        #tdiv = float(oscope.query(r"TDIV?").split(' ')[1][:-2]) #tdiv in seconds
        
        #Tune the channel - start coarse and work to higher precision
        #Set the division to X volts and record the offset
        voff  = float(oscope.query("C%s:OFFSET?"%(channel)).split(' ')[1][:-2])
        vdiv  = float(oscope.query("C%s:VDIV?"%channel).split(' ')[1][:-2])
        #oscope.write("TDIV %fS"%(tdiv))
        time.sleep(1) #let scope settle 
        
        #Collect course data - every 1000 point
        oscope.write("WFSU SP,25000,FP,0,NP,0")
        oscope.write("C%s:WF? DAT2"%(channel))
        response = oscope.read_raw()
        data = np.frombuffer(response[24:-2],dtype="int8")*(vdiv)/25.0+voff #convert bits to Volts
        #note that there is a data header of 24 bytes and 2 byte terminator
        
        #find the max 
        temp = []
        for n in np.array_split(np.abs(data),5):
            temp.append(np.max(n))   
        
        #Find the maximum and set the apropriate division
        max_v = np.ceil(np.average(temp)*1000.0 )#round to closest 1mV
        oscope.write(r"C%s:VDIV %FMV"%(channel,(max_v)/divider))
        vdiv  = float(oscope.query("C%s:VDIV?"%channel).split(' ')[1][:-2])
        
        #Quit loop once the max doesn't change more than XXmV
        if np.abs(max_v-prev_max) < 50.0 :
            condition = False
        else:
            prev_max = max_v
       
    return voff,vdiv,max_v

def data_collect(channel):
    try:
        #Collect data from a specific channel and convert binary to volts. 
        #Also perform preliminiary fft analysis to determine magnitude and phase
        vdiv = float(oscope.query("C%s:VDIV?"%(channel)).split(' ')[1][:-2])
        voff = float(oscope.query("C%s:OFST?"%(channel)).split(' ')[1][:-2])
        
        #collect data points. note that there are 24 header bytes and 2 termination bytes
        oscope.write("WFSU SP,0,FP,0,NP,0") #Get all the data points
        oscope.write("C%s:WF? DAT2"%(channel))
    
        data = np.frombuffer(oscope.read_raw()[24:-2],dtype="int8")*vdiv/25.0+voff
    except:
        return np.array([]),np.nan,np.nan,np.nan,np.nan
    #transform to frequency domain
    data_fft = fft(data)
    dxf = fftfreq(len(data_fft),samp_int)
    
    #find the max bin and calculate magnitude and phase
    bin_d = np.argmax(np.abs(data_fft))
    dmag = np.absolute(data_fft[bin_d])/len(data) * 2.0
    dphase = np.angle(data_fft[bin_d],deg=True)
    
    #return parameters
    return data,data_fft,dxf,dmag,dphase

        
        
#%% MAIN

#configure function generator to output a sine wave at 2v pp
fungen.write(r"C1:BSWV WVTP,SINE,FRQ,2000HZ,AMP,2V,OFST,0V")
fungen.write(r"C1:OUTP ON") #Turn on channel

       

#get sample rate for fft analysis
sampr = float(oscope.query("SARA?").split(' ')[1][:-5])
samp_int = 1.0/sampr
oscope.write(r"BUZZ ON")
oscope.write(r"ACQW AVERAGE,64") #average the data

#Set trigger
oscope.write(r"C1:TRig_LeVel 0V")
oscope.write(r"TRMD NORM")
#oscope.write(r"C1: OFST -2V")
inductor_data = []

coarse_num = 20
fine_num = 20

#2,5
#3.4,4

#with cap
#2,4
#2.5,3.5

#100uh
# 4,7
# 5.5,6.5
for frequencies in [np.ceil(np.logspace(3,7.25,coarse_num)), #1.5 is minimum , 30MHz is maximum
                    #np.ceil(np.logspace(2.5,3.5,fine_num)),
                    #np.ceil(np.geomspace(300,1500,15))
                    ]:
    pbar = pb.ProgressBar(max_value=len(frequencies))
    for i,freq in enumerate(frequencies):
        pbar.update(i)
        
        print("Testing %f kHz"%(freq/1000.0))
        
        logging.debug("Test %f frequency"%(freq))
        
        #Set frequency
        fungen.write(r"C1:BSWV WVTP,SINE,FRQ,%fHZ,AMP,5V,OFST,0V"%(freq))
        fungen.write(r"C1:OUTP ON")
        
        #Atleast 5 cycles 1/f = div 14 divisions
        oscope.write("TDIV %fUS"%(1.0/freq*5.0/14.0*1e6))
        
        #select the appropriate division for the scope to use as much range as possible
        #Start with a guess of 1500mV / div - then fine tune once scaled appropriately 
        voff_1,vdiv_1,vmax_1 = tune_vdiv(1)
        voff_2,vdiv_2,vmax_2 = tune_vdiv(2)
        
        time.sleep(5.0) # give some time to settle especially with the averaging
            
        temp1m = []
        temp2m = []
        temp1p = []
        temp2p = []
        for i in range(0,1):
            #data,data_fft,dxf,dmag,dphase
            #print("%d"%(i))
            _,_,_,c1_mag,c1_phase = data_collect(1)
            _,_,_,c2_mag,c2_phase = data_collect(2)
            
            temp1m.append(c1_mag)
            temp2m.append(c2_mag)
            temp1p.append(c1_phase)
            temp2p.append(c2_phase)
            
            #time.sleep(0.5)
            
        inductor_data.append([temp1m,
                              temp1p,
                              temp2m,
                              temp2p,
                              freq])
pickle.dump(inductor_data,open("inductor_data.p","wb"))
    
#%%

inductor_data = pickle.load( open( "inductor_data.p", "rb" ) )
phase = []
imp  = []
vm = []
im = []
frequencies = []
shunt_r = 99.1 #1064.0
for dat in inductor_data:
    frequencies.append(dat[4])
    p1 = np.nanmean(dat[1])
    p2 = np.nanmean(dat[3])
    
    phase.append(p1-p2)
    ind_v = np.nanmean(dat[0])-np.nanmean(dat[2]) #get inductor voltage by subtracting shunt channel
    ind_c = np.nanmean(dat[2])/shunt_r
    
    vm.append(ind_v)
    im.append(ind_c)
    imp.append(ind_v/ind_c)
   
#convert to numpy array
frequencies = np.array(frequencies)
vm = np.array(vm)
im = np.array(im)
imp = np.array(imp)
phase = np.array(phase)
sort_indx = np.argsort(frequencies)


Ls = []
for n in sort_indx:
    Ls.append((imp[n]-shunt_r)/(2*np.pi*frequencies[n]))
Ls = np.array(Ls)
L_avg = np.mean(Ls[8:18])
L_std = np.std(Ls[8:18])


#Plot stuff
fig, axs = plt.subplots(3)
fig.suptitle('Impedance and Phase plot')
axs[0].plot(frequencies[sort_indx], imp[sort_indx],'.-')
axs[1].plot(frequencies[sort_indx], phase[sort_indx],'.-',c='r')
axs[2].plot(frequencies[sort_indx],Ls,'.-',c='g')

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[1].set_xscale('log')
axs[2].set_xscale('log')

axs[0].grid("both","both")
axs[1].grid("both","both")
axs[2].grid("both","both")

axs[0].set_ylabel("Mangitude (Z)")
axs[2].set_xlabel("Frequency (Hz)")
axs[2].set_ylabel("Inductance (H)")

axs[1].set_yticks((-90,-60,-30,0,30,60,90))
axs[1].set_ylabel("Phase (deg)")


def Creactance(f,C):
    return 1.0/(2.0*np.pi*f*C)

def Ireactance(f,L):
    return 2.0*np.pi*f*L

#select data range
ind_points = [2,11]
cap_points = [32,37]

#get data and frequencies from sorted data
cap_data = imp[sort_indx][cap_points[0]:cap_points[1]]
cap_freq = frequencies[sort_indx][cap_points[0]:cap_points[1]]
axs[0].plot(cap_freq,cap_data,'.-')

ind_data = imp[sort_indx][ind_points[0]:ind_points[1]]
ind_freq = frequencies[sort_indx][ind_points[0]:ind_points[1]]
axs[0].plot(ind_freq,ind_data,'.-')

#Find best fit
C_popt, C_pcov  = scipy.optimize.curve_fit(Creactance,cap_freq,cap_data,p0=(1e-6))
I_popt, I_pcov = scipy.optimize.curve_fit(Ireactance,ind_freq,ind_data)

# plot our lines
I_range = np.logspace(1,4)
C_range = np.logspace(3,6)
axs[0].plot(I_range,Ireactance(I_range,I_popt[0]))
axs[0].plot(C_range,Creactance(C_range,C_popt[0]))
#%%

C = C_popt[0]-20e-12*2.0 #C_popt[0] remove 20pF from each probe (2 probes)
ICb = 2*np.pi*frequencies*C*vm

IRLb = np.sqrt((im*np.cos(np.deg2rad(phase)))**2.0+ \
               (ICb+im*np.sin(np.deg2rad(phase)))**2)

ab = np.arctan((ICb+im*np.sin(np.deg2rad(phase)))/ \
                (   im*np.cos(np.deg2rad(phase))))
    
VLb = np.abs(vm*np.sin(ab))
Lb = VLb/(IRLb*2*np.pi*frequencies)

axs[2].plot(frequencies[sort_indx],Lb[sort_indx])

print("L: %fH\nC: %fnF"%(I_popt[0],C_popt[0]*1e9))

SRF = 1.0/(2*np.pi*np.sqrt(I_popt[0]*C))
print("SRF: %02g kHz"%(SRF/1000))

imp_max = np.argmax(imp[sort_indx])
print("Measured max: freq = %02g kHz imp = %02g"%(frequencies[sort_indx][imp_max]/1000,imp[sort_indx][imp_max]))



plt.show()


#%%