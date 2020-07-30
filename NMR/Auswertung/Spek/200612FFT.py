#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


#times: Zeitachse des gemessenen Zeitsignals (aus .ts Datei, 1. Spalte)
#real: Realteil des Zeitsignals (2. Spalte)
#imag: Imaginärteil des Zeitsignals (4. Spalte)
#apo: Apodisation, breite der Fensterfunktion zum Glätten des Spektrums.
#phase0: Phasenkorrektur 0. Ordnung (falls die Phase des Spektrums nicht perfekt passt)
#Wichtig: das Signal sollte genau bei dem Echo anfangen, also bei einem Solid Echo auf dem Maximum,
#an dieser STelle sollte wenn die Phase perfekt gewählt ist der Imaginärteil 0 und der Realteil maximal sein.
#Die Punkte davor können verworfen werden, es können am Ende des Signals weitere Punkte mit 0 angehängt werden
#damit die Gesamtzahl der Punkte 2^n mit der ganzen Zahl n ist (das bevorzugt der FFT Algorithmus).
x = np.genfromtxt('DMSO2_T2_0K_38327_1.ts', unpack=True)
time     = x[0]
real     = x[1]
imag     = x[2]

# schneide alles weg vor real_max weg
index_real_max = np.argmax(abs(real))
time = time[0:len(time)-index_real_max]
real = real[index_real_max::]
imag = imag[index_real_max::]
# Offset wegmachen
index_zero = np.where(time > 0.00035)
mean_real = np.mean(real[index_zero])
mean_imag = np.mean(imag[index_zero])
real -= mean_real
imag -= mean_imag
# setze Werte nach 0.00035s auf Null
real[index_zero] = 0
imag[index_zero] = 0
# Um Nullen nach dem Signal ergänzt
#dt = time[1]-time[0]
#ext = np.arange(time[len(time)-1],5,dt)
#time = np.append(time,ext)
#real = np.append(real, np.zeros(len(ext)))
#imag = np.append(imag, np.zeros(len(ext)))


Phi = np.arctan(imag[0]/real[0])
Phi =(Phi*180)/np.pi             # in degree
Phi += 180                       # Phasenverschieung korrigieren
print("Phi = ",Phi,"°")

# Plot
plt.figure(figsize=(10,5))
plt.plot(time, real, "b-", label="Realteil")
plt.plot(time, imag, "r-", label="Imaginärteil")
plt.axis([0,0.0003, -4200, 1000])
plt.legend()
plt.savefig("FID.pdf", dpi = 1000)
#plt.show()
plt.clf()

def do_fft(times, real, imag, apo, phase0 = 0):
    #Abtastrate bestimmen
    deltat = abs(times[1]-times[0])
    #Frequenzachse generieren
    freqs = np.fft.fftfreq(len(times), deltat)
    #Realteil und imaginärteil zu einer komplexen Zahl zusammen setzen
    signal = real + 1j*imag
    #Der erste Punkt einer FFT muss halbiert werden, sonst entsteht ein Offset Fehler
    signal[0] = signal[0]/2.0
    #Eine Apodisation ist eine Faltung des Spektrums mit einer Fensterfunktion (hier eine Gauß Funktion)
    #eine Faltung im Frequenzraum ist im Zeitgebiet nur eine Multiplikation mit der Fourier Transformierten
    #die im Falle einer Gaußfunktion wieder eine Gauß Funktion ist.
    signal = signal*np.exp(-0.5*(times*apo*2*np.pi)**2)     # 1-2kHz
    #führe die FFT durch
    spektrum = np.fft.fft(signal)
    #führe die Phasenkorrektur durch
    spektrum = spektrum*np.exp(1j*phase0*np.pi/180.0)   # Phase0 in degree
    #ordne das Spektrum so um, dass die 0 in der Mitte ist
    freqs = np.fft.fftshift(freqs)
    spektrum = np.fft.fftshift(spektrum)
    #Trenne Real- und Imaginärteil des Spektrums
    realspek = np.real(spektrum)
    imagspek = np.imag(spektrum)

    #Normiere das Spektrum, hier so dass das Maximum des Realteil 1 ist.
    max = np.max(realspek)
    realspek = realspek/max
    imagspek = imagspek/max

    #gebe Frequernzen, Real- und Imaginärteil zurück.
    return freqs, realspek, imagspek

apo = 1e3
freqs, realspek, imagspek = do_fft(time,real,imag,apo,Phi)
freqs *= 1e-3   #in kHz

# beiden Extrema der real
x_0         = int(len(real)/2)
real_max1   = np.argmax(abs(realspek[0:x_0]))
real_max2   = np.argmax((realspek[x_0::])) + x_0
real_dx     = abs(freqs[real_max1] - freqs[real_max2])
real_half    = 0.5*abs(realspek[real_max1]-realspek[real_max2]) + realspek[real_max2]

# beiden Extrema der imag
imag_max1   = np.argmax(abs(imagspek[0:x_0]))
imag_max2   = np.argmax(abs(imagspek[x_0::])) + x_0
imag_dx     = abs(freqs[imag_max1] - freqs[imag_max2])
imag_half    = 0.5*abs(imagspek[imag_max1]-imagspek[imag_max2]) + imagspek[imag_max2]

# plot
plt.plot(freqs, realspek, "b-", label="Realteil", zorder=0)
plt.plot(freqs, imagspek, "r-", label="Imaginärteil", zorder=0)
# Abstand der beiden real Peaks
plt.arrow(freqs[real_max1],real_half, real_dx-2,0,head_width=0.05,head_length=2,fc="black", zorder=1)
plt.arrow(freqs[real_max2],real_half, -real_dx+2,0,head_width=0.05,head_length=2,fc="black", zorder=1)
plt.text(-8, real_half+0.05, str(round(real_dx,2))+' kHz')
# Abstand der beiden imag Peaks
plt.arrow(freqs[imag_max1],imag_half, imag_dx-2,0,head_width=0.05,head_length=2,fc="black", zorder=1)
plt.arrow(freqs[imag_max2],imag_half, -imag_dx+2,0,head_width=0.05,head_length=2,fc="black", zorder=1)
plt.text(-8, imag_half-0.1, str(round(imag_dx,2))+' kHz')

plt.axis([-80,80,-1,1.1])

plt.xlabel(r"Frequenz / kHz")
plt.ylabel(r"Amplitude")
plt.legend()
#plt.show()
plt.savefig("Spek.pdf", dpi = 1000)
plt.clf()
