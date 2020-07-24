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
rep_tim  = 0.7   #s

## Unrelevante Werte rausschmeißen
#index = np.where(imag < 20)
#
#real_off        = np.delete(real_off, index)
#time        = np.delete(time, index)
#real_off_err    = np.delete(real_off_err, index)


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
    signal = signal*np.exp(-0.5*(times*apo*2*np.pi)**2)
    #führe die FFT durch
    spektrum = np.fft.fft(signal)
    #führe die Phasenkorrektur durch
    spektrum = spektrum*np.exp(1j*phase0*np.pi/180.0)
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

apo = 40
freqs, realspek, imagspek = do_fft(time,real,imag,apo)
zero = np.where(freqs==0)
# finde kleinste Phase für Imag
imagspek0 = 20  #irgendein Startwert

for deg in range(0,360):
    freqs, realspek, imagspek = do_fft(time,real,imag,apo,deg)
    if(np.abs(imagspek[zero]) < imagspek0 ):
        imagspek0   = imagspek[zero]    # update, wenn kleiner
        kPhase      = deg

print("kleinste Phase = ",kPhase, "° mit ImagAmplitude = ", imagspek0)

freqs, realspek, imagspek = do_fft(time,real,imag,apo,kPhase)
freqs *= 1e-3    #kHz

# Plot
plt.figure(figsize=(10,5))

plt.plot(freqs, imagspek, "r-", label="imag")
plt.plot(freqs, realspek, "b-", label="real")
#plt.plot(np.ones(100)*2*rep_tim, np.linspace(-14,3,100))

#plt.yscale("log")
plt.xlabel(r"Frequenz / kHz")
plt.ylabel(r"Amplitude")
plt.axis([-20,20,-0.2, 1.1])
plt.legend()
#plt.show()
plt.savefig("Spek.pdf", dpi = 1000)
plt.clf()
