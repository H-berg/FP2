import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math

plt.rc('text', usetex=True)
phi, real, real_err, imaginary, imaginery_err, real_off, real_off_err, imaginary_off, imaginary_off_err, sample_temperature = np.genfromtxt('winkel.dat.nmr', unpack=True)

# fiiting klappt irgendwie nicht..
x = np.linspace(0,350, 1000)
def sinus(A,b,c,d,x):
    return A*np.sin(b*x+c)+d

offset  = max(real_off) + min(real)
offset *= 0.5
Amp     = max(real_off) - offset
freq    = 2*np.pi / max(phi)
phase   = -np.pi/2
#Normierung
real_off_err /= abs(max(real_off)-min(real_off))
real_off     /= abs(max(real_off)-min(real_off))



print(phase)


p0 = np.array([Amp, freq, phase, offset])

#phi *= np.pi/180 # in rad

#params, covariance_matrix = curve_fit(sinus, phi, real, p0)
#print((max(real) - min(real))*0.5)

i = np.argmax(real_off)





plt.plot(185*np.ones(10), np.linspace(min(real_off)-0.2, max(real_off)+0.2, 10), "--", color="#e17c16", label="gewählte Phase")
plt.errorbar(phi, real_off, yerr=real_off_err, capsize=3, fmt='.', color="#505054",label="Echo")
plt.axis([-5,355,-0.62,0.46])
plt.xlabel(r"Phase $\varphi$ / °")
plt.ylabel("Amplitude (Realanteil)")
#plt.plot(phi, sinus(phi, *params), label = "fit")
#plt.plot(phi, sinus(*p0, phi), label = "sinus mit p0")
#plt.plot(phi, imaginary, "bx", label = "Imaginär")
#plt.plot(phi, real + real_off, "mx", label= "Real mit off")
#plt.plot(phi, imaginary + imaginary_off, "gx", label = "Imaginär mit off")
plt.legend()
plt.savefig("winkel.pdf")
#plt.show()
plt.clf()

# Ausschnitt sieht nicht so top aus

#i = np.argmax(real_a)

#plt.errorbar(phi_a, real_a + real_err_a, yerr=real_err_a, fmt='.', label="Messwerte")
#plt.plot(phi_a[i]*np.ones(10), np.linspace(min(real_a)-500, max(real_a)+500, 10), label=("Maximum"))
#plt.plot(phi_a, imaginary_a, "bx", label = "Imaginär")
#plt.plot(phi_a, real_a + real_off_a, "mx", label= "Real mit off")
#plt.plot(phi_a, imaginary_a + imaginary_off_a, "gx", label = "Imaginär mit off")
#plt.legend()
#plt.show()
