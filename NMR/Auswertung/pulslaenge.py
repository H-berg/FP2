import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
time, real, real_err, imaginary, imaginery_err, real_off, real_off_err, imaginary_off, imaginary_off_err, sample_temperature = np.genfromtxt('pulslaenge.dat.nmr', unpack=True)
# zeit in musek
time *= 1e6

# offset beheben
Amp = abs(min(real_off)) + max(real_off)
real_off += Amp/2 - max(real_off)

# Normierung
maxi = max(real_off)
real_off /= (maxi + real_off_err[np.where(maxi)])
real_off_err /= maxi


# fiiting klappt irgendwie nicht..
x = np.linspace(0,max(time), 100)
def sinus(A,b,c,d,x):
    return A*np.sin(b*x+c)+d



p0 = np.array([3700, 0.4, 3*np.pi/3.7, -3700])
params, covariance_matrix = curve_fit(sinus, time, real_off)#, p0)


i = np.argmin(real_off)

plt.plot(time[i]*np.ones(10), np.linspace(min(real_off)-0.2, max(real_off)+0.2, 10), "--", color="#862F29", label="gewähltes Pulslänge")
plt.errorbar(time, real_off + real_off_err, yerr=real_off_err, capsize=3, fmt='.', label="Messwerte")
plt.xlabel(r"Zeit t / $\mu$s")
plt.ylabel("Amplitude (Realanteil)")
plt.axis([min(time)-0.2, max(time)+0.2, -1.05, 1.05])
#plt.plot(time, sinus(time, *params), label = "fit")
#plt.plot(x, sinus(*p0, x), label = "sinus mit p0")
#plt.plot(phi, imaginary, "bx", label = "Imaginär")
#plt.plot(phi, real + real_off, "mx", label= "Real mit off")
#plt.plot(phi, imaginary + imaginary_off, "gx", label = "Imaginär mit off")
plt.legend()
#plt.show()
plt.savefig("pulslaenge.pdf")
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
