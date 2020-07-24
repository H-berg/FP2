import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


plt.rc('text', usetex=True)
time, real, real_err, imaginary, imaginery_err, real_off, real_off_err, imaginary_off, imaginary_off_err, sample_temperature = np.genfromtxt('T1.dat.nmr', unpack=True)

time *= 1e3# zeit in millisek


# Offset
real_off -= min(real_off)

# Unrelevante Werte rausschmeißen
index = np.where(real_off < 20)

real_off        = np.delete(real_off, index)
time        = np.delete(time, index)
real_off_err    = np.delete(real_off_err, index)

# Normierung
maxi = max(real_off)
real_off /= (maxi + real_off_err[np.where(maxi)])
real_off_err /= maxi

def exp(t,A,b,T):
    return A*np.exp(-(t/T)**b)

t = np.linspace(0.008,max(time)+2, 100)


# Fit
p0 = np.array([max(real_off), 2.5,14])   #Schätzwerte
bounds = ([0,0,0],[np.inf,np.inf,100])   # Intervall
params, covariance_matrix = curve_fit(exp, time, real_off, p0, bounds=bounds)

uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('AbT', params, uncertainties):
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')


# Plot
plt.figure(figsize=(10,5))

plt.errorbar(time, real_off + real_off_err, yerr=real_off_err, capsize=3, fmt='.', label="Messwerte")
plt.plot(t, exp(t, *params), label = "Fit-Funktion")
#plt.plot(t,exp(t, *p0), label="Einstellung") # Schätzer testen

plt.xscale("log")
plt.xlabel(r"Evolutionszeit $\tau$(ms)")
plt.ylabel("Amplitude (Realanteil)")
plt.axis([min(time)-0.003,max(time)+5,-0.05, 1.05])
plt.legend()
plt.savefig("T1.pdf", dpi = 1000)
