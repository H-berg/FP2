import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


plt.rc('text', usetex=True)
time, real, real_err, imaginary, imaginery_err, real_off, real_off_err, imaginary_off, imaginary_off_err, sample_temperature = np.genfromtxt('sin_sin.dat.nmr', unpack=True)

time *= 1e3# Mischzeit in millisek

# Offset
real_off -= min(real_off)

# Normierung
maxi = max(real_off)
real_off /= (maxi + real_off_err[np.where(maxi)])
real_off_err /= maxi


def exp(t,S0,A,B,b,c,tau,T1):
    return S0 + (A*np.exp(-(t/tau)**b)+B)*np.exp(-(t/T1)**c)

t = np.linspace(min(time),max(time), 100000)

# Fit
p0 = np.array([0.2,0.5, 0.3,1.5, 1.5, 1,14])   #Schätzwerte
bounds = ([0,0,0,-np.inf,-np.inf,0,0],[2,1,1,np.inf,np.inf,20,100])   # Intervall
params, covariance_matrix = curve_fit(exp, time, real_off, p0, bounds=bounds)

uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('SABbctT', params, uncertainties):
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')


# Plot
plt.figure(figsize=(10,5))

plt.errorbar(time, real_off + real_off_err, yerr=real_off_err, capsize=3, fmt='.', label="Messwerte")
plt.plot(t, exp(t, *params), label = "S2-Fit-Funktion")
#plt.plot(t,exp(t, *p0), label="Einstellung") # Schätzer testen
plt.annotate(s=r'$\tau_{hop}$ = ' + str(round(params[5],2)) + " ms", xy=(params[5], 0.7), xytext=(params[5]-0.12,0.91), arrowprops=dict(color="black", arrowstyle="simple"))
plt.annotate(s=r'$T_{1Q}$ = ' + str(round(params[6],2)) + " ms", xy=(params[6], 0.2), xytext=(params[6]-21,0.41), arrowprops=dict(color="black", arrowstyle="simple"))

#plt.text(params[6]-0.09, 0.41, s=r'$T_{1Q}$ = ' + str(round(params[5],2)) + " ms")

plt.xscale("log")
plt.xlabel(r"Mischzeit t$_m$(ms)")
plt.ylabel("Amplitude (stimuliertes Echo)")
plt.axis([min(time)-0.002,max(time)+100,-0.05, 1.05])
plt.title("sin-sin")
plt.legend()
#plt.show()
plt.savefig("sin_sin.pdf", dpi = 1000)
