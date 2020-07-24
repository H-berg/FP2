import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


plt.rc('text', usetex=True)
time, real, real_err, imaginary, imaginery_err, real_off, real_off_err, imaginary_off, imaginary_off_err, sample_temperature = np.genfromtxt('cos_cos.dat.nmr', unpack=True)

time *= 1e3# Mischzeit in millisek

# Offset
real_off -= min(real_off)

# Unrelevante Werte rausschmeißen
index = np.where(real_off < 24)

real_off_cut        = np.delete(real_off, index)
time_cut        = np.delete(time, index)
real_off_err_cut    = np.delete(real_off_err, index)
print(real_off_cut)

# Normierung
maxi = max(real_off)
real_off /= (maxi + real_off_err[np.where(maxi)])
real_off_err /= maxi

T1 = 7.281573839807832016

def exp(t,S0,A,B,b,c,tau):
    return S0 + (A*np.exp(-(t/tau)**b)+B)*np.exp(-(t/T1)**c)

t = np.linspace(min(time),max(time), 100000)

# Fit
p0 = np.array([0.2,0.5, 0.3,1.5, 1.5, 1])   #Schätzwerte
bounds = ([0,0,0,-np.inf,-np.inf,0],[2,2,2,np.inf,np.inf,T1])   # Intervall
params, covariance_matrix = curve_fit(exp, time, real_off, p0, bounds=bounds)

uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('SABbcT', params, uncertainties):
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')


# Plot
plt.figure(figsize=(10,5))

plt.errorbar(time, real_off + real_off_err, yerr=real_off_err, capsize=3, fmt='.', label="Messwerte")
plt.plot(t, exp(t, *params), label = "S2-Fit-Funktion")
#plt.plot(t,exp(t, *p0), label="Einstellung") # Schätzer testen
plt.annotate(s=r'$\tau_{hop}$ = ' + str(round(params[5],2)) + " ms", xy=(params[5], 0.78), xytext=(params[5]-0.085,0.57), arrowprops=dict(color="black", arrowstyle="simple"))
plt.annotate(s=r'$T_{1}$ = ' + str(T1) + " ms", xy=(T1, 0.32), xytext=(T1-5.5,0.53), arrowprops=dict(color="black", arrowstyle="simple"))


plt.xscale("log")
plt.xlabel(r"Mischzeit t$_m$(ms)")
plt.ylabel("Amplitude (stimuliertes Echo)")
plt.axis([min(time)-0.002,max(time)+100,-0.05, 1.05])
plt.title("cos-cos")
plt.legend()
#plt.show()
plt.savefig("cos_cos.pdf", dpi = 1000)
