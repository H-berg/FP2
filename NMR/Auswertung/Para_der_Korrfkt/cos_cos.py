import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


plt.rc('text', usetex=True)
time, real, real_err, imaginary, imaginery_err, real_off, real_off_err, imaginary_off, imaginary_off_err, sample_temperature = np.genfromtxt('cos_cos.dat.nmr', unpack=True)

time *= 1e3# Mischzeit in millisek

# Offset
#real_off -= min(real_off)

## Normierung
#real_off_err /= max(real_off)-min(real_off)
#real_off /= max(real_off)-min(real_off)

T1 = 18.313

def exp(t,S0,A,B,b,c,tau):
    return S0 + (A*np.exp(-(t/tau)**b)+B)*np.exp(-(t/T1)**c)

t = np.logspace(-1.8,3, 1000)

# Fit
p0 = np.array([100,100, 100,1, 1, 1])   #Schätzwerte
bounds = ([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,np.inf,np.inf,T1])   # Intervall
params, covariance_matrix = curve_fit(exp, time, real_off, p0, bounds=bounds)

uncertainties = np.sqrt(np.diag(covariance_matrix))

Offset      = params[0]
Norm        = params[0] + params[1] + params[2]


# Plot
plt.figure(figsize=(10,5))

plt.errorbar(time, (real_off-Offset)/Norm, yerr=real_off_err/Norm, capsize=3, fmt='.', label="Mess-Signal", color="#2a3e87")
plt.plot(t, (exp(t, *params)-Offset)/Norm, color="#7875d6")
#plt.plot(t,exp(t, *p0), label="Einstellung") # Schätzer testen
plt.annotate(s=r'$\tau$ = ' + str(round(params[5],2)) + " ms", xy=(params[5], 0.73), xytext=(params[5]-0.8,0.95), arrowprops=dict(color="black", arrowstyle="simple"))
plt.annotate(s=r'$T_{1}$ = ' + str(round(T1,2)) + " ms", xy=(T1, 0.25), xytext=(T1-9,0.46), arrowprops=dict(color="black", arrowstyle="simple"))


plt.xscale("log")
plt.xlabel(r"Mischzeit t$_m$(ms)")
plt.ylabel("Amplitude S (stimuliertes Echo, Realanteil)")
#plt.axis([min(time)-0.002,max(time)+100,-0.05, 1.05])
plt.title("cos-cos")
plt.legend()
#plt.show()
plt.savefig("cos_cos.pdf", dpi = 1000)

# Output nomierter Parameter
for i in range(3):
    params[i]       /= Norm
    uncertainties[i]/= Norm

for name, value, uncertainty in zip('SABbct', params, uncertainties):
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')
