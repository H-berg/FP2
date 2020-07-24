import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import constants


plt.rc('text', usetex=True)
x = np.genfromtxt('data.txt', unpack=True)

Temp        = x[0]
T1          = x[1]
T1_err      = x[2]
tau_cos     = x[3]
tau_cos_err = x[4]
tau_sin     = x[5]
tau_sin_err = x[6]

def arrhenius(Temp, tau0, E,A,B):
    return tau0*np.exp(E/Temp+A)+B

def lin(Temp, a, b):
    return a/Temp + b

T = np.linspace(min(Temp),max(Temp), 1000)

# Fit
params_cos, covariance_matrix_cos = curve_fit(lin, Temp, np.log(tau_cos))#, p0, bounds=bounds)

uncertainties_cos = np.sqrt(np.diag(covariance_matrix_cos))

print("cos_cos:")
for name, value, uncertainty_cos in zip('ab', params_cos, uncertainties_cos):
    print(f'{name} = {value:8.3f} ± {uncertainty_cos:.3f}')
print()

params_sin, covariance_matrix_sin = curve_fit(lin, Temp, np.log(tau_sin))#, p0, bounds=bounds)

uncertainties_sin = np.sqrt(np.diag(covariance_matrix_sin))

print("sin_sin:")
for name, value, uncertainty_sin in zip('ab', params_sin, uncertainties_sin):
    print(f'{name} = {value:8.3f} ± {uncertainty_sin:.3f}')
print()

# Plot
plt.figure(figsize=(10,5))

plt.errorbar(Temp, tau_cos+ tau_cos_err, yerr=tau_cos_err, capsize=3, fmt='.', label="Messwerte cos")
plt.errorbar(Temp, tau_sin + tau_sin_err, yerr=tau_sin_err, capsize=3, fmt='.', label="Messwerte sin")
#plt.plot(T, arrhenius(T, *params), label = "S2-Fit-Funktion")
plt.plot(T, np.exp(lin(T, *params_cos)), label = "Fit cos")
plt.plot(T, np.exp(lin(T, *params_sin)), label = "Fit sin")
#plt.plot(T,arrhenius(T, *p0), label="Einstellung") # Schätzer testen
#plt.annotate(s=r'$\tau_{hop}$ = ' + str(round(params[5],2)) + " ms", xy=(params[5], 0.78), xytext=(params[5]-0.085,0.57), arrowprops=dict(color="black", arrowstyle="simple"))
#plt.annotate(s=r'$T_{1}$ = ' + str(T1) + " ms", xy=(T1, 0.32), xytext=(T1-5.5,0.53), arrowprops=dict(color="black", arrowstyle="simple"))


plt.yscale("log")
plt.xlabel(r"Temperatur T / K")
plt.ylabel(r"Korrelationszeit $\tau$ / ms")
#plt.axis([min(Temp)-1,max(Temp)+1,-0.05, 1.05])
plt.legend()
#plt.show()
plt.savefig("Temp_Korr.pdf", dpi = 1000)
plt.clf()

plt.errorbar(Temp, T1+ T1_err, yerr=T1_err, capsize=3, fmt='.', label="Messwerte T1")

plt.yscale("log")
plt.xlabel(r"Temperatur T / K")
plt.ylabel(r"Korrelationszeit $\tau$ / ms")
#plt.axis([min(Temp)-1,max(Temp)+1,-0.05, 1.05])
plt.legend()
plt.savefig("T1_T2.pdf", dpi = 1000)
