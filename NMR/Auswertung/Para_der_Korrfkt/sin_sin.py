import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


plt.rc('text', usetex=True)
time, real, real_err, imaginary, imaginery_err, real_off, real_off_err, imaginary_off, imaginary_off_err, sample_temperature = np.genfromtxt('sin_sin.dat.nmr', unpack=True)

time *= 1e3# Mischzeit in millisek

def exp(t,S0,A,B,b,c,tau,T1):
    return S0 + (A*np.exp(-(t/tau)**b)+B)*np.exp(-(t/T1)**c)

t = np.logspace(-1.8,3, 1000)

# Fit
p0 = np.array([100,100, 100,1.5, 1.5, 1,14])   #Schätzwerte
bounds = ([-np.inf,np.inf,np.inf,-np.inf,-np.inf,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf,20,100])   # Intervall
params, covariance_matrix = curve_fit(exp, time, real_off, p0)#, bounds=bounds)

uncertainties = np.sqrt(np.diag(covariance_matrix))

Offset  = params[0]
Norm    = params[0] + params[1] + params[2]


# Plot
plt.figure(figsize=(10,5))

plt.errorbar(time, (real_off-Offset)/Norm, yerr=real_off_err/Norm, capsize=3, fmt='.', label="Mess-Signal", color="#970e21")
plt.plot(t, (exp(t, *params)-Offset)/Norm, color="#ec6073")
#plt.plot(t,exp(t, *p0), label="Einstellung") # Schätzer testen
plt.annotate(s=r'$\tau$ = ' + str(round(params[5],2)) + " ms", xy=(params[5], 0.73), xytext=(params[5]-0.11,0.94), arrowprops=dict(color="black", arrowstyle="simple"))
plt.annotate(s=r'$T_{1,Q}$ = ' + str(round(params[6],2)) + " ms", xy=(params[6], 0.2), xytext=(params[6]-22.7,0.41), arrowprops=dict(color="black", arrowstyle="simple"))

#plt.text(params[6]-0.09, 0.41, s=r'$T_{1Q}$ = ' + str(round(params[5],2)) + " ms")

plt.xscale("log")
plt.xlabel(r"Mischzeit t$_m$ / ms")
plt.ylabel("Amplitude S (stimuliertes Echo, Betrag $|Re + Im|$)")
#plt.axis([min(time)-0.002,max(time)+100,-0.05, 1.05])
plt.title("sin-sin")
plt.legend()
#plt.show()
plt.savefig("sin_sin.pdf", dpi = 1000)

# Output nomierter Parameter
for i in range(3):
    params[i]       /= Norm
    uncertainties[i]/= Norm

    for name, value, uncertainty in zip('SABbctT', params, uncertainties):
        print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')
