import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


plt.rc('text', usetex=True)
time, real, real_err, imaginary, imaginery_err, real_off, real_off_err, imaginary_off, imaginary_off_err, sample_temperature = np.genfromtxt('T1.dat.nmr', unpack=True)

time *= 1e3# zeit in millisek


# Offset
#real_off -= min(real_off)

# Unrelevante Werte rausschmeißen
#index = np.where(real_off < 20)
#
#real_off        = np.delete(real_off, index)
#time        = np.delete(time, index)
#real_off_err    = np.delete(real_off_err, index)

# Normierung
#real_off_err /= max(real_off)-min(real_off)
#real_off /= max(real_off)-min(real_off)

def exp(t,A,B,b,T):
    return A*np.exp(-(t/T)**b)+B

t = np.logspace(-2,3, 100)


# Fit
p0 = np.array([-max(real_off),1,10,14])   #Schätzwerte
bounds = ([-np.inf,-np.inf,0,0],[np.inf,np.inf,10,100])   # Intervall
params, covariance_matrix = curve_fit(exp, time, real_off, p0, bounds=bounds)

uncertainties = np.sqrt(np.diag(covariance_matrix))

B = params[1] # für die Offset-Korrektur
Norm = abs(params[0]) + B
print(Norm)


# Plot
plt.figure(figsize=(10,5))

plt.errorbar(time,( real_off+B)/Norm, yerr=real_off_err/Norm, capsize=3, fmt='.', color="#505054", label="Mess-Signal")
plt.plot(t, (exp(t, *params)+B)/Norm, color="#98989e")
plt.annotate(s=r'$T_1$ = ' + str(round(params[3],2)) + " ms", xy=(params[3], 0.5), xytext=(params[3]-9,0.71), arrowprops=dict(color="black", arrowstyle="simple"))
#plt.plot(t,exp(t, *p0), label="Einstellung") # Schätzer testen

plt.xscale("log")
plt.xlabel(r"Evolutionszeit t / ms")
plt.ylabel("Signalintensität I (Realanteil)")
#plt.axis([min(time)-0.003,max(time)+300,min(real_off+B)-0.05, max(real_off+B)+0.05])
plt.legend()
plt.savefig("T1.pdf", dpi = 1000)

# Nomierte Fit-Parameter wiedergeben
params[0] /= Norm
params[1] /= Norm
uncertainties[0] /= Norm
uncertainties[1] /= Norm


for name, value, uncertainty in zip('ABbT', params, uncertainties):
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')
