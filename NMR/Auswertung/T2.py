import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


plt.rc('text', usetex=True)
time, real, real_err, imaginary, imaginery_err, real_off, real_off_err, imaginary_off, imaginary_off_err, sample_temperature = np.genfromtxt('T2.dat.nmr', unpack=True)

time *= 1e3# zeit in millisek
time *= 2

y = np.sqrt(real_off**2 + imaginary_off**2)
# Fehler nach gauß Fehlerfortpflanzung
y_err = np.sqrt( ( (real_off * real_off_err)**2 + (imaginary_off*imaginary_off_err)**2 ) / (real_off**2 + imaginary_off**2) )
# Offset
#y -= min(y)

# Normierung
y_err /= max(y)-min(y)
y /= max(y)-min(y)

#real_off -= min(real_off)
## Unrelevante Werte rausschmeißen
#index = np.where(real_off > 3720)
#
#real_off        = np.delete(real_off, index)
#time        = np.delete(time, index)
#real_off_err    = np.delete(real_off_err, index)



def exp(t,A,B,b,T2):
    return A - B*np.exp(-(t/T2)**b)

t = np.linspace(min(time),max(time), 1000)


#### Fit
p0 = np.array([max(y), max(y),-2 ,0.2])   #Schätzwerte
bounds = ([max(y),0,-np.inf,0],[np.inf,np.inf,np.inf,5])   # Intervall
params, covariance_matrix = curve_fit(exp, time, y, p0, bounds=bounds)
#
uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('ABbT', params, uncertainties):
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')

B = params[1] # für die Offset-Korrektur
# Plot
plt.figure(figsize=(10,5))

plt.errorbar(time, y-B, yerr=y_err, capsize=3, fmt='.', color="#505054", label=r"Mess-Signal")
plt.plot(t, exp(t, *params)-B, color="#98989e")
plt.annotate(s=r'$T_2$ = ' + str(round(params[3],2)) + " ms", xy=(params[3], -0.35), xytext=(params[3]-0.032,-0.14), arrowprops=dict(color="black", arrowstyle="simple"))

#plt.plot(t,exp(t, *p0), label="Einstellung") # Schätzer testen

plt.xscale("log")
plt.xlabel(r"Evolutionszeit t / s")
plt.ylabel("Signalintensität I (Realanteil)")
#plt.axis([0.035,max(time)+0.2,-0.05, 1.05])
plt.legend()
#plt.show()
plt.savefig("T2.pdf", dpi = 1000)
