import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import constants


plt.rc('text', usetex=True)
x = np.genfromtxt('data.txt', unpack=True)

Temp        = x[0]
T1          = x[1]
T1_err      = x[2]
T1Q         = x[7]
T1Q_err     = x[8]
tau_cos     = x[3]
tau_cos_err = x[4]
tau_sin     = x[5]
tau_sin_err = x[6]

# cos-cos Ausreißer raus nehmen
raus            = np.where(tau_cos>2)
Temp_raus       = Temp[raus]
tau_cos_raus    = tau_cos[raus]
tau_cos_err_raus= tau_cos_err[raus]
Temp_cos        = np.delete(Temp, raus)
tau_cos         = np.delete(tau_cos, raus)
tau_cos_err     = np.delete(tau_cos_err,raus)

def lin(Temp, a, b):
    return a/Temp + b

T = np.linspace(min(Temp),max(Temp), 1000)

# Fit
params_cos, covariance_matrix_cos = curve_fit(lin, Temp_cos, np.log(tau_cos))#, p0, bounds=bounds)

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

plt.errorbar(1000/Temp_cos, tau_cos, yerr=tau_cos_err, capsize=3, fmt='.', label="cos-cos", color="#2a3e87",zorder=2)
plt.errorbar(1000/Temp, tau_sin, yerr=tau_sin_err, capsize=3, fmt='.', label="sin-sin", color="#970e21",zorder=2)
plt.plot(1000/T, np.exp(lin(T, *params_cos)), color="#7875d6", zorder=1)
plt.plot(1000/T, np.exp(lin(T, *params_sin)), color="#ec6073",zorder=1)
# Ausreißer kennzeichnen
plt.errorbar(1000/Temp_raus, tau_cos_raus, yerr=tau_cos_err_raus,capsize=3, fmt='.', color="#2a3e87",zorder=2)
plt.plot(1000/Temp_raus, tau_cos_raus, "rx", mew=3, markersize=15,zorder=3)
#plt.plot(T, arrhenius(T, *params), label = "S2-Fit-Funktion")
#plt.plot(T,arrhenius(T, *p0), label="Einstellung") # Schätzer testen
#plt.annotate(s=r'$\tau_{hop}$ = ' + str(round(params[5],2)) + " ms", xy=(params[5], 0.78), xytext=(params[5]-0.085,0.57), arrowprops=dict(color="black", arrowstyle="simple"))
#plt.annotate(s=r'$T_{1}$ = ' + str(T1) + " ms", xy=(T1, 0.32), xytext=(T1-5.5,0.53), arrowprops=dict(color="black", arrowstyle="simple"))


plt.yscale("log")
plt.xlabel(r"Temperatur $\frac{1000}{T}$ / $K^{-1}$")
plt.ylabel(r"Korrelationszeit $\tau$ / ms")
#plt.axis([min(Temp)-1,max(Temp)+1,-0.05, 1.05])
plt.legend()
#plt.show()
plt.savefig("Korr_Temp.pdf", dpi = 1000)
plt.clf()


# -------------------- T1 & T1Q ------------------------------------------------
# Fit
params_T1, covariance_matrix_T1 = curve_fit(lin, Temp, np.log(T1))

uncertainties_T1 = np.sqrt(np.diag(covariance_matrix_T1))

print("T1:")
for name, value, uncertainty_T1 in zip('ab', params_T1, uncertainties_T1):
    print(f'{name} = {value:8.3f} ± {uncertainty_T1:.3f}')
print()

# Fit
params_T1Q, covariance_matrix_T1Q = curve_fit(lin, Temp, np.log(T1Q))

uncertainties_T1Q = np.sqrt(np.diag(covariance_matrix_T1Q))

print("T1Q:")
for name, value, uncertainty_T1Q in zip('ab', params_T1Q, uncertainties_T1Q):
    print(f'{name} = {value:8.3f} ± {uncertainty_T1Q:.3f}')
print()

plt.errorbar(1000/Temp, T1, yerr=T1_err, capsize=3, fmt='.', color="#505054", label=r"$T_1$")
plt.plot(1000/Temp, np.exp(lin(Temp, *params_T1)), color="#98989e")
plt.plot(1000/Temp_raus, T1[raus], "ro")#, markersize=15)
plt.errorbar(1000/Temp, T1Q, yerr=T1Q_err, capsize=3, fmt='.', color="#8baeaf", label=r"$T_{1,Q}$")
plt.plot(1000/Temp, np.exp(lin(Temp, *params_T1Q)), color="#a4d3d4")
plt.plot(1000/Temp_raus, T1Q[raus], "ro")#, markersize=15)


plt.yscale("log")
plt.xlabel(r"Temperatur $\frac{1000}{T}$ / $K^{-1}$")
plt.ylabel(r"Korrelationszeit $\tau$ / ms")
plt.axis([min(1000/Temp)-0.025,max(1000/Temp)+0.025,15,60])
plt.legend()
plt.savefig("T1.pdf", dpi = 1000)
