import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
t1,U1 = np.genfromtxt('A2_3.txt', unpack=True)
t2,U2 = np.genfromtxt('A2_4.txt', unpack=True)
t3,U3 = np.genfromtxt('A2_7.txt', unpack=True)
Rkrit = np.empty(3)
tkrit = np.empty(3)

fall  = np.array([160, 143, 179])
# Zeitpunkt, an dem Magnet absenkt ist
# Werte abschneiden, Zeit bei Null beginnen lassen und U_krit bestimmen

for i in range(1,4):
    locals()["t"+str(i)]    = locals()["t"+str(i)][100::]
    locals()["U"+str(i)]    = locals()["U"+str(i)][100::]
    locals()["t"+str(i)]    -= min(locals()["t"+str(i)])

fall -= 100


# Kalibrierungsdaten des Silizium-Temp-Sensors

a0  = 341.29569
da0 = 18.56165
a1  = 647.519
da1 = 93.37866
a2  = -2179.35087
da2 = 173.84225
a3  = 1971.14823
da3 = 142.02358
a4  = -687.95684
da4 = 42.9904

def T(U):
    return a0 + a1*U + a2*U*U + a3*U*U*U + a4*U*U*U*U


# Plot U(t)
plt.plot(t1,U1,".", label="Messung 1", color="#606da4",zorder=2)
plt.plot(t2,U2,".", label="Messung 2", color="#2c3767",zorder=2)
plt.plot(t3,U3,".", label="Messung 3", color="#a4aed8",zorder=2)
plt.plot(t1[fall[0]],U1[fall[0]], "o", color="#e5961c", mew=1,markerfacecolor="white", markersize=8,label=r"$U_{krit}$",zorder=1)
plt.plot(t2[fall[2]],U2[fall[2]], "o", color="#e5961c", mew=1,markerfacecolor="white", markersize=8,zorder=1)
plt.plot(t3[fall[1]],U3[fall[1]], "o", color="#e5961c", mew=1,markerfacecolor="white", markersize=8,zorder=1)

plt.xlabel("Zeit t / s")
plt.ylabel("Spannung U / V")
plt.legend()
#plt.show()
plt.savefig("U_krit.pdf")
plt.clf()

# plot T(t)
#plt.errorbar(t1, T(U1), yerr=dT(U1), capsize=3, fmt='.', label="Messung 1", color="#606da4",zorder=2)
plt.plot(t1,T(U1),".", label="Messung 1", color="#606da4",zorder=2)
plt.plot(t2,T(U2),".", label="Messung 2", color="#2c3767",zorder=2)
plt.plot(t3,T(U3),".", label="Messung 3", color="#a4aed8",zorder=2)
plt.plot(t1[fall[0]],T(U1[fall[0]]), "o", color="#e5961c", mew=1,markerfacecolor="white", markersize=8,label=r"$U_{krit}$",zorder=1)
plt.plot(t2[fall[2]],T(U2[fall[2]]), "o", color="#e5961c", mew=1,markerfacecolor="white", markersize=8,zorder=1)
plt.plot(t3[fall[1]],T(U3[fall[1]]), "o", color="#e5961c", mew=1,markerfacecolor="white", markersize=8,zorder=1)

#plt.yticks(np.linspace(min(T(U(R1,I)*1e-6)), max(T(U(R1,I)*1e-6)),10),np.linspace(min(T(U(R1,I)*1e-6)), max(T(U(R1,I)*1e-6)),10))

#plt.yscale("log")
plt.xlabel(r"Zeit t / s")
plt.ylabel(r"Temperatur T / K")
plt.legend()
#plt.show()
plt.savefig("T_krit.pdf")
plt.clf()
