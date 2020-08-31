import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
t1, RMG1, URL1, IRL1  = np.genfromtxt('A4_1A.txt',  unpack=True) # s | ohm | mV | A
t2, RMG2, URL2, IRL2  = np.genfromtxt('A4_08A.txt', unpack=True) # s | ohm | mV | A
t3, RMG3, URL3, IRL3  = np.genfromtxt('A4_06A.txt', unpack=True) # s | ohm | mV | A
t4, RMG4, URL4, IRL4  = np.genfromtxt('A4_04A.txt', unpack=True) # s | ohm | mV | A
t5, RMG5, URL5, IRL5  = np.genfromtxt('A4_02A.txt', unpack=True) # s | ohm | mV | A

IMG = 10e-6     # muA, Strom am Messgerät


#t2,U2 = np.genfromtxt('A2_4.txt', unpack=True)
#t3,U3 = np.genfromtxt('A2_7.txt', unpack=True)
#Rkrit = np.empty(3)
#tkrit = np.empty(3)

# Zeitpunkt, an dem Magnet absenkt ist
# Werte abschneiden, Zeit bei Null beginnen lassen und U_krit bestimmen

cut = np.array([180,140,330,290,280])
for i in range(1,6):
    locals()["t"+str(i)]      =   locals()["t"+str(i)][cut[i-1]::]
    locals()["RMG"+str(i)]    = locals()["RMG"+str(i)][cut[i-1]::]
    locals()["IRL"+str(i)]    = locals()["IRL"+str(i)][cut[i-1]::]
    locals()["URL"+str(i)]    = locals()["URL"+str(i)][cut[i-1]::]
    locals()["t"+str(i)]      -= min(locals()["t"+str(i)])


def R(U,I):
    return U/I

# Kalibrierungsdaten des Platin-Temp-Sensors

a0  = 19.7443
da0 = 2.50532
a1  = 3.20409
da1 = 0.33372
a2  = -0.02161
da2 = 0.01102
a3  = 1.72736*1e-4
da3 = 9.85867*1e-5

def T(R):
    return a0 + a1*R + a2*R*R + a3*R*R*R

# irgendwie ist der Kalibrierungsfit nicht so schön, da hier riesige Fehler rauskommen, naja..
def dT(R):
    return np.sqrt((da0)**2 + (U*da1)**2 + (U*U*da2)**2 + (U*U*U*da3)**2)



# Plot R(T) für 1, 0.8, 0.6, 0.4, 0.2 A
Ampere = np.linspace(1,0.2,5)
Ampere = np.round(Ampere,1)
tkrit = np.array([61,63,64,78,76])
Tkrit = np.empty(5)
dTmittel = np.zeros(5)

for i in range(1,6):
    #plt.plot(locals()["t"+str(i)], locals()["RMG"+str(i)], ".", label=str(Ampere[i-1])+r"$\,$A")
    RMG         = locals()["RMG"+str(i)]
    URL         = locals()["URL"+str(i)]
    IRL         = locals()["IRL"+str(i)]
    t           = locals()["t"+str(i)]
    TK          = tkrit[i-1]
    Tkrit[i-1]  = T(RMG[TK])    # Kritsche Temperatur speichern
    ampere      = Ampere[i-1]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim([79, 150])
    ax1.set_ylim([-0.2,11])
    ax2 = ax1.twiny()

    ax1.plot(T(RMG), R(URL,IRL), ".", label=str(ampere)+r"$\,$A",color="#606da4",zorder=1)

    ax2.tick_params(axis='x', labelcolor='#e5961c')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([T(RMG[TK])])
    ax2.set_xticklabels([round(T(RMG[TK]),1)])

    ax2.plot(np.ones(20)*T(RMG[TK]), np.linspace(-0.5,12,20), "r--", linewidth=0.5, color="#e5961c", zorder=2)
    ax2.text(T(RMG[TK])+0.2, 10, s=r"T$_{krit}$", color="#e5961c", fontsize=12)
    print("Für", ampere,"A:")
    print("kritsche Temperatur = ", T(RMG[TK]), " K")
    s       = str(round(R(URL[TK], IRL[TK]), 2)) + r"$\,$m$\Omega$"
    xy      = np.array([T(RMG[TK]),R(URL[TK], IRL[TK])])
    xytext  = np.array([T(RMG[TK])-10,R(URL[TK], IRL[TK])+1])
    plt.annotate(s, xy=xy, xytext=xytext, arrowprops=dict(color="black", arrowstyle="simple"))
    ax1.legend(loc="best")
    ax1.set_xlabel("Temperatur T / K")
    ax1.set_ylabel(r"Widerstand R$_{SL}$ / m$\Omega$")
    plt.savefig("R_T_"+str(ampere)+"A.pdf")
    plt.clf()

    # Temperaturänderung im Ablesebereich zur Fehlerabschätzung
    for k in range(1,3):
        dTmittel[i-1] += np.abs(T(RMG[TK])-T(RMG[TK-k]))
        dTmittel[i-1] += np.abs(T(RMG[TK])-T(RMG[TK+k]))
    dTmittel[i-1] /= 2*k
    print("Tempaenderung beim Ablesen = ", dTmittel[i-1], " K")
    print()

# lin. Fit
def lin(T,m,b):
    return m*T+b

Temp = np.linspace(75,117.5,100)
# I(Tkrit) zu Bestimmung des kritischen Stroms

plt.errorbar(Tkrit, Ampere, xerr=dTmittel+1, capsize=3, fmt='.', label="Kritische Temperatur", color="#606da4", zorder=3)
plt.plot(Tkrit[2], Ampere[2], "rx", mew=1,markerfacecolor="white", markersize=6,label="Ausreißer", zorder=4)
Tkrit  = np.delete(Tkrit, [2])
Ampere = np.delete(Ampere, [2])

# lin. Fit
def lin(T,m,b):
    return m*T+b

params1, covariance_matrix1 = curve_fit(lin, Tkrit, Ampere)#, p0, bounds=bounds)
uncertainties1 = np.sqrt(np.diag(covariance_matrix1))
print("Ausgleichsgerade:")
print("m = ",params1[0], "+-", uncertainties1[0])
print("b = ",params1[1], "+-", uncertainties1[1])

plt.plot(Temp, lin(Temp, *params1), label="Ausgleichsgerade",color="#acb2ba",zorder=2)
Ikrit = lin(77, *params1)
plt.plot(77, Ikrit, "o", mew=1,markerfacecolor="white", markersize=8,label=r"Kritscher Strom I$_{krit}$",color="#e5961c",zorder=1)
plt.plot(77, Ikrit, ".", color="#e5961c")
plt.text(78, lin(78, *params1)+0.2, s=str(round(Ikrit, 2))+"$\,$A",color="#e5961c")

plt.xlabel(r"Temperatur T / K")
plt.ylabel(r"Strom I / A")
plt.legend()

plt.savefig("I_krit.pdf")
plt.clf()
