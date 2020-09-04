import numpy as np
import scipy.constants as const

# Biot-Savart-Gesetz für eine kreisförmige Stromschleife aus Demtröder 2, Seite 91
def I(z,R,B):
    I = 2*B*(z*z + R*R)**(1.5)
    I /= const.mu_0 * R*R
    return I

B       = 1.59*1e-3   # T
z       = 0
R       = 0.5*15*1e-3 # m
z_err   = 1e-3          # m
I_err   = np.abs(I(z,R,B)-I(z_err,R,B)) # absolute Fehler

print("Der induzierte Strom hat eine Staerke von ", I(z,R,B), "+-", I_err,"A")
