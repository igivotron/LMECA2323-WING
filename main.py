import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse as ap

ap_parser = ap.ArgumentParser(description="Analyse des données de l'aile")
ap_parser.add_argument('--origin', '-o',type=int, default=1, help='Fichier des données expérimentales. 0: nous, 1: Dandoy')
args = ap_parser.parse_args()
data_origin = args.origin

# Constants
g = 9.81 # m/s^2
c_mean = 0.05 # m
b = 0.3 # m
S = c_mean * b # m^2
AR = b**2 / S


if data_origin == 0:
    # Mesures 1
    p_atm = 100964 # Pa
    T = 22 + 273.15 # K
    NoWindOFFSET_L = -0.086
    NoWindOFFSET_D = 0.2
else:
    # Mesures 2 (Données de Dandoy)
    p_atm = 99965 # Pa
    T = 21 + 273.15 # K
    NoWindOFFSET_L = -0.118
    NoWindOFFSET_D = 0.17


# data = pd.read_csv('data.csv', sep=';', decimal=',')
data = pd.read_csv('data_dandoy.csv', sep=';', decimal=',')
AoA = data['AoA'].values
UL = data['UL'].values
UD = data['UD'].values

dyn_pressure = pd.read_csv('M001.CSV', sep=",", decimal='.')
pressure = dyn_pressure.iloc[:, 2].values

# Calibration
Cd_arm = 0.06433
calibration_data = pd.read_csv('calibration.csv', sep=';')
MD_calib = calibration_data['MD'].values
UD_calib = calibration_data['UD'].values
UL_calib = calibration_data['UL'].values

UD_coefs = np.polyfit(UD_calib, MD_calib, 1)
UL_coefs = np.polyfit(UL_calib, MD_calib,1)


# Variables environnementales
R = 287.05 # J/(kg*K)

def get_pressure_flow(p):
    p_mean = np.mean(p)
    p_std = np.std(p)
    z = 1.96
    error_std = z * p_std / np.sqrt(len(p))
    error_p_device = 0.001 * p_mean + 0.03
    error_p_reading = 0.01/2
    return p_mean, np.sqrt(error_p_device**2 + error_p_reading**2 + error_std**2)

def get_pressure_ambiant(p):
    error_p_device = 4 # Pa Voir labo cylindre
    error_p_reading = 0.01/2
    return p, np.sqrt(error_p_device**2 + error_p_reading**2)

def get_rho(T):
    delta_T = 1/2
    p_amb, error_p_amb = get_pressure_ambiant(p_atm)
    rho = p_amb / (R * T)
    error_rho = rho * np.sqrt((error_p_amb/p_amb)**2 + (delta_T/T)**2)
    return rho, error_rho

def get_dynamic_viscosity(T):
    # Sutherland's formula for air viscosity
    error_T = 1/2
    mu = 1.458e-6 * T**(3/2) / (T + 110.4)

    dmu_dT = 1.458e-6  * ((1.5 * T**0.5 * (T + 110.4) - T**1.5)/ (T + 110.4)**2)
    error_mu = abs(dmu_dT) * error_T
    return mu, error_mu

def get_velocity(p_flow, error_p_flow):
    rho, delta_rho = get_rho(T)
    v = np.sqrt(2 * p_flow / rho)
    error_v = v * np.sqrt((error_p_flow/p_flow)**2 + (delta_rho/rho)**2)
    return v, error_v


def get_Re(U, error_U, D, mu, error_mu, rho, error_rho):
    ReD = (rho * U * D) / mu
    ReD_error = ReD * np.sqrt((error_rho / rho)**2 + (error_U / U)**2 + (error_mu / mu)**2)
    return ReD, ReD_error


def get_force(Uv, coefs, offset):
    return (coefs[0] * (Uv-offset)) * g

def get_coefficients(F, Uinf, error_Uinf, rho, error_rho, S):
    C = F / (0.5 * rho * Uinf**2 * S)
    error_C = C * np.sqrt((error_rho/rho)**2 + (2*error_Uinf/Uinf)**2)
    return C, error_C


def fit_polar(CL, CD, AoA):
    AoA_max = 4
    AoA_min = -4
    nmax = np.where(AoA <= AoA_max)[0][-1]
    nmin = np.where(AoA >= AoA_min)[0][0]
    cl = np.power(CL[nmin:nmax], 2)
    cd = CD[nmin:nmax]
    coeffs = np.polyfit(cl, cd, 1)
    return coeffs

def get_oswald_efficiency(CL, CD, AR, CD0):
    e = CL**2 / (np.pi * AR * (CD - CD0))
    return e

def get_oswald_efficiency_fit(k, CD0, AR):
    e_fit = 1 / (k * np.pi * AR)
    return e_fit

def fit_cl_curve(AoA, CL):
    coeffs = np.polyfit(AoA, CL, 1)
    Cl = np.polyval(coeffs, 0)
    alpha0 = -Cl / (2 * np.pi)
    return coeffs, np.rad2deg(alpha0)

p_flow, error_p_flow = get_pressure_flow(pressure)
Uinf, error_Uinf = get_velocity(p_flow, error_p_flow)
rho, error_rho = get_rho(T)
mu, error_mu = get_dynamic_viscosity(T)
Re, error_Re = get_Re(Uinf, error_Uinf, c_mean, mu, error_mu, rho, error_rho)

FD = get_force(UD, UD_coefs, NoWindOFFSET_D)
FL = get_force(UL, UL_coefs, NoWindOFFSET_L)
CD, error_CD = get_coefficients(FD, Uinf, error_Uinf, rho, error_rho, S)
CL, error_CL = get_coefficients(FL, Uinf, error_Uinf, rho, error_rho, S)


CD_wing = CD - Cd_arm

k, CD0 = fit_polar(CL, CD_wing, AoA)
cl_polar = np.linspace(min(CL), max(CL), 100)
cd_polar = CD0 + k * cl_polar**2

# Oswald efficiency factor
e = get_oswald_efficiency(CL, CD_wing, AR, CD0)
e_fit = get_oswald_efficiency_fit(k, CD0, AR)

# Stall AoA
stall = np.where(CL == max(CL))[0][0]
AoAs = np.linspace(min(AoA), AoA[stall+1], 100)

# Approximation linéaire de CL par 2*pi(alpha-alpha0)
coefs_cl, alpha0 = fit_cl_curve(AoA[:stall], CL[:stall])
AoAs = np.linspace(min(AoA), AoA[stall+1], 100)
cl_fit = coefs_cl[0] * AoAs + coefs_cl[1]
"""
PLOTS
"""
plt.figure(figsize=(12, 6))
plt.plot(AoA, CL, 'o-', label='Experimental CL')
plt.plot(AoA, CD_wing, 'o-', label='Experimental CD')
plt.plot(AoAs, cl_fit, 'g--', label=f'slope={coefs_cl[0]:.4f}, alpha0={alpha0:.2f}°')
plt.axvline(x = AoA[7], color='r', linestyle='--', label='Stall AoA')
plt.xlabel('AoA (°)')
plt.ylabel('Coefficients')
plt.title("Lift and Drag Coefficients vs Angle of Attack")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('CL_CD_vs_AoA.pdf')


plt.clf()
plt.plot(cd_polar, cl_polar, 'r-', label='Fitted Polar')
plt.scatter(CD_wing, CL, color='b', label='Experimental Data')
plt.xlabel('CD')
plt.ylabel('CL')
plt.title("Polar Curve (CD vs CL)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Polar_Curve.pdf')


print("===============================")
print("Origine des données:", "Dandoy" if data_origin == 1 else "Nous")
print("------------------------------")
print("Conditions de l'air:")
print(f"Densité de l'air: {rho:.2f} kg/m³ ± {error_rho:.6f} kg/m³")
print(f"Pression de l'air: {p_atm:.2f} Pa ± {get_pressure_ambiant(p_atm)[1]:.2f} Pa")
print(f"Température de l'air: {T-273.15:.2f} °C ± 0.5 °C")
print(f"Pression dynamique: {p_flow:.2f} Pa ± {error_p_flow:.2f} Pa")
print(f"Vitesse de l'air: {Uinf:.2f} m/s ± {error_Uinf:.6f} m/s")
print(f"Viscosité dynamique de l'air: {mu:.6e} Pa·s ± {error_mu:.6e} Pa·s")
print(f"Nombre de Reynolds: {Re:.2e} ± {error_Re:.2e}")
print("-----------------------------")
print("Coefficients de l'aile:")
print(f"AR: {AR:.2f}")
print(f"CD0: {CD0:.4f}")
print(f"k: {k:.4f}")
print(f"Alpha0: {alpha0:.2f}°")
print(f"Cl at stall: {CL[stall]:.4f}")
print(f"Cl at 0 AoA: {coefs_cl[1]:.4f}")
print(f"Oswald efficiency factor (fit): {e_fit:.4f}")
print("===============================")

