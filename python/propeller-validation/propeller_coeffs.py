"""
Cálculo de coeficientes hidrodinámicos de la hélice
====================================================
Lee el fichero de fuerzas generado por OpenFOAM (función 'forces')
y calcula K_T, K_Q y eta_0.

Parámetros del buque/hélice (ajustar según el caso real):
    - D    : diámetro de la hélice [m]
    - n    : velocidad de giro [rev/s]
    - rho  : densidad del fluido [kg/m³]
    - V_A  : velocidad de avance efectiva en el plano de la hélice [m/s]
    - T_transitorio : tiempo hasta el que se descarta el transitorio [s]

El eje de rotación de la hélice es X (componente axial = Fx, par = Mx).

Estructura esperada del fichero force.dat de OpenFOAM:
    # Time  Fx  Fy  Fz  Mx  My  Mz  (fuerzas de presión + viscosas)
    OpenFOAM escribe dos bloques: (pressure) y (viscous)
    Este script suma ambas contribuciones automáticamente.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PARÁMETROS — ajustar según el caso
# =============================================================================

RUTA_FORCES = Path("postProcessing/forces/0/force.dat")  # ruta al fichero

D   = 8.9          # diámetro de la hélice [m]
n   = 70.8 / 60.0  # velocidad de giro [rev/s]  (70.8 rpm -> rev/s)
rho = 1025.0       # densidad del agua de mar [kg/m³]
V_A = None         # velocidad de avance [m/s]; si None, se pide por pantalla

T_TRANSITORIO = 1.694  # descartar primeras 2 revoluciones (2 x 0.847 s)

# =============================================================================
# LECTURA DEL FICHERO DE FUERZAS
# =============================================================================

def leer_forces_openfoam(ruta):
    """
    Lee el fichero force.dat de OpenFOAM Foundation (versión 13).
    Formato esperado por línea de datos:
        t  (Fpx Fpy Fpz)  (Fvx Fvy Fvz)  (Mpx Mpy Mpz)  (Mvx Mvy Mvz)
    donde p = presión, v = viscoso.
    Devuelve un DataFrame con columnas:
        t, Fx, Fy, Fz, Mx, My, Mz  (suma presión + viscoso)
    """
    tiempos, Fx, Fy, Fz, Mx, My, Mz = [], [], [], [], [], [], []

    with open(ruta, "r") as f:
        for linea in f:
            linea = linea.strip()
            # saltar cabeceras y líneas vacías
            if not linea or linea.startswith("#"):
                continue
            # reemplazar paréntesis para poder parsear
            linea = linea.replace("(", " ").replace(")", " ")
            valores = linea.split()
            if len(valores) < 13:
                continue
            try:
                t   = float(valores[0])
                # fuerzas: presión + viscoso
                fx  = float(valores[1])  + float(valores[4])
                fy  = float(valores[2])  + float(valores[5])
                fz  = float(valores[3])  + float(valores[6])
                # momentos: presión + viscoso
                mx  = float(valores[7])  + float(valores[10])
                my  = float(valores[8])  + float(valores[11])
                mz  = float(valores[9])  + float(valores[12])
                tiempos.append(t)
                Fx.append(fx); Fy.append(fy); Fz.append(fz)
                Mx.append(mx); My.append(my); Mz.append(mz)
            except ValueError:
                continue

    df = pd.DataFrame({
        "t" : tiempos,
        "Fx": Fx, "Fy": Fy, "Fz": Fz,
        "Mx": Mx, "My": My, "Mz": Mz,
    })
    return df

# =============================================================================
# CÁLCULO DE COEFICIENTES
# =============================================================================

def calcular_coeficientes(df, D, n, rho, V_A, T_transitorio):
    """
    Calcula K_T, K_Q y eta_0 a partir del DataFrame de fuerzas.
    Eje de rotación: X  =>  empuje = Fx,  par = |Mx|
    """
    # filtrar transitorio
    df_reg = df[df["t"] >= T_transitorio].copy()
    if df_reg.empty:
        raise ValueError(
            f"No hay datos tras T_transitorio = {T_transitorio} s. "
            "Comprueba el valor o la ruta del fichero."
        )

    # empuje y par medios
    T_medio = df_reg["Fx"].mean()
    Q_medio = df_reg["Mx"].abs().mean()  # valor absoluto por convención de signo

    # coeficientes adimensionales
    K_T  = T_medio / (rho * n**2 * D**4)
    K_Q  = Q_medio / (rho * n**2 * D**5)

    # coeficiente de avance
    J    = V_A / (n * D)

    # eficiencia en aguas abiertas
    eta0 = (J * K_T) / (2 * np.pi * K_Q) if K_Q != 0 else float("nan")

    return {
        "T_medio [N]"   : T_medio,
        "Q_medio [N·m]" : Q_medio,
        "J [-]"         : J,
        "K_T [-]"       : K_T,
        "K_Q [-]"       : K_Q,
        "eta_0 [-]"     : eta0,
        "df_regimen"    : df_reg,
    }

# =============================================================================
# REPRESENTACIÓN GRÁFICA
# =============================================================================

def graficar_convergencia(df, df_reg, T_transitorio):
    """
    Muestra la evolución temporal de empuje y par,
    marcando la zona de promediado.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    T_rev = 1.0 / (70.8 / 60.0)  # periodo de una revolución

    for ax, col, etiqueta, unidad in zip(
        axes,
        ["Fx", "Mx"],
        ["Empuje T", "Par Q"],
        ["N", "N·m"],
    ):
        ax.plot(df["t"], df[col], color="steelblue", lw=0.8, label="Señal completa")
        ax.axvline(T_transitorio, color="tomato", ls="--", lw=1.2,
                   label=f"Inicio promediado ($t$ = {T_transitorio:.3f} s)")
        ax.axhline(df_reg[col].mean(), color="seagreen", ls="-", lw=1.2,
                   label=f"Media = {df_reg[col].mean():.2f} {unidad}")
        ax.set_ylabel(f"{etiqueta} [{unidad}]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Tiempo [s]")
    fig.suptitle("Convergencia del empuje y par de la hélice", fontsize=11)
    plt.tight_layout()
    plt.savefig("convergencia_fuerzas.png", dpi=150)
    plt.show()
    print("Figura guardada: convergencia_fuerzas.png")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # velocidad de avance: pedir si no está definida arriba
    if V_A is None:
        V_A = float(input(
            "Introduce la velocidad de avance V_A en el plano de la hélice [m/s]: "
        ))

    print(f"\nLeyendo fichero: {RUTA_FORCES}")
    df = leer_forces_openfoam(RUTA_FORCES)
    print(f"  Registros leídos : {len(df)}")
    print(f"  Rango temporal   : {df['t'].min():.4f} — {df['t'].max():.4f} s")

    resultados = calcular_coeficientes(df, D, n, rho, V_A, T_TRANSITORIO)

    print("\n" + "="*45)
    print("  COEFICIENTES HIDRODINÁMICOS DE LA HÉLICE")
    print("="*45)
    for clave, valor in resultados.items():
        if clave == "df_regimen":
            continue
        print(f"  {clave:<20} {valor:>12.6f}")
    print("="*45)

    graficar_convergencia(df, resultados["df_regimen"], T_TRANSITORIO)
