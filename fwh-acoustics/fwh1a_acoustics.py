import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib.ticker import FuncFormatter

from openfoam_fwh_adapter import fwh1a_arrays_from_openfoam

# CONFIGURACIÓN VALORES INICIALES MODELO
class Config:
    dir_surface = r"C:/Users/jaime/Documents/Universidad/Master/25-26/TFM/Simulaciones/acus_H/postProcessing/fwhSurface"
    surfNom_cont = "propellerFWHSolid"

    press = "p"
    press_cin = True

    rho0 = 1025.0
    c0 = 1482.0
    p_ref = 1e-6

    t_ini = 0
    t_fin = 8.47

    vel_mode = "rotacion"  # "movimiento" o "rotacion"

    rpm = 70.8
    n_palas = 4
    eje_rotacion = (1.0, 0.0, 0.0)
    origen_rotacion = (0.0, 0.0, 0.0)

    # DEFINICIÓN PUNTOS OBSERVADORES
    obs = np.array([
        [275.0, 0.0, 0.0],
        [-275.0, 0.0, 0.0],
        [550.0, 0.0, 0.0],
        [-550.0, 0.0, 0.0],
    ])

    dir_results = r"C:/Users/jaime/Documents/Universidad/Master/25-26/TFM/Simulaciones/acus_H/acoustic_results"

# PREPARACIÓN DATOS HIDRODINÁMICOS PARA CÁLCULO ACÚSTICO
def prep_inputs(tiempos, y, n, area, p, v):
    if Config.press_cin:
        p = Config.rho0 * p

    p = p - p.mean(axis=0, keepdims=True)

    n_norm = np.linalg.norm(n, axis=2)
    n = n / np.maximum(n_norm[:, :, None], 1e-300)

    a = np.gradient(v, tiempos, axis=0, edge_order=2)
    p_prima = np.gradient(p, tiempos, axis=0, edge_order=2)

    return y, n, area, p, v, a, p_prima

# DEFINICIÓN DE LA FUNCIÓN DE INTERPOLACIÓN
def interp_vector(tau, tiempos, valores):
    return np.array([np.interp(tau, tiempos, valores[:, i]) for i in range(3)])

# CÁLCULO CONTRIBUCIÓN DE CADA CARA AL OBSERVADOR A PARTIR DE LA FÓRMULA FWH 1A
def contri_cara(obs, t_obs, id_cara, tiempos, y, n, area, p, v, a, p_prima):
    y_hist = y[:, id_cara, :]
    r_hist = np.linalg.norm(obs[None, :] - y_hist, axis=1)
    t_llegada = tiempos + r_hist / Config.c0

    if t_obs < t_llegada[0] or t_obs > t_llegada[-1]:
        return 0.0

    # TIEMPO DE RETARDO PARA LLEGADA DE ONDA ACÚSTICA DESDE LA CARA AL OBSERVADOR
    tau = np.interp(t_obs, t_llegada, tiempos)

    # VARIABLES INTERPOLADAS EN EL TIEMPO DE RETARDO
    y_tau = interp_vector(tau, tiempos, y[:, id_cara, :])
    n_tau = interp_vector(tau, tiempos, n[:, id_cara, :])
    v_tau = interp_vector(tau, tiempos, v[:, id_cara, :])
    a_tau = interp_vector(tau, tiempos, a[:, id_cara, :])

    p_tau = np.interp(tau, tiempos, p[:, id_cara])
    p_prima_tau = np.interp(tau, tiempos, p_prima[:, id_cara])
    area_tau = np.interp(tau, tiempos, area[:, id_cara])

    vec_r = obs - y_tau
    r = np.linalg.norm(vec_r)

    if r <= 1e-300:
        return 0.0

    r_hat = vec_r / r

    # CÁLCULO DE LOS TÉRMINOS DE LA FÓRMULA FWH 1A
    M = v_tau / Config.c0
    M2 = np.dot(M, M)
    Mr = np.dot(M, r_hat)
    denom = max(1.0 - Mr, 1e-6)

    Mdot = a_tau / Config.c0
    Mdot_r = np.dot(Mdot, r_hat)

    vn = np.dot(v_tau, n_tau)
    vdot_n = np.dot(a_tau, n_tau)

    L = p_tau * n_tau
    Ldot = p_prima_tau * n_tau

    Lr = np.dot(L, r_hat)
    Lm = np.dot(L, M)
    Ldot_r = np.dot(Ldot, r_hat)

    mov = r * Mdot_r + Config.c0 * (Mr - M2)

    # TÉRMINO DE ESPESOR (MONOPOLAR)
    espesor = (
        Config.rho0 * vdot_n / (r * denom**2)
        + Config.rho0 * vn * mov / (r**2 * denom**3)
    )
    
    # TÉRMINO DE CARGA (DIPOLAR)
    carga = (
        Ldot_r / (Config.c0 * r * denom**2)
        + (Lr - Lm) / (r**2 * denom**2)
        + Lr * mov / (Config.c0 * r**2 * denom**3)
    )

    return (espesor + carga) * area_tau / (4.0 * np.pi)

# CÁLCULO PRESIÓN ACÚSTICA EN LOS OBSERVADORES
def calc_press_fwh1a(tiempos, y, n, area, p, v, obs):
    y, n, area, p, v, a, p_prima = prep_inputs(tiempos, y, n, area, p, v)

    nt = len(tiempos)
    nf = y.shape[1]

    obs = np.asarray(obs, dtype=float)
    p_obs = np.zeros(nt)

    for it, t_obs in enumerate(tiempos):
        total = 0.0

        for id_cara in range(nf):
            total += contri_cara(
                obs, t_obs, id_cara,
                tiempos, y, n, area, p, v, a, p_prima,
            )

        p_obs[it] = total

    return p_obs

# CÁLCULO DE SPL
def calc_spl(senal):
    senal = np.asarray(senal, dtype=float)

    # Se elimina offset/DC antes del RMS acustico.
    senal_fluct = senal - np.mean(senal)

    rms = np.sqrt(np.mean(senal**2))
    return 20.0 * np.log10(max(rms, 1e-300) / Config.p_ref)

# CÁLCULO ESPECTRO DE POTENCIA MEDIANTE WELCH
def calc_espectro(tiempos, senal):
    senal = np.asarray(senal, dtype=float)

    # Importante para no contaminar bajas frecuencias con la componente media.
    senal_fluct = senal - np.mean(senal)

    dt = np.mean(np.diff(tiempos))
    fs = 1.0 / dt

    nperseg = min(max(len(senal_fluct) // 2, 4), 1024)
    freqs, psd = welch(senal_fluct, fs=fs, window="hann", nperseg=nperseg, detrend=False, scaling="density")

    spl_dens = 10.0 * np.log10(np.maximum(psd, 1e-300) / Config.p_ref**2)

    return freqs, spl_dens

# CÁLCULO DE PRMS A PARTIR DE LA FFT
def calc_fft_prms(tiempos, senal):
    dt = np.mean(np.diff(tiempos))

    # Se elimina la componente media antes de ventanear.
    # Esto evita un pico artificial en f=0 y leakage hacia bajas frecuencias.
    senal_fluct = senal - np.mean(senal)

    ventana = np.hanning(len(senal_fluct))
    senal_win = senal_fluct * ventana

    fft_vals = np.fft.rfft(senal_win)
    freqs = np.fft.rfftfreq(len(senal_win), d=dt)

    # Correccion de amplitud por ganancia coherente de la ventana.
    amp_peak = 2.0 * np.abs(fft_vals) / np.sum(ventana)

    # La componente DC no debe duplicarse.
    if len(amp_peak) > 0:
        amp_peak[0] *= 0.5

    # Si N es par, la frecuencia de Nyquist tampoco debe duplicarse.
    if len(senal_win) % 2 == 0 and len(amp_peak) > 1:
        amp_peak[-1] *= 0.5

    prms = amp_peak / np.sqrt(2.0)
    prms = np.maximum(prms, 1e-300)

    return freqs, prms

# CÁLCULO DE FRECUENCIA DE PASO DE PALA Y ARMÓNICOS
def frec_paso_pala():
    eje = Config.rpm / 60.0
    bpf = eje * Config.n_palas

    return {
        "eje": eje,
        "BPF": bpf,
        "2BPF": 2.0 * bpf,
        "3BPF": 3.0 * bpf,
    }
    
def main():
    os.makedirs(Config.dir_results, exist_ok=True)

    tiempos, y, n, area, p, v = fwh1a_arrays_from_openfoam(
        dir_surface=Config.dir_surface,
        campo_press=Config.press,
        surfNom_cont=Config.surfNom_cont,
        t_ini=Config.t_ini,
        t_fin=Config.t_fin,
        vel_mode=Config.vel_mode,
        rpm=Config.rpm,
        eje=Config.eje_rotacion,
        origen=Config.origen_rotacion,
    )

    print("OpenFOAM surface data cargado")
    print("tiempos:", tiempos)
    print("y:", y.shape)
    print("n:", n.shape)
    print("area:", area.shape)
    print("p:", p.shape)
    print("v:", v.shape)
    print("Frecuencias:", frec_paso_pala())

    for i, observer in enumerate(Config.obs, start=1):
        print(f"\nObservador {i}: {observer}")

        p_acus = calc_press_fwh1a(
            tiempos=tiempos,
            y=y,
            n=n,
            area=area,
            p=p,
            v=v,
            obs=observer,
        )

        spl = calc_spl(p_acus)
        freqs_welch, espectro = calc_espectro(tiempos, p_acus)
        freqs_fft, prms_fft = calc_fft_prms(tiempos, p_acus)

        print(f"SPL = {spl:.2f} dB re 1 uPa")
        #print(freqs_fft)

        np.savetxt(
            os.path.join(Config.dir_results, f"obs_{i}_press.csv"),
            np.column_stack([tiempos, p_acus]),
            delimiter=",",
            header="time,p_acus_Pa",
            comments="",
        )

        plt.figure()
        plt.plot(tiempos, p_acus)
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Presión acústica [Pa]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.dir_results, f"obs_{i}_press.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(freqs_welch, espectro)
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Nivel PSD [dB re 1 uPa^2/Hz]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.dir_results, f"obs_{i}_espectro.png"), dpi=200)
        plt.close()

        # Para escala logaritmica hay que eliminar f = 0.
        mask = freqs_fft > 0
        
        plt.figure()
        plt.plot(freqs_fft[mask], prms_fft[mask])
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Presión RMS espectral [Pa]")
        plt.xscale("log")
        plt.yscale("log")
        
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
        
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.dir_results, f"obs_{i}_fft_prms.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    main()