import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib.ticker import FuncFormatter

from openfoam_fwh_adapter import fwh1a_arrays_from_openfoam

# CONFIGURACIÓN VALORES INICIALES MODELO
class Config:
    dir_surface = r"C:/Users/jaime/Desktop/prop/laminar/postProcessing/fwhSurface"
    surfNom_cont = "propellerFWHSolid"

    press = "p"
    press_cin = True

    rho0 = 1025.0
    c0 = 1482.0
    p_ref = 1e-6

    t_ini = 3.388
    t_fin = 8.47

    vel_mode = "rotacion"  # "movimiento" o "rotacion"

    rpm = 70.8
    n_palas = 4
    eje_rotacion = (1.0, 0.0, 0.0)
    origen_rotacion = (0.0, 0.0, 0.0)

    # Receptores de directividad: corona en el plano yz, eje x fijo.
    radio_corona = 275.0
    x_corona = 0.0
    angulos_corona_deg = np.arange(0.0, 360.0, 30.0)

    # DEFINICIÓN PUNTOS OBSERVADORES
    obs_corona = np.column_stack([
        np.full_like(angulos_corona_deg, x_corona, dtype=float),
        radio_corona * np.cos(np.deg2rad(angulos_corona_deg)),
        radio_corona * np.sin(np.deg2rad(angulos_corona_deg)),
    ])

    obs_axiales = np.array([
        [275.0, 0.0, 0.0],
        [-275.0, 0.0, 0.0],
        [550.0, 0.0, 0.0],
        [-550.0, 0.0, 0.0],
    ])

    obs = np.vstack([obs_corona, obs_axiales])

    dir_results = r"C:/Users/jaime/Desktop/prop/laminar/acoustic_results"

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

    rms = np.sqrt(np.mean(senal_fluct**2))
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

def calc_spl_tonal_fft(tiempos, senal, freq_objetivo):
    freqs, prms = calc_fft_prms(tiempos, senal)

    idx = np.argmin(np.abs(freqs - freq_objetivo))
    freq_bin = freqs[idx]
    prms_bin = prms[idx]

    spl = 20.0 * np.log10(max(prms_bin, 1e-300) / Config.p_ref)

    return spl, freq_bin, prms_bin


def plot_directividad_polar(angulos_deg, spl_vals, freq, nombre_archivo):
    theta = np.deg2rad(angulos_deg)
    spl_vals = np.asarray(spl_vals, dtype=float)

    theta_cerrado = np.r_[theta, theta[0]]
    spl_cerrado = np.r_[spl_vals, spl_vals[0]]

    rmin = 5.0 * np.floor((np.min(spl_vals) - 5.0) / 5.0)
    rmax = 5.0 * np.ceil((np.max(spl_vals) + 5.0) / 5.0)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(theta_cerrado, spl_cerrado, marker="o")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(rmin, rmax)
    ax.set_title(f"Directividad tonal a {freq:.2f} Hz")
    #ax.set_ylabel("SPL [dB re 1 uPa]")
    ax.text(
        -0.15, 0.5,
        "SPL [dB re 1 uPa]",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
    )
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=200)
    plt.close()


#def plot_decaimiento_axial(obs_axiales, spl_vals, nombre_archivo):
#    obs_axiales = np.asarray(obs_axiales, dtype=float)
#    spl_vals = np.asarray(spl_vals, dtype=float)

#    dist = np.linalg.norm(obs_axiales - np.asarray(Config.origen_rotacion), axis=1)

#    order = np.argsort(dist)
#    dist = dist[order]
#    spl_vals = spl_vals[order]

#    dist_ref = dist[0]
#    spl_ref = spl_vals[0]
#    ref_6db = spl_ref - 20.0 * np.log10(dist / dist_ref)

#    plt.figure()
#    plt.plot(dist, spl_vals, "o-", label="FWH")
#    plt.plot(dist, ref_6db, "--", label="-6 dB por duplicacion")
#    plt.xlabel("Distancia al disco de la helice [m]")
#    plt.ylabel("SPL [dB re 1 uPa]")
#    plt.grid(True)
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig(nombre_archivo, dpi=200)
#    plt.close()

def plot_decaimiento_axial(obs_axiales, spl_vals, nombre_archivo):
    obs_axiales = np.asarray(obs_axiales, dtype=float)
    spl_vals = np.asarray(spl_vals, dtype=float)

    x = obs_axiales[:, 0]
    dist = np.abs(x)

    plt.figure()

    for signo, etiqueta in [(1, "aguas arriba"), (-1, "aguas abajo")]:
        mask = np.sign(x) == signo

        if np.count_nonzero(mask) == 0:
            continue

        dist_lado = dist[mask]
        spl_lado = spl_vals[mask]

        order = np.argsort(dist_lado)
        dist_lado = dist_lado[order]
        spl_lado = spl_lado[order]

        plt.plot(dist_lado, spl_lado, "o-", label=f"FWH {etiqueta}")

        if len(dist_lado) >= 2:
            dist_ref = dist_lado[0]
            spl_ref = spl_lado[0]
            ref_6db = spl_ref - 20.0 * np.log10(dist_lado / dist_ref)

            plt.plot(
                dist_lado,
                ref_6db,
                "--",
                label=f"-6 dB por duplicacion {etiqueta}",
            )

    plt.xlabel("Distancia axial al disco de la helice [m]")
    plt.ylabel("SPL BPF [dB re 1 uPa]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=200)
    plt.close()

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

    frecs = frec_paso_pala()
    bpf = frecs["BPF"]
    bpf2 = frecs["2BPF"]

    p_acus_list = []
    spl_global_list = []
    spl_bpf_list = []
    spl_2bpf_list = []

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

        spl_bpf, freq_bpf_bin, _ = calc_spl_tonal_fft(tiempos, p_acus, bpf)
        spl_2bpf, freq_2bpf_bin, _ = calc_spl_tonal_fft(tiempos, p_acus, bpf2)

        p_acus_list.append(p_acus)
        spl_global_list.append(spl)
        spl_bpf_list.append(spl_bpf)
        spl_2bpf_list.append(spl_2bpf)

        print(f"SPL global = {spl:.2f} dB re 1 uPa")
        print(f"SPL BPF = {spl_bpf:.2f} dB re 1 uPa, bin {freq_bpf_bin:.3f} Hz")
        print(f"SPL 2BPF = {spl_2bpf:.2f} dB re 1 uPa, bin {freq_2bpf_bin:.3f} Hz")

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
        plt.ylabel("Presion acustica [Pa]")
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

        mask = freqs_fft > 0

        plt.figure()
        plt.plot(freqs_fft[mask], prms_fft[mask])
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Presion RMS espectral [Pa]")
        plt.xscale("log")
        plt.yscale("log")

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))

        plt.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(Config.dir_results, f"obs_{i}_fft_prms.png"), dpi=200)
        plt.close()

    n_corona = len(Config.obs_corona)

    spl_bpf_corona = np.asarray(spl_bpf_list[:n_corona])
    spl_2bpf_corona = np.asarray(spl_2bpf_list[:n_corona])

    plot_directividad_polar(
        Config.angulos_corona_deg,
        spl_bpf_corona,
        bpf,
        os.path.join(Config.dir_results, "directividad_BPF.png"),
    )

    plot_directividad_polar(
        Config.angulos_corona_deg,
        spl_2bpf_corona,
        bpf2,
        os.path.join(Config.dir_results, "directividad_2BPF.png"),
    )

    spl_axiales = np.asarray(spl_bpf_list[n_corona:])

    plot_decaimiento_axial(
        Config.obs_axiales,
        spl_axiales,
        os.path.join(Config.dir_results, "decaimiento_axial_BPF.png"),
    )

    np.savetxt(
        os.path.join(Config.dir_results, "directividad_BPF.csv"),
        np.column_stack([Config.angulos_corona_deg, spl_bpf_corona, spl_2bpf_corona]),
        delimiter=",",
        header="angulo_deg,SPL_BPF_dB_re_1uPa,SPL_2BPF_dB_re_1uPa",
        comments="",
    )

    #for i, observer in enumerate(Config.obs, start=1):
    #    print(f"\nObservador {i}: {observer}")

    #    p_acus = calc_press_fwh1a(
    #        tiempos=tiempos,
    #        y=y,
    #        n=n,
    #        area=area,
    #        p=p,
    #        v=v,
    #        obs=observer,
    #    )

    #    spl = calc_spl(p_acus)
    #    freqs_welch, espectro = calc_espectro(tiempos, p_acus)
    #    freqs_fft, prms_fft = calc_fft_prms(tiempos, p_acus)

    #    print(f"SPL = {spl:.2f} dB re 1 uPa")
    #    #print(freqs_fft)

    #    np.savetxt(
    #        os.path.join(Config.dir_results, f"obs_{i}_press.csv"),
    #        np.column_stack([tiempos, p_acus]),
    #        delimiter=",",
    #        header="time,p_acus_Pa",
    #        comments="",
    #    )

    #    plt.figure()
    #    plt.plot(tiempos, p_acus)
    #    plt.xlabel("Tiempo [s]")
    #    plt.ylabel("Presión acústica [Pa]")
    #    plt.grid(True)
    #    plt.tight_layout()
    #    plt.savefig(os.path.join(Config.dir_results, f"obs_{i}_press.png"), dpi=200)
    #    plt.close()

    #    plt.figure()
    #    plt.plot(freqs_welch, espectro)
    #    plt.xlabel("Frecuencia [Hz]")
    #    plt.ylabel("Nivel PSD [dB re 1 uPa^2/Hz]")
    #    plt.grid(True)
    #    plt.tight_layout()
    #    plt.savefig(os.path.join(Config.dir_results, f"obs_{i}_espectro.png"), dpi=200)
    #    plt.close()

        # Para escala logaritmica hay que eliminar f = 0.
    #    mask = freqs_fft > 0
        
    #    plt.figure()
    #    plt.plot(freqs_fft[mask], prms_fft[mask])
    #    plt.xlabel("Frecuencia [Hz]")
    #    plt.ylabel("Presión RMS espectral [Pa]")
    #    plt.xscale("log")
    #    plt.yscale("log")
        
    #    ax = plt.gca()
    #    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
        
    #    plt.grid(True, which="both")
    #    plt.tight_layout()
    #    plt.savefig(os.path.join(Config.dir_results, f"obs_{i}_fft_prms.png"), dpi=200)
    #    plt.close()

if __name__ == "__main__":
    main()