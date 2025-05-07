import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants

ruta_archivo = "/content/drive/MyDrive/LAB/25-04-24-19-34.data"
datos_tiempo_ns = np.loadtxt(ruta_archivo, usecols=0)

# --- Sección 1: Ajuste exponencial con fondo ---
# Filtrado y conversión a microsegundos
tiempo_filtro_ns = 20000
datos_us = datos_tiempo_ns[datos_tiempo_ns < tiempo_filtro_ns] / 1000.0

# Histograma
ancho_bin_us = 0.5
tiempo_max_us = tiempo_filtro_ns / 1000.0
intervalos_histograma = np.arange(0, tiempo_max_us + ancho_bin_us, ancho_bin_us)
frecuencia_hist, bordes_bin = np.histogram(datos_us, bins=intervalos_histograma)
tiempos_datos = bordes_bin[:-1] + ancho_bin_us / 2.0
frecuencia_datos = frecuencia_hist

# Incertidumbres para el ajuste (sqrt(N))
incertidumbre_frecuencia = np.sqrt(frecuencia_datos)
incertidumbre_frecuencia[incertidumbre_frecuencia == 0] = 1.0

# Definir modelo con fondo: A·exp(-t/τ) + B
def funcion_ajuste_fondo(t, A, tau, B):
    return A * np.exp(-t / tau) + B

# Estimaciones iniciales
B_inicial = np.mean(frecuencia_datos[-5:])
A_inicial = max(1.0, frecuencia_datos[0] - B_inicial)
tau_inicial = 2.2

parametros_iniciales = [A_inicial, tau_inicial, B_inicial]
limites_ajuste = ([0, 0, 0], [np.inf, np.inf, np.inf])

# Ajuste curve_fit
parametros_optimizados, covarianza_parametros = curve_fit(
    funcion_ajuste_fondo, tiempos_datos, frecuencia_datos,
    p0=parametros_iniciales, sigma=incertidumbre_frecuencia,
    absolute_sigma=True, bounds=limites_ajuste
)
A_ajuste, tau_ajuste_fondo, B_ajuste = parametros_optimizados
A_error, tau_error_fondo, B_error = np.sqrt(np.diag(covarianza_parametros))

# Preparar curva suave para graficar ajuste
tiempos_suaves = np.linspace(0, tiempo_max_us, 500)
etiqueta_ajuste_fondo = f'Ajuste: A·e^(-t/τ) + B\nτ = {tau_ajuste_fondo:.2f} ± {tau_error_fondo:.2f} µs'

# Gráficas: datos + ajuste y residuales
fig_fondo, (ax1_fondo, ax2_fondo) = plt.subplots(
    2, 1, figsize=(7, 7), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

# Gráfica principal del ajuste con fondo
ax1_fondo.errorbar(tiempos_datos, frecuencia_datos, yerr=incertidumbre_frecuencia, fmt='o', ms=4,
                   label='Datos (Cuentas/bin)', alpha=0.8)
ax1_fondo.plot(tiempos_suaves, funcion_ajuste_fondo(tiempos_suaves, *parametros_optimizados), 'r-', lw=2,
               label=etiqueta_ajuste_fondo)
ax1_fondo.set_yscale('log')
ax1_fondo.set_ylabel('Frecuencia de decaimientos', fontsize=13)
ax1_fondo.set_title('Frecuencia de decaimientos vs. tiempo\n(Ajuste exponencial con fondo)')
ax1_fondo.legend(fontsize='small')
ax1_fondo.grid(True, linestyle=':', alpha=0.6)

# Gráfica de residuales normalizados del ajuste con fondo
residuales_fondo = (frecuencia_datos - funcion_ajuste_fondo(tiempos_datos, *parametros_optimizados)) / incertidumbre_frecuencia
ax2_fondo.axhline(0, color='black', lw=1, linestyle='--')
ax2_fondo.errorbar(tiempos_datos, residuales_fondo, yerr=1, fmt='o', ms=4, color='orange')
ax2_fondo.set_xlabel('Tiempo [µs]', fontsize=13)
ax2_fondo.set_ylabel('Residuales normalizados', fontsize=11)
ax2_fondo.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('ajuste_fondo.pdf')
plt.show()

# --- Sección 2: Cálculo de G_F / (hbar c)^3 ---
# Datos experimentales del ajuste anterior
tau_experimental_us = tau_ajuste_fondo
tau_error_experimental_us = tau_error_fondo

# Conversión a segundos
tau_s = tau_experimental_us * 1e-6
tau_error_s = tau_error_experimental_us * 1e-6

# Constantes
hbar_GeV_s = constants.hbar / constants.eV * 1e-9
m_mu_GeV = constants.physical_constants['muon mass energy equivalent in MeV'][0] * 1e-3

# Conversión de tau a GeV^-1
tau_GeV_inv = tau_s / hbar_GeV_s
tau_error_GeV_inv = tau_error_s / hbar_GeV_s

# Cálculo de G_F en GeV^-2
G_F_calculado = np.sqrt(192 * np.pi**3 / (tau_GeV_inv * m_mu_GeV**5))

# Propagación de error para G_F
G_F_error_calculado = (G_F_calculado / (2 * tau_GeV_inv)) * tau_error_GeV_inv

# --- Sección 3: Cálculo de rho ---
# Datos observados
tau_observado_s = tau_ajuste_fondo
tau_error_observado_s = tau_error_fondo

# Vidas medias aceptadas
tau_mas_s = 2.1969811
tau_mas_error_s = 0.0000022

tau_menos_s = 2.043
tau_menos_error_s = 0.003

# Cálculo de rho
# Numerador y denominador de la fórmula despejada:
a_rho = tau_menos_s
b_rho = tau_mas_s
T_rho = tau_observado_s

numerador_rho = a_rho * b_rho - T_rho * b_rho
denominador_rho = T_rho * a_rho - a_rho * b_rho
rho_calculado = numerador_rho / denominador_rho

# Derivadas parciales para propagación de error de rho
d_rho_dT = (-b_rho * denominador_rho - numerador_rho * a_rho) / denominador_rho**2
d_rho_da = (b_rho * denominador_rho - numerador_rho * (T_rho - b_rho)) / denominador_rho**2
d_rho_db = (a_rho * denominador_rho - numerador_rho * (T_rho - a_rho)) / denominador_rho**2

# Propagación de error para rho
rho_error_calculado = np.sqrt(
    (d_rho_dT * tau_error_observado_s)**2 +
    (d_rho_da * tau_menos_error_s)**2 +
    (d_rho_db * tau_mas_error_s)**2
)

# --- Sección 4: Segundo ajuste ln(N/N0) vs t ---
# Carga y filtrado
tiempo_filtro_ns_lin = 10000
datos_us_lin = datos_tiempo_ns[datos_tiempo_ns < tiempo_filtro_ns_lin] / 1000.0

# Histograma
ancho_bin_us_lin = 0.5
tiempo_max_us_lin = tiempo_filtro_ns_lin / 1000.0
intervalos_histograma_lin = np.arange(0, tiempo_max_us_lin + ancho_bin_us_lin, ancho_bin_us_lin)
frecuencia_hist_lin, bordes_bin_lin = np.histogram(datos_us_lin, bins=intervalos_histograma_lin)
tiempos_centros_lin = bordes_bin_lin[:-1] + ancho_bin_us_lin / 2.0

# Corrección de fondo (usando B_ajuste del primer ajuste)
frecuencia_corregida_lin = frecuencia_hist_lin - B_ajuste
frecuencia_corregida_lin[frecuencia_corregida_lin <= 0] = np.nan

# Normalizar para que ln(N/N0) pase por cero
N0_lin = frecuencia_corregida_lin[~np.isnan(frecuencia_corregida_lin)][0]
ln_normalizado_lin = np.log(frecuencia_corregida_lin / N0_lin)

# Ajuste lineal sin intercepto: ln_norm = m * t
valido_lin = ~np.isnan(ln_normalizado_lin)
tiempos_ajuste_lin = tiempos_centros_lin[valido_lin]
y_ajuste_lin = ln_normalizado_lin[valido_lin]

# Incertidumbre en y para el ajuste lineal y propagación a τ
incertidumbre_y_lin = 1 / np.sqrt(frecuencia_corregida_lin[valido_lin])

# Ajuste lineal ponderado (sin intercepto)
pesos_lin = 1 / incertidumbre_y_lin**2
m_lin = np.sum(pesos_lin * tiempos_ajuste_lin * y_ajuste_lin) / np.sum(pesos_lin * tiempos_ajuste_lin**2)
S_lin = np.sum(pesos_lin * tiempos_ajuste_lin**2)
m_error_lin = np.sqrt(1 / S_lin) if S_lin > 0 else np.nan

tau_lin = -1 / m_lin if m_lin != 0 else np.nan
tau_error_lin = abs(m_error_lin / m_lin**2) if m_lin != 0 else np.nan


# Gráficas del ajuste lineal
fig_lin, (ax1_lin, ax2_lin) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1_lin.errorbar(tiempos_ajuste_lin, y_ajuste_lin, yerr=incertidumbre_y_lin, fmt='o',
                 label='ln(N/N₀)', alpha=0.8)
ax1_lin.plot(tiempos_ajuste_lin, m_lin * tiempos_ajuste_lin, 'r-', label=f'Ajuste: m={m_lin:.5f} µs⁻¹')
ax1_lin.set_ylabel('ln(N/N₀)')
ax1_lin.set_title('Ajuste lineal a ln(N/N₀) vs. tiempo')
ax1_lin.legend()
ax1_lin.grid(True, linestyle=':', alpha=0.6)

residuales_lin = (y_ajuste_lin - m_lin * tiempos_ajuste_lin) / incertidumbre_y_lin
ax2_lin.axhline(0, ls='--', color='black')
ax2_lin.errorbar(tiempos_ajuste_lin, residuales_lin, yerr=1, fmt='o', color='orange')
ax2_lin.set_xlabel('Tiempo [µs]')
ax2_lin.set_ylabel('Residuales\nNormalizados')
ax2_lin.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('ajuste_lineal_sin_fondo.pdf')
plt.show()


# --- Sección 5: Cálculos de precisión y exactitud ---
# Para el primer ajuste (con fondo)
tau_teorico_1 = 2.1237
tau_experimental_1 = tau_ajuste_fondo
incertidumbre_tau_1 = tau_error_fondo

precision_tau_1 = incertidumbre_tau_1 / tau_experimental_1
exactitud_tau_1 = abs(tau_teorico_1 - tau_experimental_1) / incertidumbre_tau_1

# Para G_F (usando resultados del primer ajuste de tau)
G_F_teorico = 1.16639e-5
G_F_experimental = G_F_calculado
incertidumbre_G_F = G_F_error_calculado

precision_G_F = incertidumbre_G_F / G_F_experimental if G_F_experimental !=0 else np.nan
exactitud_G_F = abs(G_F_teorico - G_F_experimental) / incertidumbre_G_F if incertidumbre_G_F !=0 else np.nan


# Para el segundo ajuste (lineal)
tau_teorico_2 = 2.1237
tau_experimental_2 = tau_lin
incertidumbre_tau_2 = tau_error_lin

precision_tau_2 = incertidumbre_tau_2 / tau_experimental_2 if tau_experimental_2 !=0 else np.nan
exactitud_tau_2 = abs(tau_teorico_2 - tau_experimental_2) / incertidumbre_tau_2 if incertidumbre_tau_2 !=0 else np.nan
