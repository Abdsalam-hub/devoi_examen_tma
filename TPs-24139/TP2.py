import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 2.1 Analyse spectrale d’un son pur
# ==========================================

# 1. Paramètres du signal
fs = 44100          # Fréquence d'échantillonnage
f1 = 440            # La3
f2 = 880            # La4
duration = 0.5      # Durée (s)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Génération du signal (somme de deux fréquences)
signal_pur = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# 2. Calcul de la FFT
n = len(signal_pur)
fft_values = np.fft.fft(signal_pur)
# Fréquences associées aux coefficients
freqs = np.fft.fftfreq(n, 1/fs)

# 3. Tracé du module (Spectre d'amplitude)
# On ne garde que la moitié positive du spectre
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(freqs[:n//2], np.abs(fft_values)[:n//2])
plt.title("2.1 Spectre du signal pur (La3 + La4)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1500) # Zoom sur la zone d'intérêt
plt.grid(True)

# ==========================================
# 2.2 Identification et Filtrage d’un bruit
# ==========================================

# 1. Ajout d'un sifflement haute fréquence (5000 Hz)
f_bruit = 5000
bruit_sifflement = 0.3 * np.sin(2 * np.pi * f_bruit * t)
signal_bruite = signal_pur + bruit_sifflement

# 2. Visualisation du spectre bruité
fft_bruite = np.fft.fft(signal_bruite)
plt.subplot(3, 1, 2)
plt.plot(freqs[:n//2], np.abs(fft_bruite)[:n//2], color='red')
plt.title("2.2 Spectre du signal bruité (Pic à 5000 Hz visible)")
plt.xlabel("Fréquence (Hz)")
plt.grid(True)

# 3. Filtrage : Mise à zéro des coefficients autour de 5000 Hz
# On définit une zone de coupure (ex: entre 4800 et 5200 Hz)
fft_filtree = fft_bruite.copy()
# On applique le filtre sur les fréquences positives et négatives (symétrie)
masque_bruit = (np.abs(freqs) > 4800) & (np.abs(freqs) < 5200)
fft_filtree[masque_bruit] = 0

# 4. Transformée de Fourier Inverse (IFFT)
signal_filtre = np.fft.ifft(fft_filtree).real

# 5. Comparaison visuelle
plt.subplot(3, 1, 3)
plt.plot(t[:1000], signal_pur[:1000], label="Original", alpha=0.8)
plt.plot(t[:1000], signal_filtre[:1000], label="Filtré", linestyle='--')
plt.title("2.2 Comparaison Temporelle (Zoom)")
plt.xlabel("Temps (s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Analyse 2.2 : Le pic à 5000 Hz a été supprimé dans le domaine fréquentiel.")
