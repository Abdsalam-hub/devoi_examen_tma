import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1.1 Génération d’un signal sinusoïdal
# ==========================================

# Paramètres du signal
f0 = 10         # Fréquence (Hz)
A = 1           # Amplitude
phi = 0         # Phase
fs = 100        # Fréquence d'échantillonnage (Hz)
duration = 1    # Durée en secondes

# Création du vecteur temps et du signal x(t)
t = np.arange(0, duration, 1/fs)
x = A * np.sin(2 * np.pi * f0 * t + phi)

# Visualisation
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, x, label='Signal pur $x(t)$', color='blue')
plt.title("1.1 Génération du Signal Sinusoïdal")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# ==========================================
# 1.2 Ajout de bruit et analyse
# ==========================================

# Génération d'un Bruit Blanc Gaussien (BBG)
# On choisit un écart-type de 0.5 pour voir l'impact du bruit
bruit = np.random.normal(0, 0.5, len(x))

# Création du signal bruité y(t)
y = x + bruit

# Visualisation comparative
plt.subplot(3, 1, 2)
plt.plot(t, y, label='Signal bruité $y(t)$', alpha=0.6, color='orange')
plt.plot(t, x, label='Signal pur $x(t)$', linewidth=2, color='blue')
plt.title("1.2 Signal avec Bruit Blanc Gaussien")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# ==========================================
# 1.3 Signaux élémentaires et Convolution
# ==========================================

# 1. Création du signal porte (Rect)
rect = np.zeros(100)
rect[20:41] = 1  # Valant 1 entre n=20 et n=40

# 2. Calcul de la convolution du signal porte avec lui-même
conv_result = np.convolve(rect, rect, mode='full')

# 3. Visualisation de la convolution
# L'axe des indices pour la convolution 'full' est de taille (len(rect)*2 - 1)
indices_conv = np.arange(len(conv_result))

plt.subplot(3, 1, 3)
plt.plot(indices_conv, conv_result, color='green', label='Résultat : Triangle')
plt.title("1.3 Convolution d'une Porte par elle-même")
plt.xlabel("Indices $n$")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# --- Analyse de la forme ---
print("Analyse 1.3 : La convolution de deux fonctions 'Porte' identiques")
print("donne un signal de forme TRIANGULAIRE.")
