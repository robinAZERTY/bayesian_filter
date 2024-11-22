import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import de la classe Ekf
from ekf.ekf import Ekf  # Remplacez Ekf par le nom de la classe que vous voulez importer

# Définition de la fonction de prédiction
def f(X, U, C):
    """
    Modèle du four, calcul du changement de température
    X : température actuelle
    U : puissance du chauffage
    C : paramètres du system (dt, thermal_inertia, thermal_losses, external_temperature)
    """
    ret = np.array([X[0] + C[0] * (U[0] / C[1] - (X[0] - C[3]) / C[2])])
    return ret

# Fonction de mesure (ici, la température mesurée)
def h(X, C=None):
    """
    Fonction de mesure, on suppose que le capteur mesure directement la température.
    """
    return X

dt_u = 1  # Intervalle de temps (en secondes) pour les commandes
dt_z = 10  # Intervalle de temps (en secondes) pour les mesures
# Initialisation de l'état initial (température initiale)
x_init = 20.0  # Température initiale en °C

# Bruit des commandes (puissance de chauffage)
control_noise_power = 500  # Bruit de la commande de chauffage en W*s^1/2
# Bruit du capteur de température
sensor_noise = 25  # Bruit de mesure de température en °C*s^1/2


R = np.array([[sensor_noise**2/dt_z]])  # Covariance du bruit de mesure


# Initialisation du filtre de Kalman étendu
ekf = Ekf(1)
ekf.x = np.array([0])  # Initialisation de l'état du filtre
P0 = np.eye(1) * 200*200  # Grande incertitude initiale
ekf.P = P0.copy()  # Grande incertitude initiale
ekf.c = np.array([dt_u, 10000, 100, 20])  # Paramètres du système
ekf.cov_u = np.array([[control_noise_power**2/dt_u]])  # Covariance du bruit de commande

# Données d'entrée (puissance de chauffage et mesures de température)
T = 500  # temps total de simulation
t = np.arange(0, T, dt_u)
# Simuler l'évolution de la température du four
true_temperatures = np.zeros(len(t))
measured_temperatures = np.zeros(len(t))
measured_temperatures[:] = np.nan
t_measurement = np.zeros(len(t))
control_inputs = np.zeros(len(t))

# Température initiale du four
true_temperatures[0] = x_init

# Générer des mesures bruyantes et des commandes
last_z = 0
for i in range(1, len(t)):
    this_t = t[i]
    if this_t < 300:
        U = 20*this_t+1000
    else:
        U = 5000  # Arrêt du chauffage
    control_inputs[i] = U
    true_temperatures[i] = f(np.array([true_temperatures[i-1]]), np.array([U]), np.array(ekf.c))  # Évolution de la température réelle
    if  this_t - last_z >= dt_z:
        measured_temperatures[i] = true_temperatures[i] + np.random.normal(0, np.sqrt(R[0,0]))  # Mesure bruitée
        last_z = this_t

# Estimation de la température avec EKF
estimated_temperatures = np.zeros((len(t)))
estimated_temperatures[0] = ekf.x.copy()
state_covariances = np.zeros((len(t), 1, 1))
state_covariances[0] = ekf.P.copy()
Fx = np.zeros((len(t), 1, 1))
Fu = np.zeros((len(t), 1, 1))
perturbation = np.random.normal(0, np.sqrt(ekf.cov_u[0,0]), len(t))
control_inputs += perturbation
for i in range(1, len(t)):
    this_t = t[i]
    ekf.predict(f, np.array([control_inputs[i]]))
    Fx[i] = ekf.Fx.copy()
    Fu[i] = ekf.Fu.copy()
    if not np.isnan(measured_temperatures[i]):
        ekf.update(h, np.array([measured_temperatures[i]]), R)
    estimated_temperatures[i] = ekf.x.copy()
    state_covariances[i] = ekf.P.copy()


# Création du DataFrame Pandas pour afficher les conditions d'initialisation et les résultats
#fill nan values when the measurement is not available    
df = pd.DataFrame({
    'Time': t,
    'XX': true_temperatures,
    'UU': control_inputs,
    'ZZ': measured_temperatures,
    'XX_hat': estimated_temperatures,
    'P00': state_covariances[:, 0, 0],
    'Fx': Fx[:, 0, 0],
    'Fu': Fu[:, 0, 0],
})
# Conditions d'initialisation du filtre et résultats
init_conditions = {
    'R': R,
    'ekf.C': ekf.c,
    'ekf.X': np.array(estimated_temperatures[0]).reshape(1),
    'ekf.P': P0,
    'ekf.Cov_U': ekf.cov_u,
}



# Génération du contenu à insérer entre les balises
data_content = ""
for col in df.columns:
    values = ", ".join(f"{v:.6f}" if pd.notna(v) else "NAN" for v in df[col])
    data_content += f"float {col}[] = {{ {values} }};\n"
data_content += "\n"

# Code de génération de la fonction init_filter()
init_filter_content = "void init_filter()\n{\n"

# Parcours des conditions d'initialisation pour créer le contenu
for key, value in init_conditions.items():
    if len(value.shape) == 1:
        for i, val in enumerate(value):
            init_filter_content += f"    {key}[{i}] = {val};\n"
    elif len(value.shape) == 2:
        for i, row in enumerate(value):
            for j, val in enumerate(row):
                init_filter_content += f"    {key}({i}, {j}) = {val};\n"
    else:
        print("Error: Unsupported shape for", key)
        
        

# Ajout de l'initialisation de `ekf.X` avec la première valeur de `X`
init_filter_content += "    ekf.setMeasurementFunction(h," + str(R.shape[0]) + ");\n"
init_filter_content += "    ekf.initted = true;\n"
init_filter_content += "}\n"

data_content += init_filter_content

# Lecture du fichier C++ et remplacement entre balises
input_file = 'test/A_unitary/ekf/A_unidimensional_linear_stable/test_run/main.cpp'
with open(input_file, 'r') as file:
    content = file.read()

# Remplace le texte entre les balises <DATA_START> et <DATA_END>
new_content = re.sub(r'// <DATA_START>.*?// <DATA_END>', f'// <DATA_START>\n{data_content}// <DATA_END>', content, flags=re.DOTALL)

# Écriture des modifications dans le fichier
with open(input_file, 'w') as file:
    file.write(new_content)
    
    # Affichage des résultats
plt.figure()
plt.plot(t, true_temperatures, label='True temperature')
plt.plot(t, measured_temperatures, 'o', label='Measured temperature')
plt.plot(t, estimated_temperatures, label='Estimated temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()