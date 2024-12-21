import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import de la classe Ukf
from ukf.ukf import Ukf  # Remplacez ukf par le nom de la classe que vous voulez importer

# Définition de la fonction de prédiction
def f(X, U, C):
    """
    Modèle de l'évolution de la position d'un objet sur un plan.
    X : [x,y,vt,theta] : position, vitesses tangentielle et direction de l'objet
    U : [at,w] : accélération tangentielle et vitesse angulaire
    C : [dt,b1x,b1y,b2x,b2y] : intervalle de temps et positions des balises
    """
    ret = X.copy()
    ct = np.cos(X[3])
    st = np.sin(X[3])
    ret[0] += ct*X[2]*C[0]
    ret[1] += st*X[2]*C[0]
    ret[2] += U[0]*C[0]
    ret[3] += U[1]*C[0]    
    return ret

# Fonction de mesure (voltage inversement proportionnel à le distance au carré entre l'objet et la balise 1)
def h1(X, C):
    """
    Fonction de mesure de la distance entre l'objet et la balise 1.
    """
    ret = 100/((X[0]-C[1])**2+(X[1]-C[2])**2)
    # print(ret)
    return np.array([ret])

# Fonction de mesure (voltage inversement proportionnel à le distance au carré entre l'objet et la balise 2)
def h2(X, C):
    """
    Fonction de mesure de la distance entre l'objet et la balise 2.
    """
    ret = 100/((X[0]-C[3])**2+(X[1]-C[4])**2)
    # print(ret)
    return np.array([ret])

dt_u = 0.01  # Intervalle de temps (en secondes) pour les commandes
dt_z1 = 0.2  # Intervalle de temps (en secondes) pour les mesures de la balise 1
dt_z2 = 0.21
# Initialisation de l'état initial (température initiale)
x_init = 20.0  # Température initiale en °C

# Bruit des commandes (puissance de chauffage)
acc_noise_density = 0.05  # Bruit de la commande des accéléromètres en m/s²*s^1/2
gyro_noise_density = 0.1  # Bruit de la commande des gyroscopes en rad/s*s^1/2
# Bruit du capteur de température
sensor_noise_density = 0.02  # Bruit de mesure de température en V*s^1/2


R1 = np.array([[sensor_noise_density**2/dt_z1]])  # Covariance du bruit de mesure
R2 = np.array([[sensor_noise_density**2/dt_z2]])  # Covariance du bruit de mesure

# Initialisation du filtre de Kalman étendu
ukf = Ukf(4)
P0 = np.eye(4) * 200*200*0.00001  # Grande incertitude initiale
ukf.P = P0.copy()  # Grande incertitude initiale
ukf.c = np.array([dt_u, 0, 30, 10, 5])  # Paramètres du système
ukf.cov_u = np.diag([acc_noise_density**2/dt_u, gyro_noise_density**2/dt_u])  # Covariance du bruit de commande

T = 10  # temps total de simulation
t = np.arange(0, T, dt_u)
# Simuler une trajectoire
true_omega = np.sin(t/T*np.pi*1.5)
true_tv = np.cos(t/T*np.pi/2)*10
true_x = np.zeros((len(t), 4))
true_x[0] = np.array([0, 0, true_tv[0], 0])
ukf.x = true_x[0].copy()
for i in range(1, len(t)):
    vx = np.cos(true_x[i-1,3])*true_tv[i]
    vy = np.sin(true_x[i-1,3])*true_tv[i]
    true_x[i,0] = true_x[i-1,0]+dt_u*vx
    true_x[i,1] = true_x[i-1,1]+dt_u*vy
    true_x[i,2] = true_tv[i]
    true_x[i,3] = true_x[i-1,3] + dt_u*true_omega[i]

# simulation des commandes et des mesures
u = np.zeros((len(t), 2))
z1 = np.zeros((len(t)))*np.nan
z2 = np.zeros((len(t)))*np.nan
last_z1 = 0
last_z2 = 0
for i in range(len(t)):
    if i > 0:
        u[i,0] = (true_tv[i]-true_tv[i-1])/dt_u + np.random.randn()*acc_noise_density/np.sqrt(dt_u)
        u[i,1] = true_omega[i] + np.random.randn()*gyro_noise_density/np.sqrt(dt_u)
    if t[i]-last_z1 >= dt_z1:
        last_z1 = t[i]
        z1[i] = h1(true_x[i], ukf.c) + np.random.randn()*sensor_noise_density/np.sqrt(dt_z1)
    if t[i]-last_z2 >= dt_z2 :
        last_z2 = t[i]
        z2[i] = h2(true_x[i], ukf.c) + np.random.randn()*sensor_noise_density/np.sqrt(dt_z2)


ukf.x = true_x[0].copy()  # Initialisation de l'état
ukf_X = np.zeros((len(t), 4))
ukf_P = np.zeros((len(t), 4, 4))
ukf_X[0] = ukf.x.copy()
ukf_P[0] = ukf.P.copy()
ukf_M0 = np.zeros((len(t), 1))
ukf_M1 = np.zeros((len(t), 1))
ukf_M0.fill(np.nan)
ukf_M1.fill(np.nan)

#simulation du filtre
for i in range(1, len(t)):
    ukf.predict(f,u[i])
    if not np.isnan(z1[i]):
        ukf_M0[i] = ukf.mahalanobis(h1, z1[i], R1)
        ukf.update(h1, z1[i], R1)
    if not np.isnan(z2[i]):
        ukf_M1[i] = ukf.mahalanobis(h2, z2[i], R2)
        ukf.update(h2, z2[i], R2)
        
    ukf_X[i] = ukf.x.copy()
    ukf_P[i] = ukf.P.copy()
    
# Création du DataFrame Pandas pour afficher les conditions d'initialisation et les résultats
#fill nan values when the measurement is not available    
df = pd.DataFrame({
    'Time': t,
    'XX0_hat' : ukf_X[:,0],
    'XX1_hat' : ukf_X[:,1],
    'XX2_hat' : ukf_X[:,2],
    'XX3_hat' : ukf_X[:,3],
    'UU0' : u[:,0],
    'UU1' : u[:,1],
    'ZZ0' : z1,
    'ZZ1' : z2,
    'MM0' : ukf_M0[:,0],
    'MM1' : ukf_M1[:,0],
    
})
# # Conditions d'initialisation du filtre et résultats
init_conditions = {
    'R0': R1,
    'R1': R2,
    'ukf.C': ukf.c,
    'ukf.X': ukf_X[0],
    'ukf.P': ukf_P[0],
    'ukf.Cov_U': ukf.cov_u,
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
        
        

# Ajout de l'initialisation de `ukf.X` avec la première valeur de `X`
init_filter_content += "    ukf.setMeasurementFunction(h0," + str(R1.shape[0]) + ",0);\n"
init_filter_content += "    ukf.setMeasurementFunction(h1," + str(R2.shape[0]) + ",1);\n"
init_filter_content += "}\n"

data_content += init_filter_content

input_file = 'test/A_unitary/ukf/B_multidimensional_nonlinear_unstable/test_run/main.cpp'
with open(input_file, 'r') as file:
    content = file.read()
 
# Remplace le texte entre les balises <DATA_START> et <DATA_END>
new_content = re.sub(r'// <DATA_START>.*?// <DATA_END>', f'// <DATA_START>\n{data_content}// <DATA_END>', content, flags=re.DOTALL)

# Écriture des modifications dans le fichier
with open(input_file, 'w') as file:
    file.write(new_content)
    
#plot trajectories
plt.figure()
plt.subplot(2,1,1)
plt.plot(true_x[:,0], true_x[:,1], label='True trajectory')
plt.plot(ukf_X[:,0], ukf_X[:,1], label='Estimated trajectory')
plt.scatter(ukf.c[1], ukf.c[2], label='Beacon 1', color='red')
plt.scatter(ukf.c[3], ukf.c[4], label='Beacon 2', color='blue')
plt.legend()
plt.title('Trajectories')
plt.xlabel('X')
plt.ylabel('Y')

# mahalanobis distances
plt.subplot(2,1,2)
plt.scatter(t, ukf_M0, label='Mahalanobis distance to beacon 1')
plt.scatter(t, ukf_M1, label='Mahalanobis distance to beacon 2')
plt.legend()
plt.title('Mahalanobis distances')
plt.xlabel('Time')
plt.ylabel('Mahalanobis distance')
plt.show()

