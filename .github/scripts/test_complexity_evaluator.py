import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools
from scipy.integrate import dblquad
import json

filter_type = "ekf"
esp_results = {"key" : [], "data" : np.array([[]])}

"""communication functions"""
def wait_for_esp32(ser):
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line == filter_type+"-go":
                break

def listen_header(ser):
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        esp_results["key"] = line.split("\t")
        esp_results["data"] = np.array([[]], dtype=int).reshape(0, len(esp_results["key"]))
        
def listen_data(ser):
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line == "end":
            return False
        data = line.split("\t")
        data = np.array([data], dtype=int)
        esp_results["data"] = np.append(esp_results["data"], data, axis=0)
    return True


"""analysis functions"""
def plot(data_i_j_k, df, lin1, lin2, lin3, poly, poly2, show=True):
    
    # 2D plot of the prediction time in function of X_DIM and U_DIM
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # First plot: prediction time vs X_DIM and U_DIM
    ax = axes[0]
    ax.set_xlabel('X_DIM')
    ax.set_ylabel('U_DIM')
    ax.set_title('Prediction Time')


    # Compute the meshgrid for contour plot
    X1 = np.linspace(0, np.max(df['X_DIM']) * 1.2, 100)
    X2 = np.linspace(0, np.max(df['U_DIM']) * 1.2, 100)
    X1, X2 = np.meshgrid(X1, X2)

    # Predict time using the model
    Y = lin1.predict(poly.fit_transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
    
    # Créer un objet Normalize pour une échelle de couleur commune à la prédictioon
    min_val, max_val = np.min(Y), np.max(Y)
    norm = colors.Normalize(vmin=min_val/1000, vmax=max_val/1000)

    # plot prediction as a transparent image
    imgshow = ax.imshow(Y / 1000, cmap='viridis', extent=(0, np.max(df['X_DIM']) * 1.2, 0, np.max(df['U_DIM']) * 1.2), origin='lower', aspect='auto', alpha=0.75, norm=norm)
    plt.colorbar(imgshow, ax=ax, label="Time (ms)")
    # Add contour plot
    CS = ax.contour(X1, X2, Y / 1000, levels=np.logspace(0, 3, 10), cmap='viridis', linestyles='solid')
    scatter = ax.scatter(df['X_DIM'], df['U_DIM'], c=df['prediction']/1000, cmap='viridis', label='samples', alpha=1, norm=norm) # to match color with the image, we need to set min and max values of cmap

    # Add text labels on contours
    for i, collection in enumerate(CS.collections):
        for path in collection.get_paths():
            # Calculate the mid-point of the contour path
            x, y = path.vertices[:, 0], path.vertices[:, 1]
            # Find the middle point to place the text
            mid_x, mid_y = np.mean(x), np.mean(y)
            # Add text
            ax.text(mid_x, mid_y, f'{CS.levels[i]:.2f}', color='white', fontsize=12, ha='center', va='center', 
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))

    # Second plot: update time vs X_DIM and Z_DIM
    ax = axes[1]
    ax.set_xlabel('X_DIM')
    ax.set_ylabel('Z_DIM')
    ax.set_title('Update Time')


    # Compute the meshgrid for contour plot
    X1 = np.linspace(0, np.max(df['X_DIM']) * 1.2, 100)
    X2 = np.linspace(0, np.max(df['Z_DIM']) * 1.2, 100)
    X1, X2 = np.meshgrid(X1, X2)

    # Predict time using the model
    Y = lin2.predict(poly.fit_transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

    # Créer un objet Normalize pour une échelle de couleur commune à la prédictioon
    min_val, max_val = np.min(Y), np.max(Y)
    norm = colors.Normalize(vmin=min_val/1000, vmax=max_val/1000)
    
    # plot prediction as a transparent image
    imgshow = ax.imshow(Y / 1000, cmap='viridis', extent=(0, np.max(df['X_DIM']) * 1.2, 0, np.max(df['Z_DIM']) * 1.2), origin='lower', aspect='auto', alpha=0.75, norm=norm)
    plt.colorbar(imgshow, ax=ax, label="Time (ms)")

    # Add contour plot
    CS = ax.contour(X1, X2, Y / 1000, levels=np.logspace(0, 3, 10), cmap='viridis', linestyles='solid')
    scatter = ax.scatter(df['X_DIM'], df['Z_DIM'], c=df['update']/1000, cmap='viridis', label='samples', alpha=1, norm=norm) # to match color with the image, we need to set min and max values of cmap

    # Add text labels on contours
    for i, collection in enumerate(CS.collections):
        for path in collection.get_paths():
            # Calculate the mid-point of the contour path
            x, y = path.vertices[:, 0], path.vertices[:, 1]
            # Find the middle point to place the text
            mid_x, mid_y = np.mean(x), np.mean(y)
            # Add text
            ax.text(mid_x, mid_y, f'{CS.levels[i]:.2f}', color='white', fontsize=12, ha='center', va='center', 
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))

    # Show the plots
    plt.tight_layout()
    if show:
        plt.show()
    
    #export this plot in lib/ESP32AlgebraFilters/docs/ekf_complexity_analysis
    fig.savefig('lib/ESP32AlgebraFilters/docs/'+filter_type+'_complexity_analysis.png', dpi=200)


def analyse(results):
    df = pd.DataFrame(results["data"], columns=results["key"])
    data_i_j_k = df.groupby(['X_DIM', 'U_DIM', 'Z_DIM']).agg(list)
    
    data_grouped = df.groupby(['X_DIM', 'U_DIM', 'Z_DIM'])
    data_i_j_k_mean_std = data_grouped.agg(["mean", "std", "median", "max", "min"])
    df = pd.DataFrame(data_i_j_k_mean_std.index.tolist(), columns=['X_DIM', 'U_DIM', 'Z_DIM'])
    df['prediction'] = data_i_j_k_mean_std['prediction']['mean'].tolist()
    df['update'] = data_i_j_k_mean_std['update']['mean'].tolist()
    df['alloc'] = data_i_j_k_mean_std['alloc']['max'].tolist()
    X1,X2,X3 = df[['X_DIM', 'U_DIM']], df[['X_DIM', 'Z_DIM']], df[['X_DIM', 'U_DIM', 'Z_DIM']]
    y1,y2,y3 = df[['prediction']], df[['update']], df[['alloc']]
    poly = PolynomialFeatures(degree=3)
    poly2 = PolynomialFeatures(degree=2)
    X_poly1, X_poly2, X_poly3 = poly.fit_transform(X1), poly.fit_transform(X2), poly2.fit_transform(X3)
    poly.fit(X_poly1, y1)
    poly.fit(X_poly2, y2)
    poly2.fit(X_poly3, y3)
    lin1, lin2, lin3 = LinearRegression(), LinearRegression(), LinearRegression()
    lin1.fit(X_poly1, y1)
    lin2.fit(X_poly2, y2)
    lin3.fit(X_poly3, y3)
    
    
    return data_i_j_k, df, lin1, lin2, lin3, poly, poly2, X_poly1, X_poly2, X_poly3, y1, y2, y3
    

"""report functions"""
superscript_map = ["^0", "^1", "^2", "^3", "^4", "^5", "^6", "^7", "^8", "^9"]
def generate_polynomial_terms(variables, max_degree):
    terms = []
    for degree in range(0, max_degree + 1):
        for combination in itertools.combinations_with_replacement(variables, degree):
            # terms.append('*'.join(combination))
            terms.append(combination)
    return terms

def coefficients2str(score, cst, coefficients,variables,max_degree, digits=2):
    polynomial_terms = generate_polynomial_terms(variables, max_degree)
    dic = {}
    dic["fitting score"] = score
    for i in range(len(coefficients)):
        powers = np.zeros(len(variables))
        for term in polynomial_terms[i]:
            powers[variables.index(term)] += 1
        var = ''
        for j in range(len(variables)):
            if powers[j] > 0:
                var += f'{variables[j]}'
                if int(powers[j]) != 1:
                    var += superscript_map[int(powers[j])]
        if var == '':
            dic["cst"] = cst
        else:
            dic[var] = coefficients[i]
            
    return dic

ekf_complexity_analysis = {}
def read_ekf_complexity_analysis():
    #read current release in lib/ESP32AlgebraFilters/library.json
    with open('lib/ESP32AlgebraFilters/library.json', 'r') as file:
        library = json.load(file)
    global ekf_complexity_analysis
    with open('lib/ESP32AlgebraFilters/docs/ekf_complexity_analysis.json', 'r') as file:
        ekf_complexity_analysis = json.load(file)
        
    #update the release with the new complexity analysis
    ekf_complexity_analysis["current release"] = {}
    ekf_complexity_analysis["current release"]["version"] = library["version"]


# Fonction du polynôme en fonction des variables x et y
coefficients = None
intercept = None
def P(x, y):
    # Utiliser les termes du polynôme dans une fonction
    terms = poly.fit_transform(np.array([[x, y]]))  # créer les termes polynomiaux pour (x, y)
    return np.dot(terms, coefficients) + intercept


def compute_ekf_complexity_analysis(X_poly1, X_poly2, X_poly3, y1, y2, y3, lin1, lin2, lin3):

    score1 = lin1.score(X_poly1, y1)
    score2 = lin2.score(X_poly2, y2)
    score3 = lin3.score(X_poly3, y3)
    ekf_complexity_analysis["current release"]["prediction"] = coefficients2str(score1,lin1.intercept_[0],lin1.coef_[0], ['x','u'], 3,3)
    ekf_complexity_analysis["current release"]["update"] = coefficients2str(score2,lin2.intercept_[0],lin2.coef_[0], ['x','z'], 3,3)
    ekf_complexity_analysis["current release"]["alloc"] = coefficients2str(score3,lin3.intercept_[0],lin3.coef_[0], ['x','u','z'], 3,3)
    
    global coefficients
    global intercept
    coefficients = lin1.coef_[0]
    intercept = lin1.intercept_[0]
    integral, _ = dblquad(P, ekf_complexity_analysis["general uses cases"]["x min"], ekf_complexity_analysis["general uses cases"]["x max"], lambda x: ekf_complexity_analysis["general uses cases"]["u min"], lambda x: ekf_complexity_analysis["general uses cases"]["u max"])
    ekf_complexity_analysis["current release"]["prediction"]["average in general uses cases"] =  integral / ((ekf_complexity_analysis["general uses cases"]["x max"] - ekf_complexity_analysis["general uses cases"]["x min"]) * (ekf_complexity_analysis["general uses cases"]["u max"] - ekf_complexity_analysis["general uses cases"]["u min"]))
    coefficients = lin2.coef_[0]
    intercept = lin2.intercept_[0]
    integral, _ = dblquad(P, ekf_complexity_analysis["general uses cases"]["x min"], ekf_complexity_analysis["general uses cases"]["x max"], lambda x: ekf_complexity_analysis["general uses cases"]["z min"], lambda x: ekf_complexity_analysis["general uses cases"]["z max"])
    ekf_complexity_analysis["current release"]["update"]["average in general uses cases"] =  integral / ((ekf_complexity_analysis["general uses cases"]["x max"] - ekf_complexity_analysis["general uses cases"]["x min"]) * (ekf_complexity_analysis["general uses cases"]["z max"] - ekf_complexity_analysis["general uses cases"]["z min"]))
    coefficients = lin3.coef_[0]
    intercept = lin3.intercept_[0]
    integral, _ = dblquad(P, ekf_complexity_analysis["general uses cases"]["x min"], ekf_complexity_analysis["general uses cases"]["x max"], lambda x: ekf_complexity_analysis["general uses cases"]["z min"], lambda x: ekf_complexity_analysis["general uses cases"]["z max"])
    ekf_complexity_analysis["current release"]["alloc"]["average in general uses cases"] =  integral / ((ekf_complexity_analysis["general uses cases"]["x max"] - ekf_complexity_analysis["general uses cases"]["x min"]) * (ekf_complexity_analysis["general uses cases"]["z max"] - ekf_complexity_analysis["general uses cases"]["z min"]))
    
def format_percentage(value, threshold):
    percentage = value * 100  # Conversion en pourcentage
    # Vert si <= seuil, et rouge si > seuil
    color_code = ""
    if value < -threshold:
        color_code = "\033[92m"
    elif value > threshold:
        color_code = "\033[91m"
        
    reset_code = "\033[0m"
    return f"{color_code}{percentage:.3f}%{reset_code}"

def compare_efficiency(max_regression_relative_error=0.015):
    #compare the ["average in general uses cases"] of the current release with the best release
    best = ekf_complexity_analysis["best release"]
    current = ekf_complexity_analysis["current release"]
    better = False
    regret = False
    relative_prediction_evolution = (current["prediction"]["average in general uses cases"] - best["prediction"]["average in general uses cases"]) / best["prediction"]["average in general uses cases"]
    relative_update_evolution = (current["update"]["average in general uses cases"] - best["update"]["average in general uses cases"]) / best["update"]["average in general uses cases"]
    relative_alloc_evolution = (current["alloc"]["average in general uses cases"] - best["alloc"]["average in general uses cases"]) / best["alloc"]["average in general uses cases"]
    if relative_prediction_evolution < -max_regression_relative_error:
        better = True
    elif relative_prediction_evolution > max_regression_relative_error:
        regret = True
    if relative_update_evolution < -max_regression_relative_error:
        better = True
    elif relative_update_evolution > max_regression_relative_error:
        regret = True
    if relative_alloc_evolution < -max_regression_relative_error:
        better = True
    elif relative_alloc_evolution > max_regression_relative_error:
        regret = True
        
    print("Prediction time evolution:", format_percentage(relative_prediction_evolution, max_regression_relative_error))
    print("Update time evolution:", format_percentage(relative_update_evolution, max_regression_relative_error))
    print("Memory allocation evolution:", format_percentage(relative_alloc_evolution, max_regression_relative_error))
    
    if better:
        print("The current release is better than the best release")
        ekf_complexity_analysis["best release"] = current
    elif regret:
        print("The current release is worse than the best release")
        
    return better, regret
    
def write_ekf_complexity_analysis():
    with open('lib/ESP32AlgebraFilters/docs/'+filter_type+'_complexity_analysis.json', 'w') as file:
        json.dump(ekf_complexity_analysis, file, indent=4)
    

if __name__ == "__main__":
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    ORANGE = "\033[93m"
    RESET = "\033[0m"

    print(f"{ORANGE}Make sure you flashed the ESP32 with the 'B_efficiency/{filter_type}/test_complexity/complexity.cpp' sketch, then reset the ESP32 if the test does not start by itself.{RESET}")
    print("You can upload the sketch by running the following command in the terminal:")
    print(f"{BLUE}pio test -e esp32dev -f B_efficiency/{filter_type}/test_complexity/complexity.cpp{RESET}")
    ports = serial.tools.list_ports.comports()
    ports = [port for port in ports if "Bluetooth" not in port.description]
    if len(ports) == 1:
        print("automatically using the only available serial port")
        ser = serial.Serial(ports[0].device, 115200)
    else:
        print("please select the serial port to use:")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device}")
        port_index = int(input("port index: "))
        ser = serial.Serial(ports[port_index].device, 115200)
    
    wait_for_esp32(ser)
    print(f"{GREEN}Connexion initialized{RESET}")
    listen_header(ser)
    while listen_data(ser):
        pass
    ser.close()
    data_i_j_k, df, lin1, lin2, lin3, poly, poly2, X_poly1, X_poly2, X_poly3, y1, y2, y3 = analyse(esp_results)
    plot(data_i_j_k, df, lin1, lin2, lin3, poly, poly2, show=False)
    read_ekf_complexity_analysis()
    compute_ekf_complexity_analysis(X_poly1, X_poly2, X_poly3, y1, y2, y3, lin1, lin2, lin3)
    compare_efficiency()
    write_ekf_complexity_analysis()