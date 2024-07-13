import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import csv
import glob
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import solve_ivp
from ipywidgets import interact, FloatSlider, Layout, interactive
from scipy.optimize import minimize


# Part 1

timeValues_List = list()
meanIntensity_List = list()
meanIntensity_CalibrationCurve_List = list()
proteinConcentration_List = list()
proteinConcentration_nM_List = list()
numberOfProteinMolecules_List = list()
rateOfChangeProteinMolecules_List = list()

# This function utilizes the images taken for the experiment of the K401 motor protein
def calculateMeanIntensity(paths):
    for i in range(0, len(paths)): 
        # Load the image as a matrix
        image_path = paths[i]
        image_matrix = io.imread(image_path)
        meanIntensity = image_matrix.mean()
        meanIntensity_List.append(meanIntensity)

# This function utilizes 9 sample images to analyze the relationship between "Mean Intensity" and "Protein Concentration"
def getConcentration(calibrationCurvePaths, mw_kda): # This function takes a list of image paths and molecular weight in kDa as arguments
    for i in range(0, len(calibrationCurvePaths)):
        # Load the image as a matrix
        image_path = calibrationCurvePaths[i]
        image_matrix = io.imread(image_path)
        meanIntensity = image_matrix.mean()
        meanIntensity_CalibrationCurve_List.append(meanIntensity) 

    df = pd.DataFrame(meanIntensity_CalibrationCurve_List).reset_index() # Create a data frame 
    df = df.rename(columns={"index":"Protein Concentration (microgram / milliliter)", 0:"Mean Intensity"})
    sampleConcentration_Values = [0, 2, 5, 10, 20, 40, 80, 160, 320]
    df["Protein Concentration (microgram / milliliter)"] = sampleConcentration_Values

    # Get the equation (linear) of best fit for the Protein Concentration
    x = df["Protein Concentration (microgram / milliliter)"]
    y = df["Mean Intensity"]

    slope, intercept = np.polyfit(x, y, 1) # Multiple return values is allowed in Python
    
    line_of_best_fit = slope * x + intercept
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df["Protein Concentration (microgram / milliliter)"], df["Mean Intensity"], marker='o', linestyle='none', label='Data points')
    plt.plot(x, line_of_best_fit, label=f'Line of Best Fit: y = {slope:.2f}x + {intercept:.2f}', color='red')
    plt.title('Mean Intensity vs Protein Concentration')
    plt.xlabel('Protein Concentration (microgram / milliliter)')
    plt.ylabel('Mean Intensity')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Transform the dependent variables
    for i in range(0, len(meanIntensity_List)):
        proteinConcentration = (meanIntensity_List[i] - intercept) / slope
        proteinConcentration_List.append(proteinConcentration)
        proteinConcentration_nM = ((proteinConcentration * 1e-3) / (mw_kda * 1e3)) * 1e9 # Convert to nM
        proteinConcentration_nM_List.append(proteinConcentration_nM)

def constructDataFrames(timeInterval):
    global meanIntensity_List
    global proteinConcentration_List
    global proteinConcentration_nM_List

    minimumIntensityValue = min(meanIntensity_List)
    adjustedMeanIntensity_List = [x - minimumIntensityValue for x in meanIntensity_List] # Subtract the minimum mean intensity value from ALL values
    meanIntensity_List = adjustedMeanIntensity_List

    minimumProteinConcentration = min(proteinConcentration_List)
    adjustedProteinConcentration_List = [x - minimumProteinConcentration for x in proteinConcentration_List]
    proteinConcentration_List = adjustedProteinConcentration_List

    minimumProteinConcentration_nM = min(proteinConcentration_nM_List)
    adjustedProteinConcentration_List_nM = [x - minimumProteinConcentration_nM for x in proteinConcentration_nM_List]
    proteinConcentration_nM_List = adjustedProteinConcentration_List_nM    

    df = pd.DataFrame(meanIntensity_List).reset_index() # Create a data frame 
    df = df.rename(columns={"index":"Time (min)", 0:"Mean Intensity"})
    df["Time (min)"] = df["Time (min)"] * timeInterval # Manipulate the "time" values

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (min)'], df['Mean Intensity'], marker='o')
    plt.title('Mean Intensity vs Time')
    plt.xlabel('Time (t)')
    plt.ylabel('Mean Intensity')
    plt.grid(True)
    plt.show()

    df2 = pd.DataFrame(proteinConcentration_List).reset_index()
    df2 = df2.rename(columns={"index":"Time (min)", 0:"Protein Concentration (nanogram / microliter)"})
    df2["Time (min)"] = df2["Time (min)"] * timeInterval # Manipulate the "time" values
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df2['Time (min)'], df2['Protein Concentration (nanogram / microliter)'], marker='o')
    plt.title('Protein Concentration vs Time')
    plt.xlabel('Time (t)')
    plt.ylabel('Protein Concentration (microgram / milliliter)')
    plt.grid(True)
    plt.show()

    df3 = pd.DataFrame(proteinConcentration_nM_List).reset_index()
    df3 = df3.rename(columns={"index":"Time (min)", 0:"Protein Concentration (nM)"})
    df3["Time (min)"] = df3["Time (min)"] * timeInterval # Manipulate the "time" values

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df3['Time (min)'], df3['Protein Concentration (nM)'], marker='o')
    plt.title('Protein Concentration (nM) vs Time')
    plt.xlabel('Time (t)')
    plt.ylabel('Protein Concentration (nM)')
    plt.grid(True)
    plt.show()

def getNumberOfProteinMolecules(dropletVolume, timeInterval, mw_kda):
    global numberOfProteinMolecules_List
    proteinMass_List = [i * dropletVolume for i in proteinConcentration_List] # List comprehension technique
    numberOfProteinMolecules_List = [(j * 6e14) / (mw_kda * 1e3) for j in proteinMass_List] # This expression was derived from several intermediate calculations

    df = pd.DataFrame(numberOfProteinMolecules_List).reset_index() # Create a data frame 
    df = df.rename(columns={"index":"Time (min)", 0:"Number of Protein Molecules"})
    df["Time (min)"] = df["Time (min)"] * timeInterval # Manipulate the "time" values

    # Plot the data

    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (min)'], df['Number of Protein Molecules'], marker='o')
    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (t)')
    plt.ylabel('Number of Protein Molecules')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (min)'], df['Number of Protein Molecules'], marker='o')
    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (t)')
    plt.ylabel('Number of Protein Molecules')
    # y axis log scale
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def getRateOfChangeProteinMolecules(timeInterval):
    global timeValues_List
    global rateOfChangeProteinMolecules_List
    
    p_vals = np.array(numberOfProteinMolecules_List) # Converts a Python list to a numpy array
    length = len(numberOfProteinMolecules_List)
    maxTimeValue = (length - 1) * timeInterval 
    t_vals = np.linspace(0, maxTimeValue, length) # Creates a numpy array
    timeValues_List = t_vals.tolist()

    # Estimate the numerical derivative of the number of protein molecules with respect to time
    dp_dt = np.gradient(p_vals, t_vals)
    rateOfChangeProteinMolecules_List = dp_dt.tolist()

    # apply gaussian filter with sigma 2
    dp_dt = gaussian_filter1d(dp_dt, sigma=2)
    
    # Plot the estimated derivative
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, dp_dt, label='Numerical derivative', marker='o', color="green")
    plt.xlabel('Time (t)')
    plt.ylabel('Rate of change of the number of protein molecules')
    plt.title('Rate of change of the number of protein molecules with respect to time')
    plt.legend()
    plt.grid(True)
    plt.show()

def saveData(fileName): # Saves the data to a CSV file
    # save dataframes to csv files
    dataFile = open(fileName, 'w', newline="")
    writerCSV = csv.writer(dataFile)
    headerRow = list()
    headerRow.append("Time")
    headerRow.append("Mean Intensity")
    headerRow.append("Protein Concentration (ng/µL)")
    headerRow.append("Protein Concentration (nM)")
    headerRow.append("Number of Protein Molecules (PM)")
    headerRow.append("Rate of Change of Number of PM")
    writerCSV.writerow(headerRow)
    for i in range(0, len(meanIntensity_List)):
        dataRow = list()
        dataRow.append(timeValues_List[i])
        dataRow.append(meanIntensity_List[i])
        dataRow.append(proteinConcentration_List[i])
        dataRow.append(proteinConcentration_nM_List[i])
        dataRow.append(numberOfProteinMolecules_List[i])
        dataRow.append(rateOfChangeProteinMolecules_List[i])
        writerCSV.writerow(dataRow)
    dataFile.close()

def runProteinAnalysis(paths, calibrationCurvePaths, timeInterval, dropletVolume, mw_kda, fileName):
    calculateMeanIntensity(paths) # 1st function to be called
    getConcentration(calibrationCurvePaths, mw_kda) # 2nd function to be called
    constructDataFrames(timeInterval) # Time interval is passed in minutes. # 3th function to be called
    getNumberOfProteinMolecules(dropletVolume, timeInterval, mw_kda) # For the experiment of the K401 protein, the droplet had a volume of 2 microliters. 4th function to be called
    getRateOfChangeProteinMolecules(timeInterval) # 5th function to be called
    saveData(fileName) # 6th function to be called


# Part 2

def showProteinDataTogether(files):
    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in files:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Mean Intensity'], marker='o') # Plot the "Mean Intensity" data only
    
    plt.title('Mean Intensity vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Mean Intensity')
    plt.grid(True)
    plt.savefig("Mean_Intensity.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in files:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Protein Concentration (microgram / milliliter)'], marker='o') # Plot the "Protein Concentration" data only

    plt.title('Protein Concentration vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Protein Concentration (microgram / milliliter)')
    plt.grid(True)
    plt.savefig("Protein_Concentration.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in files:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Number of Protein Molecules'], marker='o') # Plot the "Number of Protein Molecules" data only

    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Number of Protein Molecules')
    plt.grid(True)  
    plt.savefig("Number_of_Protein_Molecules.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in files:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Rate of Change of Protein Molecules'], marker='o') # Plot the "Rate of Change of Protein Molecules" data only

    plt.title('Rate of Change of Protein Molecules vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Rate of Change of Protein Molecules')
    plt.grid(True)
    plt.savefig("Rate_of_Change_of_Protein_Molecules.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop


# Part 3

# Calculate the [R_p D] complex using the equation in the paper's supplementary information. 
def calculate_RpD(R_p, D, K_TX): # Accept parameters to calculate the [R_p D] complex
    return 0.5 * (R_p + D + K_TX - np.sqrt((R_p + D + K_TX)**2 - 4 * R_p * D))

# Define the differential equation for protein concentration
def dPdt(T, P, Q, S, tau_0, tau_f, k3, k11): # Not only accept the variables T and P. Also, accept parameters that will be treated as constants in the ODE.
    if T > tau_0 + tau_f:
        return Q * (1 - np.exp(-(T - tau_0 - tau_f) / k3)) - (S * P) / (k11 + P)
    else:
        return 0 

# Create a function to plot the oscillators with given parameters
def plot_proteinConcentration(k_TL, k_TX, R_p, D, tau_m, N_p, K_TL, R, N_m, k_deg, X_p, K_p, tau_0, tau_f): #Parameters that will be able to be modified by the sliders
    
    RpD = calculate_RpD(R_p, D, k_TX) # For simplicity purposes, calculate [R_p D] complex using a function
    Q = (k_TL * k_TX * RpD * tau_m) / (N_p * (1 + K_TL / R) * N_m)  
    S = k_deg * X_p
    k3 = tau_m
    k11 = K_p

    # Time ranges from T = 0 to T = 5000 seconds
    T = np.linspace(0, 5000, len(proteinConcentration_nM_List)) # Same size as the experimental data of the protein concentration

    P_initial = 0  # At t = 0, the protein concentration P(0) = 0

    # All of the constants such as Q, S, tau_0... need to be passed as arguments into the solve_ivp()function
    p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11)) 
    
    plt.figure(figsize=(10, 6))  # Clear the figure before plotting
    plt.plot(T, proteinConcentration_nM_List, label='Experimental Curve', linestyle='--', color='orange')
    plt.plot(T, p.y[0], label='Theoretical Curve') # We need to access the "y" values from the object 'p' we created before
    plt.title('Protein Concentration vs. Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Protein Concentration (nM)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create interactive sliders for the parameters used to calculate the protein concentration

style = {'description_width': '300px'}  # Adjust the width as needed

def showModel():
    interact(plot_proteinConcentration, 
            k_TL=FloatSlider(value=10.0 , min=1.0, max=200.0, step=0.1, description='k_TL (amino acids/s)', layout=Layout(width='900px'), style=style),
            k_TX=FloatSlider(value=1.0 , min=1.0, max=200.0, step=0.1, description='k_TX (rNTP/s)', layout=Layout(width='900px'), style=style),
            R_p=FloatSlider(value=30.0, min=10.0, max=90.0, step=1, description='RNA polymerase concentration (nM)', layout=Layout(width='900px'), style=style), # We know for sure the value of RNA polymerase
            D=FloatSlider(value=30.0, min=20.0, max=200.0, step=1, description='DNA concentration (nM)', layout=Layout(width='900px'), style=style), ## We know for sure the value of DNA concentration
            tau_m=FloatSlider(value=655.0 , min=300.0, max=1200.0, step=1.0, description='mRNA lifetime (seconds)', layout=Layout(width='900px'), style=style),
            N_p=FloatSlider(value=401, min=100, max=4500, step=100, description='protein length (amino acids)', layout=Layout(width='900px'), style=style), ## We know for sure the number of aminoacids
            K_TL = FloatSlider(value=5, min=0.1, max=10, step=0.1, description='Michaelis-Menten constant for translation (nM)', layout=Layout(width='900px'), style=style),
            R=FloatSlider(value=190.0, min=50.0, max=350.0, step=10, description='ribosome concentration (nM)', layout=Layout(width='900px'), style=style), # We know for sure the ribosome concentration (we can estimate it)
            N_m=FloatSlider(value=3000, min=1500, max=4500, step=100, description='mRNA Length (Nucleotides)', layout=Layout(width='900px'), style=style), ## We know for sure the number of nucleotides (this is based on the DNA design)
            k_deg=FloatSlider(value=0.02, min=0.01, max=5.0, step=0.01, description='protein degradation rate constant (1/s)', layout=Layout(width='900px'), style=style), 
            X_p=FloatSlider(value=9.0, min=1.0, max=20.0, step=0.1, description='protease concentration (nM)', layout=Layout(width='900px'), style=style), # We also know for sure this parameter
            K_p=FloatSlider(value=25.92, min=0.1, max=30.0, step=0.1, description='Michaelis-Menten constant for degradation (nM)', layout=Layout(width='900px'), style=style),
            tau_0=FloatSlider(value=0, min=0.0, max=100.0, step=0.1, description='transcription delay (seconds)', layout=Layout(width='900px'), style=style), # We know for sure the value of this parameter. Its value is 0
            tau_f=FloatSlider(value=300, min=0.0, max=2000.0, step=1.0, description='protein folding delay (seconds)', layout=Layout(width='900px'), style=style)) ## The value of this parameter is 20 



# Part 4

length = len(proteinConcentration_nM_List)
proteinConcentration_nM_List_NP = np.array(proteinConcentration_nM_List) # Important line: convert to NumPy array

subset_indices = np.linspace(0, length - 1, length, dtype=int) # Define indices: 0, 10, 20, 30, ..
subset_ProteinConcentration_nM_List = proteinConcentration_nM_List_NP[subset_indices] # Access the values locates in these indices

# Calculate the [R_p D] complex using the equation in the paper's supplementary information. 
def calculate_RpD(R_p, D, K_TX): # Accept parameters to calculate the [R_p D] complex
    discriminant = (R_p + D + K_TX)**2 - 4 * R_p * D
    if discriminant < 0:
        return 1e-6 # Return a small positive value if the discriminant is negative
    else:
        return 0.5 * (R_p + D + K_TX - np.sqrt(discriminant))

# Define the differential equation for protein concentration
def dPdt(T, P, Q, S, tau_0, tau_f, k3, k11): # Not only accept the variables T and P. Also, accept parameters that will be treated as constants in the ODE.
    if T > tau_0 + tau_f:
        return Q * (1 - np.exp(-(T - tau_0 - tau_f) / k3)) - (S * P) / (k11 + P)
    else:
        return 0 

def solve_ODE(params):

    k_TL, k_TX, R_p, tau_m, K_TL, R, k_deg, X_p, K_p, tau_0, tau_f = params # In Python, this is a way to define variables, given a list of values. Only 1 line is required

    # These are the parameters for which we know the values of
    N_p = 401 # We know the number of aminoacids, because we are analyzing the "k401" kinesin motor protein
    N_m = 3000 # We also know the number of nucleotides 
    D = 30 # We also know the DNA concentration

    RpD = calculate_RpD(R_p, D, k_TX) # For simplicity purposes, we calculated [R_p D] complex using a function
    Q = (k_TL * k_TX * RpD * tau_m) / (N_p * (1 + K_TL / R) * N_m)  
    S = k_deg * X_p
    k3 = tau_m
    k11 = K_p

    # Time ranges from T = 0 to T = 5000 seconds
    T = np.linspace(0, 5000, len(subset_ProteinConcentration_nM_List)) # Same size as the experimental data of the protein concentration

    P_initial = 0  # At t = 0, the protein concentration P(0) = 0

    # All of the constants such as Q, S, k3... need to be passed as arguments into the solve_ivp()function
    p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method ="LSODA") # Use the "LSODA" method to solve the ODE
    return p.y[0]

# The objective function uses the method of "Sum of Squared Errors (SSE)"
def objective_function(params):
    pModel = solve_ODE(params)
    return np.sum((subset_ProteinConcentration_nM_List-pModel)**2)

# 1st element: k_TL; 2nd element: k_TX; 3th element: R_p; 4th element: tau_m; 5th element: K_TL; 6th element: R; 7th element: k_deg; 8th element: X_p; 9th element: K_p; 10th element: tau_0, 11th element: tau_f
initial_guesses = [10, 1, 30, 720, 5, 190, 0.01, 9, 4, 0, 300] # The guesses are the same for ALL the motor proteins 

result = minimize(objective_function, initial_guesses, method='TNC') # "L-BFGS-B" is a popular method to minimize an objective function. "TNC" is another method to minimize an objective function
optimizedParameters = result.x #  Since "result" is an object, we need to access a certain attribute of "result" to extract the optimized parameters

def showOptimizedModel():

    # Print the optimized parameters
    print("Optimized parameters:")
    print("k_TL:", optimizedParameters[0])
    print("k_TX:", optimizedParameters[1])
    print("R_p:", optimizedParameters[2])
    print("tau_m:", optimizedParameters[3])
    print("K_TL:", optimizedParameters[4])
    print("R:", optimizedParameters[5])
    print("k_deg:", optimizedParameters[6])
    print("X_p:", optimizedParameters[7])
    print("K_p:", optimizedParameters[8])
    print("tau_0:", optimizedParameters[9])
    print("tau_f:", optimizedParameters[10])

    optimizedModel = solve_ODE(optimizedParameters)
    T = np.linspace(0, 5000, len(subset_ProteinConcentration_nM_List)) # Same size as the experimental data of the protein concentration
    plt.figure(figsize=(10, 6))  # Clear the figure before plotting
    plt.plot(T, subset_ProteinConcentration_nM_List, label='Experimental Curve', linestyle='--', color='orange')
    plt.plot(T, optimizedModel, label='Theoretical Curve') # We need to access the "y" values from the object that stores the solution to the ODE
    plt.title('Protein Concentration vs. Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Protein Concentration (nM)')
    plt.legend()
    plt.grid(True)
    plt.show()

showOptimizedModel()