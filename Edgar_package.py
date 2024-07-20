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
import random
import seaborn as sns


# Part 1

#Initialize global variables
timeValues_List = list()
meanIntensity_List = list()
proteinConcentration_List = list()
proteinConcentration_nM_List = list()
numberOfProteinMolecules_List = list()
rateOfChangeProteinMolecules_List = list()
optimizedParameters = list()

# This function utilizes the images taken for an experiment of a kinesin motor protein
def calculateMeanIntensity(paths):
    for i in range(0, len(paths)): 
        # Load the image as a matrix
        image_path = paths[i]
        image_matrix = io.imread(image_path)
        meanIntensity = image_matrix.mean()
        meanIntensity_List.append(meanIntensity)

# This function utilizes 9 sample images to analyze the relationship between "Mean Intensity" and "Protein Concentration"
def getConcentration(calibrationCurvePaths, mw_kda): # This function takes a list of image paths and molecular weight in kDa as arguments
    
    meanIntensity_CalibrationCurve_List = list()
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
    plt.xlabel('Time (min)')
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
    plt.xlabel('Time (min)')
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
    plt.xlabel('Time (min)')
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
    plt.xlabel('Time (min)')
    plt.ylabel('Number of Protein Molecules')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (min)'], df['Number of Protein Molecules'], marker='o')
    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (min)')
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
    plt.xlabel('Time (min)')
    plt.ylabel('Rate of change of the number of protein molecules')
    plt.title('Rate of change of the number of protein molecules with respect to time')
    plt.legend()
    plt.grid(True)
    plt.show()

def saveExperimentalData(experiment_fileName): # Saves the data to a CSV file
    # save dataframes to csv files
    dataFile = open(experiment_fileName, 'w', newline="")
    writerCSV = csv.writer(dataFile)
    headerRow = list()
    headerRow.append("Time (min)")
    headerRow.append("Mean Intensity (A.U.)")
    headerRow.append("Protein Concentration (ng/µL)")
    headerRow.append("Protein Concentration (nM)")
    headerRow.append("Number of Protein Molecules")
    headerRow.append("Rate of Change of Number of Protein Molecules (PM/min)")
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


# Part 2

# Calculate the [R_p D] complex using the equation in the paper's supplementary information. 
def calculate_RpD2(R_p, D, k_TX): # Accept parameters to calculate the [R_p D] complex
    discriminant = (R_p + D + k_TX)**2 - 4 * R_p * D
    if discriminant < 0:
        return 1e-6 # Return a small positive value if the discriminant is negative
    else:
        return 0.5 * (R_p + D + k_TX - np.sqrt(discriminant))

# Define the differential equation for protein concentration
def dPdt(T, P, Q, S, tau_0, tau_f, k3, k11): # Not only accept the variables T and P. Also, accept parameters that will be treated as constants in the ODE.
    if T > tau_0 + tau_f:
        return Q * (1 - np.exp(-(T - tau_0 - tau_f) / k3)) - (S * P) / (k11 + P)
    else:
        return 0 

def solve_ODE(params, N_p, N_m, D):

    k_TL, k_TX, R_p, tau_m, K_TL, R, k_deg, X_p, K_p, tau_0, tau_f = params # In Python, this is a way to define variables, given a list of values. Only 1 line is required

    RpD = calculate_RpD2(R_p, D, k_TX) # For simplicity purposes, we calculated [R_p D] complex using a function
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
def objective_function(params, N_p, N_m, D):
    pModel = solve_ODE(params, N_p, N_m, D)
    return np.sum((subset_ProteinConcentration_nM_List-pModel)**2)

def optimize_parameters(initial_guesses, N_p, N_m, D):
    global optimizedParameters
    # The lower bounds of tau_m and K_p are a very small positive number (not 0), to avoid having issues of dividing by 0
    bounds = [(0, 100), (0, 100), (0, 500), (1e-6, 1e4), (0, 100), (1e-6, 1e5), (0, 100), (0, 500), (1e-6, 100), (0, 1e6), (0, 2000)]
    result = minimize(objective_function, initial_guesses, args=(N_p, N_m, D), method='TNC', bounds=bounds)  # "L-BFGS-B" is a popular method to minimize an objective function. "TNC" is another method to minimize an objective function
    optimizedParameters = result.x  #  Since "result" is an object, we need to access a certain attribute of "result" to extract the optimized parameters

def optimize_parameters_many_times(initial_guesses, N_p, N_m, D):
    bounds = [(0, 100), (0, 100), (0, 500), (1e-6, 1e4), (0, 100), (1e-6, 1e5), (0, 100), (0, 500), (1e-6, 100), (0, 1e6), (0, 2000)]
    result = minimize(objective_function, initial_guesses, args=(N_p, N_m, D), method='TNC', bounds=bounds)  # "L-BFGS-B" is a popular method to minimize an objective function. "TNC" is another method to minimize an objective function
    currentOptimizedParameters = result.x  #  Since "result" is an object, we need to access a certain attribute of "result" to extract the optimized parameters
    return currentOptimizedParameters

# Here, the algorithm to optimize parameters is only used once. This function shows a demo graph
def showOptimizedModel(N_p, N_m, D): 

    global optimizedParameters
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

    optimizedModel = solve_ODE(optimizedParameters, N_p, N_m, D)
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

def runParameterOptimization(N_p, N_m, D, theory_file_name):

    global optimizedParameters
    parameter_names = ["k_TL", "k_TX", "R_p", "tau_m", "K_TL", "R", "k_deg", "X_p", "K_p", "tau_0", "tau_f"]

    """Add parameters to a DataFrame. 
    The first row will be the values of the parameters used for the demo. graph """
    parameter_values = optimizedParameters  
    parameters_df = pd.DataFrame([parameter_values], columns=parameter_names)

    for param, value in zip(["N_p", "N_m", "D"], [N_p, N_m, D]):
        parameters_df[param] = value 

    knownParameters = [N_p, N_m, D]
    parametersRangeMatrix = [] # This will become a matrix
    for i in range(100):
        #Create a list of 11 random float values. All values are within the lower and upper bounds previously set
        random_initial_guesses = [round(random.uniform(low, high), 2) for low, high in [(0, 100), (0, 100), (0, 500), (1e-6, 1e4), (0, 100), (1e-6, 1e5), (0, 100), (0, 500), (1e-6, 100), (0, 1e6), (0, 2000)]]
        currentOptimizedParameters = optimize_parameters_many_times(random_initial_guesses, N_p, N_m, D) # Calculate new values for the optimal parameters
        currentOptimizedParameters_List = currentOptimizedParameters.tolist() # Convert a "np.ndarray" object to a Python list
        newRow = currentOptimizedParameters_List + knownParameters # Combine 2 Python lists into a single list
        parameters_df.loc[len(parameters_df)] = newRow # Add a new row of optimized parameters to the DataFrame
        if (i == 50): # Plot a random iteration
            optimizedParameters = np.array([])
            optimizedParameters = np.array(currentOptimizedParameters_List)
            showOptimizedModel(N_p, N_m, D)
    
    for i in parameter_names:
        minValue = parameters_df[i].min() # Extracts the minimum value of each ENTIRE column
        maxValue = parameters_df[i].max() # Extracts the maximum value of each ENTIRE column
        parameterRange = [minValue, maxValue]
        parametersRangeMatrix.append(parameterRange)
    print("Range of parameters:")
    print()
    print(parametersRangeMatrix)

    saveTheoreticalData(parameters_df, theory_file_name) # Save the DataFrame to a .csv file
    
def saveTheoreticalData(theory_df, theory_file_name):
    theory_df.to_csv(theory_file_name, index=False)


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

def showModel(optimizedParameters, N_p, N_m, D):

    k_TL, k_TX, R_p, tau_m, K_TL, R , k_deg, X_p, K_p, tau_0, tau_f = optimizedParameters

    # Create interactive sliders for the parameters used to calculate the protein concentration
    style = {'description_width': '300px'}  # Adjust the width as needed
   
    interact(plot_proteinConcentration, 
            k_TL=FloatSlider(value=k_TL , min=0.0, max=100, step=0.1, description='k_TL (amino acids/s)', layout=Layout(width='900px'), style=style),
            k_TX=FloatSlider(value=k_TX , min=0.0, max=100, step=0.1, description='k_TX (rNTP/s)', layout=Layout(width='900px'), style=style),
            R_p=FloatSlider(value=R_p, min=0.0, max=500, step=0.1, description='RNA polymerase concentration (nM)', layout=Layout(width='900px'), style=style), 
            D=FloatSlider(value=D, min=0.0, max=1000, step=1, description='DNA concentration (nM)', layout=Layout(width='900px'), style=style), ## We know for sure the value of DNA concentration
            tau_m=FloatSlider(value=tau_m , min=0.0, max=1e4, step=0.1, description='mRNA lifetime (seconds)', layout=Layout(width='900px'), style=style),
            N_p=FloatSlider(value=N_p, min=0.0, max=10000, step=1, description='protein length (amino acids)', layout=Layout(width='900px'), style=style), ## We know for sure the number of aminoacids
            K_TL = FloatSlider(value=K_TL, min=0.0, max=100, step=0.1, description='Michaelis-Menten constant for translation (nM)', layout=Layout(width='900px'), style=style),
            R=FloatSlider(value=R, min=0.0, max=1e5, step=0.1, description='ribosome concentration (nM)', layout=Layout(width='900px'), style=style), 
            N_m=FloatSlider(value=N_m, min=0.0, max=10000, step=1, description='mRNA Length (Nucleotides)', layout=Layout(width='900px'), style=style), ## We know for sure the number of nucleotides (this is based on the DNA design)
            k_deg=FloatSlider(value=k_deg, min=0.0, max=1e5, step=0.1, description='protein degradation rate constant (1/s)', layout=Layout(width='900px'), style=style), 
            X_p=FloatSlider(value=X_p, min=0.0, max=500, step=0.1, description='protease concentration (nM)', layout=Layout(width='900px'), style=style), 
            K_p=FloatSlider(value=K_p, min=0.0, max=100, step=0.1, description='Michaelis-Menten constant for degradation (nM)', layout=Layout(width='900px'), style=style),
            tau_0=FloatSlider(value=tau_0, min=0.0, max=1e6, step=0.1, description='transcription delay (seconds)', layout=Layout(width='900px'), style=style), 
            tau_f=FloatSlider(value=tau_f, min=0.0, max=2000, step=0.1, description='protein folding delay (seconds)', layout=Layout(width='900px'), style=style))



# *** Wrapper function 
def runIndividualAnalysis(paths, calibration_curve_paths, time_interval, droplet_volume, mw_kda, N_p, N_m, D, initial_guesses, experiment_file_name, theory_file_name):
    global subset_ProteinConcentration_nM_List, optimizedParameters, timeValues_List, meanIntensity_List, proteinConcentration_List, proteinConcentration_nM_List, numberOfProteinMolecules_List, rateOfChangeProteinMolecules_List

    # Part 1
    calculateMeanIntensity(paths)
    getConcentration(calibration_curve_paths, mw_kda)
    constructDataFrames(time_interval)
    getNumberOfProteinMolecules(droplet_volume, time_interval, mw_kda)
    getRateOfChangeProteinMolecules(time_interval)
    saveExperimentalData(experiment_file_name)

    # Part 2
    length = len(proteinConcentration_nM_List)
    proteinConcentration_nM_List_NP = np.array(proteinConcentration_nM_List)
    subset_indices = np.linspace(0, length - 1, length, dtype=int)
    subset_ProteinConcentration_nM_List = proteinConcentration_nM_List_NP[subset_indices]

    optimize_parameters(initial_guesses, N_p, N_m, D)

    showOptimizedModel(N_p, N_m, D)
    showModel(optimizedParameters, N_p, N_m, D)

    runParameterOptimization(N_p, N_m, D, theory_file_name)
    # Part 3
    showModel(optimizedParameters, N_p, N_m, D)

    #Clear the contents of all the lists used. All of these lists are global variables and need to be cleared for the next protein analysis
    optimizedParameters = np.array([]) # Empty the numpy array to prepare the analysis of the next protein experiment
    timeValues_List.clear()
    meanIntensity_List.clear()
    proteinConcentration_List.clear()
    proteinConcentration_nM_List.clear()
    numberOfProteinMolecules_List.clear()
    rateOfChangeProteinMolecules_List.clear()


# Part 4

def showExperimentalDataTogether():

    experimentalFiles = sorted(glob.glob("experimentalData_k*"))
    motorProteins_Names = [file.replace("experimentalData_", "").replace(".csv", "") for file in experimentalFiles]

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Mean Intensity (A.U.)'], marker='o') # Plot the "Mean Intensity" data only
    
    plt.title('Mean Intensity vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Mean Intensity (A.U.)')
    plt.grid(True)
    plt.legend(motorProteins_Names)
    plt.savefig("Mean_Intensity.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Protein Concentration (ng/µL)'], marker='o') # Plot the "Protein Concentration" data only

    plt.title('Protein Concentration vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Protein Concentration (ng/µL)')
    plt.grid(True)
    plt.legend(motorProteins_Names)
    plt.savefig("Protein_Concentration.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Protein Concentration (nM)'], marker='o') # Plot the "Number of Protein Molecules" data only

    plt.title('Protein Concentration (nM) vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Protein Concentration (nM)')
    plt.grid(True)
    plt.legend(motorProteins_Names)  
    plt.savefig("Protein_Concentration_(nM).png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Number of Protein Molecules'], marker='o') # Plot the "Number of Protein Molecules" data only

    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Number of Protein Molecules')
    plt.grid(True)
    plt.legend(motorProteins_Names)  
    plt.savefig("Number_of_Protein_Molecules.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Rate of Change of Number of Protein Molecules (PM/min)'], marker='o') # Plot the "Rate of Change of Protein Molecules" data only

    plt.title('Rate of Change of Protein Molecules vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Rate of Change of Protein Molecules (PM/min)')
    plt.grid(True)
    plt.legend(motorProteins_Names)
    plt.savefig("Rate_of_Change_of_Protein_Molecules.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop


# Part 5

def showTheoreticalDataTogether():

    theoreticalFiles = glob.glob("optimizedParameters_k*")
    motorProteins_Names = [file.replace("optimizedParameters_", "").replace(".csv", "") for file in theoreticalFiles]

    completeDataFrame = pd.DataFrame() # Initialize an empty DataFrame that will be filled in a loop
    for i in range(len(theoreticalFiles)):
        dataFrame_Protein = pd.read_csv(theoreticalFiles[i]) # Save into a Pandas dataframe ALL the data for the current protein
        dataFrame_Protein["Kinesin Motor Protein"] = motorProteins_Names[i]
        completeDataFrame = pd.concat([completeDataFrame, dataFrame_Protein], ignore_index=True)

    # "Melt" the data. Convert the DataFrame from wide format to long format.
    """With the "long" format, the DataFrame has 3 columns. One for the kinesin motor proteins, 
    another one for the names of the parameters, and the third column is for the values of the parameters.
    """
    melted_data = completeDataFrame.melt(id_vars='Kinesin Motor Protein', var_name='Parameter Name', value_name='Value of Parameter')
    
    # Create the categorical plot
    plt.figure(figsize=(15, 10))
    sns.stripplot(x='Parameter Name', y='Value of Parameter', hue='Kinesin Motor Protein', data=melted_data, dodge=True, jitter=True, alpha=0.7)
    plt.xticks(rotation=90)
    plt.title('Parameter Values for the Kinesin Motor Proteins')
    plt.show()
    
    