import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.stats import norm
from sklearn.mixture import GaussianMixture

def get_csv_path():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print('script directory:', script_dir)

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]

    if csv_files:
        print(f"Found whisperseg CSV files: {csv_files}")
        return f"{os.path.join(script_dir, csv_files[0])}", script_dir
    
    else:
        print("No whisperseg CSV files found in the current directory.")
        print("Folder path:",script_dir)
        quit()

def read_csv(file_path):
    return pd.read_csv(file_path)

def calculate_duration(x):
    return x['offset'].max() - x['onset'].min()
    

def calculate_num_syls(x):
    return len(x)

def calculate_stats_from_df(df):
    df_stats = df.groupby('filename').agg(
        duration=('offset', lambda x: calculate_duration(df.loc[x.index])),
        num_syl=('filename', 'count'),  # Count occurrences per filename
        syl_per_sec=('filename', lambda x: len(x) / calculate_duration(df.loc[x.index]))
        ).reset_index()
    return df_stats

def calculate_gmm_components(data):
        bics = []
        aics = []
        for k in range(1, 10):  # Testing 1 to 10 components
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(data.reshape(-1, 1))
            bics.append(gmm.bic(data.reshape(-1, 1)))
            aics.append(gmm.aic(data.reshape(-1, 1)))
        plt.plot(range(1, 10), bics, label='BIC')
        plt.plot(range(1, 10), aics, label='AIC')
        plt.xlabel('Number of Components')
        plt.ylabel('Score')
        plt.legend()
        plt.title('BIC and AIC for Model Selection')
        plt.show()

        lowest_bic = np.argmin(bics) + 1  # Choose the number of components with the lowest BIC
        lowest_aic = np.argmin(aics) + 1  # Choose the number of components with the lowest AIC

        # Choose the number of components based on BIC or AIC
        if lowest_bic < lowest_aic:
            num_components = lowest_bic
        else:
            num_components = lowest_aic

        print('num components:',num_components)

        return num_components

def copy_files_to(keep_sessions, script_path, copy_folder_name ='keep',verbose=False):      
    os.makedirs(f"{script_path}\{copy_folder_name}", exist_ok=True)

    for filename in keep_sessions['filename'].tolist(): 
        source_file = os.path.join(script_path, filename)
        destination_file = os.path.join(script_path, copy_folder_name, filename)

        # Checks if source_file is a file and not a directory before copying
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file) #copy2 preserves metadata
            if verbose:
                print(source_file, "copied to", destination_file)

    return



csv_path, script_path = get_csv_path()
print("Script path:",script_path)

# Read the CSV file into a DataFrame
session_df = read_csv(csv_path)

# Display Dataframe
print(session_df)

# Caculate duration, number of syllables, and syl/s from DataFrame
session_stats = calculate_stats_from_df(session_df)

# Display stats Dataframe
print(session_stats)


# Syllable 20-30 ms duration   cut above 20syl/sec
data = np.array(session_stats[session_stats['syl_per_sec'] < 20]['syl_per_sec'].to_list()) 

# Fit GMM
#num_components = calculate_gmm_components(data)  # Choose an appropriate number of Gaussians
num_components = 3  # Choose an appropriate number of Gaussians
gmm = GaussianMixture(n_components=num_components, random_state=42)
gmm.fit(data.reshape(-1, 1))

# Plot histogram
plt.hist(data, bins=1000, density=True, alpha=0.6, color='gray')

# Generate curve
gmm_x = np.linspace(min(data), max(data), 1000)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

# Plot the estimated Gaussian components
plt.plot(gmm_x, gmm_y, label=f'{num_components}-Component GMM', color='grey',alpha=0.05)

# Get GMM parameters
weights = gmm.weights_  # Mixing coefficients
means = gmm.means_.flatten()  # Means of Gaussians
covariances = gmm.covariances_.flatten()  # Variances (diagonal covariance)

for weight, mean, cov in zip(weights, means, covariances):
    pdf = weight * norm.pdf(gmm_x, mean, np.sqrt(cov))  # Compute weighted Gaussian PDF
    plt.plot(gmm_x, pdf, linewidth=2, label=f"Component (μ={mean:.2f}, σ={np.sqrt(cov):.2f})")

plt.legend()
plt.xlabel('Data Values')
plt.ylabel('Density')
plt.title('Gaussian Mixture Model Fit')
plt.show()

# Predict component labels for each data point
component_labels = gmm.predict(data.reshape(-1, 1))

# Get indices for each component
component_indices = {i: np.where(component_labels == i)[0] for i in range(num_components)}

# Print indices for each component
for comp, indices in component_indices.items():
    print(f"Component {comp}: {len(indices)} points, Indices: {indices[:10]}...")  # Show first 10 indices

#print(component_indices[1])

#return

# Combine second and last indicies
#keep_indicies = np.concatenate((component_indices[1], component_indices[2]))

keep_indicies = component_indices[1]

# Filter out the required sessions from stats df
keep_sessions = session_stats.iloc[keep_indicies]

# Make dir and copys over keep from filenames from df_stats
copy_files_to(keep_sessions, script_path, copy_folder_name ='keep',verbose=True)

# Copy stats csv to keep 
session_stats.to_csv(os.path.join(script_path,'keep','session_stats.csv'), index=False)


'''
Notes:
guassian distribution mixture modeling (2nd bell curve pass) (overlap with histogram)
manual thresholds??
inspect the overlap of the two first distributions

check validitity with random sampling check with Test and Train data set

pull to github

'''





