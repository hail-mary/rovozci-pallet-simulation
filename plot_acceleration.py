import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'
# Set the directory containing CSV files
acceleration_dir = "Acceleration"

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(acceleration_dir, "half_*.csv"))

# Set font sizes
# plt.rcParams['font.size'] = 15
# plt.rcParams['axes.titlesize'] = 19
# plt.rcParams['axes.labelsize'] = 17
# plt.rcParams['xtick.labelsize'] = 15
# plt.rcParams['ytick.labelsize'] = 15
# plt.rcParams['legend.fontsize'] = 13

# Create a figure with subplots
plt.figure(figsize=(15, 10))

# Plot each file
for file_path in csv_files:
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert DataFrame columns to numpy arrays
        time = df['time'].to_numpy()
        total_acc = df['ay'].to_numpy()
        
        # Get the filename without extension for the legend
        file_name = os.path.basename(file_path).replace('.csv', '')
        
        # Plot time vs total acceleration
        plt.plot(time, total_acc, label=file_name)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# Customize the plot
plt.xlabel('Time (s)')
plt.ylabel('y Acceleration (m/s²)')
# plt.title('Time vs x Acceleration for Different Scenarios')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.show()
# plt.savefig('x_fullacceleration_plot.png', bbox_inches='tight', dpi=300)
# plt.close()

print("Plot has been saved as 'acceleration_plot.png'") 