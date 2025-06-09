import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'Times New Roman'
# Set the directory containing CSV files
acceleration_dir = "Acceleration"

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(acceleration_dir, "full_forward.csv"))
axis = 'ay'
# Define the filter window size
window_size = 10

# Create a figure with subplots
plt.figure(figsize=(14, 8))

# Plot each file
for file_path in csv_files:
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert DataFrame columns to numpy arrays
        time = df['time'].to_numpy()
        total_acc = df[axis].to_numpy()
        
        # Get the filename without extension for the legend
        file_name = os.path.basename(file_path).replace('.csv', '')
        
        # Apply moving average filter
        # time = time[window_size//2:-window_size//2+1]
        # total_acc = np.convolve(total_acc, np.ones(window_size)/window_size, mode='valid')
        
        # Plot time vs total acceleration
        plt.plot(time, total_acc, label=file_name)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# Customize the plot
plt.xlabel('Time (s)')
if axis == 'ax':
    plt.ylabel('Lateral Acceleration (m/s²)')
elif axis == 'ay':
    plt.ylabel('Forward Acceleration (m/s²)')
elif axis == 'az':
    plt.ylabel('Vertical Acceleration (m/s²)')
# plt.title('Time vs x Acceleration for Different Scenarios')
plt.grid(True)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
# plt.show()
file_name = 'full_forward_raw_acceleration'
plt.savefig(file_name + '.pdf')
# plt.close()

print("Plot has been saved as '" + file_name + ".pdf'") 