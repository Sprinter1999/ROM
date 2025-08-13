import matplotlib.pyplot as plt
import numpy as np
import ast

# Define class names based on the provided image
class_names = [
    'Agriculture',
    'Airport',
    'Bareland',
    'Beach',
    'Bridge',
    'Forest',
    'Highway',
    'Industrial',
    'Meadow',
    'Mountain',
    'Overpass',
    'Parkinglot',
    'Port',
    'Residential',
    'River'
]

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(ast.literal_eval(line.strip()))
    return np.array(data)

def create_scatter_plot(data, title, filename, is_difference=False):
    plt.figure(figsize=(12, 8))
    
    if is_difference:
        diff_points_x, diff_points_y, diff_sizes = [], [], []
        gray_points_x, gray_points_y, gray_sizes = [], [], []

        max_size = np.max(np.abs(data)) # For scaling marker size

        for client_idx, client_data in enumerate(data):
            for class_idx, count in enumerate(client_data):
                if count != 0:
                    diff_points_x.append(client_idx)
                    diff_points_y.append(class_idx)
                    diff_sizes.append(np.abs(count) * 500 / max_size)
                else:
                    gray_points_x.append(client_idx)
                    gray_points_y.append(class_idx)
                    gray_sizes.append(np.abs(count) * 500 / max_size)
        
        # Plot gray points first (no change)
        plt.scatter(gray_points_x, gray_points_y, s=gray_sizes, c='gray', alpha=0.7, edgecolors='w', linewidth=0.5)
        # Plot all non-zero differences in a single color (e.g., a lighter red)
        plt.scatter(diff_points_x, diff_points_y, s=diff_sizes, c='lightcoral', alpha=0.7, edgecolors='w', linewidth=0.5)

    else:
        client_ids = []
        class_indices = []
        sizes = []
        colors = []

        max_size = np.max(np.abs(data)) # For scaling marker size

        for client_idx, client_data in enumerate(data):
            for class_idx, count in enumerate(client_data):
                if count != 0:
                    client_ids.append(client_idx)
                    class_indices.append(class_idx)
                    sizes.append(np.abs(count) * 500 / max_size)
                    colors.append(plt.cm.viridis(class_idx / len(class_names))) # Color by class

        plt.scatter(client_ids, class_indices, s=sizes, c=colors, alpha=0.7, edgecolors='w', linewidth=0.5)

    plt.yticks(np.arange(len(class_names)), class_names, fontsize=18) # Increased font size
    plt.xlabel('Client ID', fontsize=20) # Increased font size
    plt.ylabel('Class Name', fontsize=20) # Increased font size
    plt.title(title, fontsize=22) # Increased font size
    plt.xticks(fontsize=18) # Increased font size
    plt.grid(False) # Removed grid lines
    plt.xlim(-1, 135) # Set x-axis limit from 0 to 135
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Load data
clean_data = load_data('clean_distributions.txt')
noisy_data = load_data('noisy_distributions.txt')

# Calculate the difference for label flipping visualization
difference_data = noisy_data - clean_data

# Create scatter plots with updated titles
create_scatter_plot(clean_data, 'Original Label Distribution', 'original_distribution_scatter_v10.png')
create_scatter_plot(noisy_data, 'Noisy Label Distribution', 'noisy_distribution_scatter_v10.png')
create_scatter_plot(difference_data, 'Label Flipping (Noisy - Original)', 'label_flipping_scatter_v10.png', is_difference=True)

print('Scatter plots generated: original_distribution_scatter_v10.png, noisy_distribution_scatter_v10.png, and label_flipping_scatter_v10.png')