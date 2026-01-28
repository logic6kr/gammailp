import pandas as pd
import numpy as np
import os 

variables = 5

def generate_data(num_samples=1000, num_features=5, num_classes=2):
    """
    Generate a synthetic dataset with the specified number of samples, features, and classes.
    
    Args:
        num_samples (int): Number of samples to generate.
        num_features (int): Number of features for each sample.
        num_classes (int): Number of classes for classification.    
    """
    # Generate random features

    size = (300, 5)  # Example size (3 rows, 4 columns)
    random_array = np.random.randint(2, size=size)  # Generates 0 or 1
    print(random_array)
    label = []
    for row in random_array:
        first_item = row[0]
        thrid_item = row[2]
        if first_item == 1 and thrid_item == 1:
            label.append(1)
        else:
            label.append(0)
    label = np.array(label)
    final_data = np.column_stack((random_array, label))
    # Convert to DataFrame
    df = pd.DataFrame(final_data, columns=[f'feature_{i}' for i in range(num_features)] + ['label'])
    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)
    # Save to CSV
    df.to_csv('gammaILP/logic_back/demo_data.csv', index=False)

if __name__ == "__main__":
    # Generate data with 1000 samples, 5 features, and 2 classes
    generate_data(num_samples=300, num_features=5, num_classes=2)
    print("Data generation complete. Data saved to 'demo_data.csv'.")
        