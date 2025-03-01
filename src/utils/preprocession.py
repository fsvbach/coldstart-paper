import os
import numpy as np
import pandas as pd

from numpy.random import Generator, PCG64

generator = Generator(PCG64(seed=0))

def sparsen(row, p):
    row = row.copy()
    idx = generator.choice(row.index, round(len(row)*p), replace=False)
    row.loc[idx] = np.nan
    return row

def impute_dataframe(dataframe, mean=None):
    if mean is None:
        mean = dataframe.mean().fillna(.5)
    return (dataframe - mean).fillna(0) + mean


def assign_probabilities(continuous_data, values):
    # Expand the continuous data for broadcasting with values
    val_matrix = np.expand_dims(continuous_data, axis=2)
    value_matrix = np.expand_dims(values, axis=0)
    value_matrix = np.expand_dims(value_matrix, axis=0)
    
    # Calculate distances from each continuous value to each discrete value
    distances = np.abs(val_matrix - value_matrix)
    
    # Calculate probabilities inversely proportional to distances
    probabilities = 1 / (distances + 1e-10)  # Adding a small constant to avoid division by zero
    
    # Normalize probabilities to ensure they sum to 1 along the last axis
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    
    return probabilities

def sample_discrete_values(probabilities, values):
    # Get cumulative probabilities
    cumulative_probs = np.cumsum(probabilities, axis=2)
    # Generate random numbers in the range [0, 1]
    random_samples = generator.random(cumulative_probs.shape[:2] + (1,))
    # Find the indices where the random number is less than the cumulative probabilities
    return values[np.argmax(cumulative_probs > random_samples, axis=2)]

def discretize(df, values=np.array([0, 0.5, 1])):
    probabilities = assign_probabilities(df.values, values)
    discrete_values = sample_discrete_values(probabilities, values)
    return pd.DataFrame(discrete_values, index=df.index, columns=df.columns)


def load_results(directory = '../results/aqvaa'):
    results   = []
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename.count('_') == 6:  # Check if the file matches the naming pattern
            # Construct the full file path
            filepath = os.path.join(directory, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(filepath, index_col=0)
            # Extract parts of the filename minus the extension
            parts = filename[:-4].split('_')  # Remove '.csv' and split by underscore
            # Add new columns to the DataFrame
            results.append((*parts, df))

    return pd.DataFrame(results, columns=['InitialData', 'Method', 'NumberQueries', 'Model', 'UpdateRate', 'NumberVoters', 'Seed', 'DataFrame'])


def normalize_and_round(matrix):
    """
    Normalize each row of the matrix so that its sum equals 100, then
    round the normalized values to integers such that each row sums to 100.
    
    Parameters:
        matrix (np.ndarray): A 2D numpy array.
        
    Returns:
        np.ndarray: A 2D array of integers with each row summing to 100.
    """
    matrix = np.array(matrix, dtype=float)
    rounded_matrix = np.zeros_like(matrix)
    
    # Process each row individually.
    for i, row in enumerate(matrix):
        row_sum = row.sum()
        if row_sum == 0:
            # If the row sums to zero, we can either leave it as all zeros or decide on a fallback.
            continue
        
        # Normalize the row to sum to 100.
        normalized = row / row_sum * 100
        
        # Get the integer parts and the remainders.
        int_parts = np.floor(normalized)
        remainders = normalized - int_parts
        
        # Compute how many points we still need to assign.
        current_sum = int(int_parts.sum())
        remainder_points = int(100 - current_sum)
        
        # Get indices sorted by the remainder (largest first)
        indices = np.argsort(-remainders)
        for j in range(remainder_points):
            int_parts[indices[j]] += 1
            
        rounded_matrix[i] = int_parts
        
    return rounded_matrix.astype(int)