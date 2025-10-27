import numpy as np
from decimal import Decimal, getcontext
import pandas as pd  # Import pandas
import matplotlib.pyplot as plt
import os  # For checking file existence to create unique file names

# Set precision for Decimal (high enough for large Fibonacci numbers)
getcontext().prec = 10  # You can adjust this depending on the required precision

# Define the Random Fibonacci Sequence function with a variable p(n)
def random_fibonacci_pn(p_func, n_terms, a, b):
    """
    Generates the Fibonacci sequence with a random addition/subtraction rule,
    where the probability of addition p(n) varies based on the index n.
    
    Parameters:
    p_func (function): A function that takes an index n and returns the probability p(n).
    n_terms (int): The number of Fibonacci terms to generate.
    a (float): Parameter to control the slope of p(n).
    b (float): Parameter to control the "flatness" or "steepness" of the probability.
    
    Returns:
    np.ndarray: The Fibonacci sequence F.
    """
    F = np.zeros(n_terms, dtype=object)  # Use object type to allow for Decimal objects
    F[0], F[1] = Decimal(1), Decimal(1)  # Initial values F_0 = 1, F_1 = 1
    
    for n in range(2, n_terms):
        p = p_func(n, n_terms, a, b)  # Get p(n) for the current index n
        # Decide whether to add or subtract based on p(n)
        F[n] = F[n-1] + F[n-2] if np.random.rand() < p else F[n-1] - F[n-2]
    
    return F

# Define the nth root function for high precision using Decimal
def exact_nth_root(F, n):
    """
    Computes the exact nth root of |F_n| using high precision.
    
    Parameters:
    F (int): The nth Fibonacci number.
    n (int): The index n for which to compute the nth root.
    
    Returns:
    Decimal: The exact nth root of |F_n|.
    """
    # Convert Fibonacci number to Python's native int type for compatibility with Decimal
    F_decimal = Decimal(abs(F))  # Convert to Decimal for higher precision
    nth_root = F_decimal ** (Decimal(1) / Decimal(n))  # Compute the exact nth root
    return nth_root

# Define a flexible p_function where you can specify any function
def p_function(n, n_terms, a, b):
    """
    A flexible probability function p(n), where the form of p(n) can be defined dynamically.
    
    Parameters:
    n (int): The index of the Fibonacci sequence.
    n_terms (int): The total number of Fibonacci terms.
    a (float): Parameter to control the slope of p(n).
    b (float): Parameter to control the steepness of the probability function.
    
    Returns:
    float: The probability p(n) at index n.
    """
    
    ### Functions to run ###
    # return 1-1/(n**a)
    # return 1/(n**a)

    # return 1/np.log(float(n+10)) 
    # return 1- 1/np.log(float(n+10)) 

    # return 1/3

    # return abs(np.sin(n/a))

    return abs(np.sin(np.log(n)))

# Function to run multiple trials and export results
def run_trials(p_func, n_terms, k, a, b):
    """
    Run k trials of generating a random Fibonacci sequence with p(n) and calculating the nth root growth.
    
    Parameters:
    p_func (function): The function for p(n) that defines the probability for each index n.
    n_terms (int): The number of Fibonacci terms to generate.
    k (int): The number of trials to run per iteration.
    a (float): Parameter to control the slope of p(n).
    b (float): Parameter to control the "flatness" or "steepness" of p(n).
    """
    all_growth_values = []  # Store nth root values for each trial
    
    # Initialize empty DataFrame to store data
    df = pd.DataFrame()
    
    # Run k trials
    for trial in range(k):
        print(f"Running trial {trial + 1} of {k}...")
        F = random_fibonacci_pn(p_func, n_terms, a, b)  # Generate the Fibonacci sequence
        growth = [exact_nth_root(F[n], n+1) for n in range(n_terms)]  # Calculate the nth root growth
        all_growth_values.append([float(growth[n_terms-1])])  # Collect the nth root value of the last Fibonacci number
        print(f"Trial {trial + 1}: {round(float(growth[n_terms-1]), 8)}")  # Print nth root for current trial

    # Define the Excel file path based on `n_terms` and `a`
    file_path = f"C:/Users/Josh Reynolds/Documents/Indpt Study - Prob Theory/Paper/excel files/Functions/Sin Log/RFS_Data_n_{n_terms}_a{a}.xlsx"
    
    # Add results to the DataFrame in a new column
    df[f"Trial Results"] = [value[0] for value in all_growth_values]
    
    # Check if the file already exists and append if necessary
    if os.path.exists(file_path):
        # If file exists, append the results
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, index=False, header=False)
    else:
        # If the file does not exist, create a new one
        df.to_excel(file_path, index=False)

    print(f"Data has been saved to '{file_path}'.")


# Main function to control the parameters and run the experiment
def main():
    n_terms = 10000 # Set the number of Fibonacci terms
    k = 1000  # Set the number of trials per run
    a =  1 #Control the slope of p(n)
    b = 1  # Control the steepness of the curve
    
    
    # Run the trials
    run_trials(p_function, n_terms, k, a, b)


# Call the main function
if __name__ == "__main__":
    main()
