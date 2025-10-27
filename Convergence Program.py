import numpy as np
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import pandas as pd

# --- Set precision for Decimal (high enough for large Fibonacci numbers) ---
getcontext().prec = 10  # Adjust if higher precision is needed

# --- Random Fibonacci Sequence Generator with variable p(n) ---
def random_fibonacci_pn(p_func, n_terms, a, b):
    """
    Generates the Random Fibonacci Sequence with probabilistic addition/subtraction.
    
    Parameters:
        p_func (function): Function returning probability p(n) at index n
        n_terms (int): Number of Fibonacci terms to generate
        a (float): Parameter controlling slope of p(n)
        b (float): Parameter controlling steepness of p(n)
        
    Returns:
        np.ndarray: Fibonacci sequence as Decimal objects
    """
    F = np.zeros(n_terms, dtype=object)
    F[0], F[1] = Decimal(1), Decimal(1)
    
    for n in range(2, n_terms):
        p = p_func(n, n_terms, a, b)
        F[n] = F[n-1] + F[n-2] if np.random.rand() < p else F[n-1] - F[n-2]
    
    return F

# --- Exact nth root calculation using Decimal ---
def exact_nth_root(Fn, n):
    """
    Computes the nth root of |F_n| with high precision.
    """
    return Decimal(abs(Fn)) ** (Decimal(1) / Decimal(n))

# --- Flexible probability function p(n) ---
def p_function(n, n_terms, a, b):
    """
    Example probability function: decreasing power function
    """
    # return 1-1 / (n ** a)
    # return 1/2
    # return 2/3
    # return 1/3
    # return 1/np.log(float(n+10)) 
    # return 1- 1/np.log(float(n+10)) 
    return abs(np.sin(n/a))

# --- Load mean value from Excel ---
def load_mean_from_excel(file_path, sheet_name="Sheet1"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df['mean'].iloc[0]  # assumes column 'mean' exists

# --- Run multiple trials and plot nth root growth ---
def run_trials(p_func, n_terms, k, a, b, mean):
    """
    Run k trials of Random Fibonacci Sequence and plot nth root growth.
    """
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'y', 'r', 'purple']  # colors for trials

    all_growth = []

    for trial in range(k):
        print(f"Running trial {trial + 1}...")
        F = random_fibonacci_pn(p_func, n_terms, a, b)
        growth = [exact_nth_root(F[n], n+1) for n in range(n_terms)]
        all_growth.append([float(g) for g in growth])
        
        # Plot nth root growth for trial with thinner lines
        plt.plot(range(1, n_terms+1), all_growth[-1], color=colors[trial % len(colors)], 
                 label=f"Trial {trial + 1}", linewidth=0.8)
        print(f"Trial {trial + 1} final nth root: {float(growth[-1]):.8f}")

    # Plot mean line with slightly thicker dashed line
    plt.axhline(mean, color='k', linestyle='--', label=f"Mean value: {mean}", linewidth=1.2)

    # Zoomed y-axis based on typical nth root range
    plt.ylim(0.95, 1.5)

    plt.xlabel('Index n')
    plt.ylabel('Nth Root')
    plt.title('Nth Root Growth of Random Fibonacci Sequence')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.show()

# --- Main function ---
def main():
    n_terms = 10000  # number of Fibonacci terms
    k = 5  # number of trials
    a, b = 1, 1  # parameters controlling p(n)

    # Load mean from Excel file
    file_path = "C:/Users/Josh Reynolds/Documents/Indpt Study - Prob Theory/Paper/excel files/Functions/Sin Function/RFS_Data_n_10000_a1.xlsx"
    mean = load_mean_from_excel(file_path)

    # Run trials and plot
    run_trials(p_function, n_terms, k, a, b, mean)

# --- Run the program ---
if __name__ == "__main__":
    main()
