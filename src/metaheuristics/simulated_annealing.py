from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import random
import numpy as np
import pandas as pd
import json

from src.metaheuristics.utils import generate_random_solution, compute_route_distance
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

def simulated_annealing(city_distances_df, N = 100, M = 20, T_start = 1000.0, cooling_rate = 0.95):

    """

    N: number of "temperature steps" (outer loop).
    M: number of Metropolis iterations per temperature (inner loop).
    T_start: initial temperature.
    cooling_rate: factor to multiply temperature by each outer loop.
    """

    current_route = generate_random_solution(city_distances_df)
    
    def get_distance(route):
        return compute_route_distance(route, city_distances_df)
    
    current_distance = get_distance(current_route)
    
    sa_best_route = current_route[:]
    sa_best_distance = current_distance
    
    T = T_start
    
    for i in range(N):  
        for j in range(M):

            # Creating a neighbor by swapping two cities
            neighbor = current_route[:]
            i, j = random.sample(range(len(neighbor)), 2)
            temp = neighbor[i]
            neighbor[i] = neighbor[j]
            neighbor[j] = temp
            
            neighbor_distance = get_distance(neighbor)
            
            # If better, accept
            if neighbor_distance < current_distance:
                current_route = neighbor
                current_distance = neighbor_distance
            else:
                
                # Probability e^(-(delta)/T)
                delta = neighbor_distance - current_distance
                acceptance_prob = np.exp(-delta / T)
                if random.random() < acceptance_prob:
                    current_route = neighbor
                    current_distance = neighbor_distance
            
            # Updating best if found better
            if current_distance < sa_best_distance:
                sa_best_distance = current_distance
                sa_best_route = current_route[:]
        
        # Decreasing temperature
        T = T * cooling_rate
    
    return sa_best_route, sa_best_distance

def test_simulated_annealing():
    
    file_path = PROCESSED_DATA_DIR / "distance_matrix.csv"
    city_distances_df = pd.read_csv(file_path, index_col=0)
    
    global sa_best_route, sa_best_distance
    sa_best_route, sa_best_distance = simulated_annealing(city_distances_df)
    
    # Save results to models/genetic_algorithm/
    output_dir = PROCESSED_DATA_DIR / "simulated_annealing"

    output_path = output_dir / "best_result.json"
    with open(output_path, "w") as f:
        json.dump({
            "best_route": sa_best_route,
            "best_distance": sa_best_distance
        }, f, indent=4)

if __name__ == "__main__":
    typer.run(test_simulated_annealing)