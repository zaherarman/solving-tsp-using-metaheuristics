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

def metropolis_hastings(city_distances_df, N = 500, temperature = 1000.0):

    """
    
    city_distances_df: DataFrame of distances
    N: number of iterations
    temperature: constant temperature for acceptance probability
    
    """
    current_route = generate_random_solution(city_distances_df)
        
    def get_distance(route):
        return compute_route_distance(route, city_distances_df)
    
    current_distance = get_distance(current_route)
    
    mh_best_route = current_route[:]
    mh_best_distance = current_distance
    
    for i in range(N):
        
        # Picking a neighbor by swapping two cities
        neighbor = current_route[:]
        i, j = random.sample(range(len(neighbor)), 2)
        temp = neighbor[i]
        neighbor[i] = neighbor[j]
        neighbor[j] = temp
        
        neighbor_distance = get_distance(neighbor)
        
        # Accept if better, or with probability
        if neighbor_distance < current_distance:
            current_route = neighbor
            current_distance = neighbor_distance
        else:
            # Metropolis acceptance
            delta = neighbor_distance - current_distance
            acceptance_prob = np.exp(-delta / temperature)
            if random.random() < acceptance_prob:
                current_route = neighbor
                current_distance = neighbor_distance
        
        # Updating best
        if current_distance < mh_best_distance:
            mh_best_distance = current_distance
            mh_best_route = current_route[:]
    
    return mh_best_route, mh_best_distance

def test_metropolis_hastings():
    
    file_path = PROCESSED_DATA_DIR / "distance_matrix.csv"
    city_distances_df = pd.read_csv(file_path, index_col=0)
    
    global mh_best_route, mh_best_distance
    mh_best_route, mh_best_distance = metropolis_hastings(city_distances_df)
    
    # Save results to models/genetic_algorithm/
    output_dir = PROCESSED_DATA_DIR / "metropolis_hastings"

    output_path = output_dir / "best_result.json"
    with open(output_path, "w") as f:
        json.dump({
            "best_route": mh_best_route,
            "best_distance": mh_best_distance
        }, f, indent=4)

if __name__ == "__main__":
    typer.run(test_metropolis_hastings)