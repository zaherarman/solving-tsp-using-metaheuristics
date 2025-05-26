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

from collections import deque

def get_neighbors(solution):

    """

    Helper function for Tabu Searhc
    Generate neighbors by swapping any two cities (2-swap).
    Returns a list of neighbor solutions (each a list of city names).

    """

    neighbors = []
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

def tabu_search(city_distances_df, N = 200, M = 50):
    
    """
    N: max number of iterations (outer loop).
    M: max size of the tabu list (number of routes to remember).
    
    """
    
    current_route = generate_random_solution(city_distances_df)
    
    def get_distance(route):
        return compute_route_distance(route, city_distances_df)
    
    current_distance = get_distance(current_route)
    
    ts_best_route = current_route[:]
    ts_best_distance = current_distance
    
    # Using a deque to store tabu routes
    tabu_list = deque()
    
    for i in range(N):
        # Get all neighbors (2-swap)
        neighbors = get_neighbors(current_route)
        
        # Filtering out neighbors in the tabu list
        allowed_neighbors = []
        for nb in neighbors:
            if nb not in tabu_list:
                allowed_neighbors.append(nb)
        
        # If all neighbors are tabu, consider them
        if len(allowed_neighbors) == 0:
            allowed_neighbors = neighbors
        
        # Picking the best neighbor
        best_neighbor = None
        best_neighbor_dist = float('inf')
        
        for nb in allowed_neighbors:
            dist = get_distance(nb)
            if dist < best_neighbor_dist:
                best_neighbor_dist = dist
                best_neighbor = nb
        
        # Moving to that best neighbor
        current_route = best_neighbor
        current_distance = best_neighbor_dist
        
        # Updating global best
        if current_distance < ts_best_distance:
            ts_best_distance = current_distance
            ts_best_route = current_route[:]
        
        # Adding to tabu list
        tabu_list.append(best_neighbor)
        
        # If tabu list is too large, remove the oldest
        if len(tabu_list) > M:
            tabu_list.popleft()
    
    return ts_best_route, ts_best_distance

def test_tabu_search():
    
    file_path = PROCESSED_DATA_DIR / "distance_matrix.csv"
    city_distances_df = pd.read_csv(file_path, index_col=0)
    
    global ts_best_route, ts_best_distance
    ts_best_route, ts_best_distance = tabu_search(city_distances_df)
    
    # Save results to models/genetic_algorithm/
    output_dir = PROCESSED_DATA_DIR / "tabu_search"

    output_path = output_dir / "best_result.json"
    with open(output_path, "w") as f:
        json.dump({
            "best_route": ts_best_route,
            "best_distance": ts_best_distance
        }, f, indent=4)

if __name__ == "__main__":
    typer.run(test_tabu_search)