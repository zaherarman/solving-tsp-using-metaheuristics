from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import random 
import numpy as np
import pandas as pd
import json

from src.metaheuristics.utils import compute_route_distance, generate_random_solution 
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

def crossover(parent1, parent2):
    size = len(parent1)
    
    # Pick two cut points
    cut_points = random.sample(range(size), 2)
    a = min(cut_points)
    b = max(cut_points)
    
    child = [None] * size
    
    # Copy slice from parent1
    for i in range(a, b):
        child[i] = parent1[i]
    
    # Fill remaining with cities from parent2 in order
    parent2_index = 0
    for i in range(size):
        if child[i] is None:
            while parent2[parent2_index] in child:
                parent2_index += 1
            child[i] = parent2[parent2_index]
            parent2_index += 1
    
    return child

def mutate(route):

    #Swap mutation: pick two positions at random and swap them.
    i, j = random.sample(range(len(route)), 2)
    temp = route[i]
    route[i] = route[j]
    route[j] = temp

def genetic_algorithm(city_distances_df, population_size = 50, crossover_rate = 0.6, mutation_rate = 0.1, elitist_rate = 0.1, alpha = 500.0, max_generations = 300):

    """

    population_size: number of routes in the population.
    crossover_rate: fraction of population used for crossover.
    mutation_rate: fraction of offspring that get mutated.
    elitist_rate: fraction of top routes carried over to next generation.
    alpha: stop early if the best_distance < alpha.
    max_generations: maximum iterations before stopping.

    """

    # Step 1: Initialize random population
    population = []
    for _ in range(population_size):
        route = generate_random_solution(city_distances_df)
        population.append(route)

    def get_distance(route):
        return compute_route_distance(route, city_distances_df)

    ga_best_route = None
    best_distance = float('inf')

    generation = 0
    while generation < max_generations:

        # 1. Evaluate population 
        population_with_distances = []
        for route in population:
            dist = get_distance(route)
            population_with_distances.append((route, dist))
        
        # Sort by distance ascending (best = smallest distance)
        population_with_distances.sort(key=lambda x: x[1])
        
        # Current best route in this generation
        current_best_ga_route, current_best_distance = population_with_distances[0]
        
        # Updating global best
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            ga_best_route = current_best_ga_route[:]
        
        # Early stopping condition
        if best_distance < alpha:
            break
        
        # 2. Selection (Elite solutions)
        new_population = []
        num_elite = int(elitist_rate * population_size)
        
        for i in range(num_elite):
            elite_route = population_with_distances[i][0]
            new_population.append(elite_route)
        
        # 3. Crossover
        num_parents = int(crossover_rate * population_size)
        parents = []
        
        for i in range(num_parents):

            # top 'i' parents
            parent_route = population_with_distances[i][0]
            parents.append(parent_route)
        
        random.shuffle(parents)
        
        offspring = []

        # Pair em up
        for i in range(0, num_parents, 2):
            if i + 1 < num_parents:
                p1 = parents[i]
                p2 = parents[i+1]

                # two children
                child1 = crossover(p1, p2)
                child2 = crossover(p2, p1)
                offspring.append(child1)
                offspring.append(child2)
        
        # 4. Mutation
        num_to_mutate = int(mutation_rate * len(offspring))
        
        for i in range(num_to_mutate):

             #  Pick random child
            r = random.randrange(len(offspring)) 
            mutate(offspring[r])
        
        # Adding offspring to new population

        for child in offspring:
            new_population.append(child)
        
        # 5. Survival
        # Combine old + new, then pick top 'population_size'
        combined = []
        
        # Adding old population (with known distances)
        for route_dist_pair in population_with_distances:
            combined.append(route_dist_pair)
        
        # Adding new offspring (need to compute distance)
        for child in offspring:
            dist = get_distance(child)
            combined.append((child, dist))
        
        # Sorting combined
        combined.sort(key=lambda x: x[1])
        
        # Rebuilding population with top routes
        next_population = []
        for i in range(population_size):
            best_combined_route = combined[i][0]
            next_population.append(best_combined_route)
        
        population = next_population
        generation += 1

    return ga_best_route, best_distance

def test_genetic_algorithm():
    file_path = PROCESSED_DATA_DIR / "distance_matrix.csv"
    city_distances_df = pd.read_csv(file_path, index_col=0)

    global ga_best_route, ga_best_distance
    ga_best_route, ga_best_distance = genetic_algorithm(city_distances_df)
    
    # Save results to models/genetic_algorithm/
    output_dir = PROCESSED_DATA_DIR / "genetic_algorithm"

    output_path = output_dir / "best_result.json"
    with open(output_path, "w") as f:
        json.dump({
            "best_route": ga_best_route,
            "best_distance": ga_best_distance
        }, f, indent=4)

if __name__ == "__main__":
    typer.run(test_genetic_algorithm)