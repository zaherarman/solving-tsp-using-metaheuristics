from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import random

app = typer.Typer()

def compute_route_distance(route, city_distances_df):

    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += city_distances_df.loc[route[i], route[i+1]]

    # Returning to the start city
    total_distance += city_distances_df.loc[route[-1], route[0]]
    return total_distance

def generate_random_solution(city_distances_df):
    
    #Generates a random permutation of the city names from the DataFrame index.
    city_names = list(city_distances_df.index)
    random.shuffle(city_names)
    return city_names