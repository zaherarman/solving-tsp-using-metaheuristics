from pathlib import Path

from loguru import logger
from tqdm import tqdm
from typing import List

import typer

import pandas as pd
import requests

from src.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

URLS = {
    "city_names.txt":  "https://people.sc.fsu.edu/~jburkardt/datasets/cities/usca312_name.txt",
    "ctc_distance.txt":"https://people.sc.fsu.edu/~jburkardt/datasets/cities/usca312_dist.txt",
    "dms_coordinates.txt":"https://people.sc.fsu.edu/~jburkardt/datasets/cities/usca312_dms.txt",
    "xy_coordinates.txt": "https://people.sc.fsu.edu/~jburkardt/datasets/cities/usca312_xy.txt",
}

@app.command("download")
def download_all():
    
    """
    
    Fetch all raw .txt files into data/external.
    
    """
    
    for name, url in URLS.items():
        dataset = EXTERNAL_DATA_DIR / name
        if dataset.exists():
            typer.echo(f"✓ {name} already exists")
            continue
        typer.echo(f"Downloading {name}")
        resp = requests.get(url)
        resp.raise_for_status()
        dataset.write_bytes(resp.content)
        typer.echo(f" saved {dataset}")

def read_cities(filepath: Path) -> pd.DataFrame:
    
    """
    
    Reads a .txt file containing city name and returns a DataFrame.

    """
    
    cities_df = pd.read_csv(filepath, header=None, names=["City","State/Province"], comment="#")
    return cities_df

def get_canadian_cities(cities_df: pd.DataFrame) -> pd.DataFrame:
    
    '''

    Return a Dataframe that contains only canadian cities, filtered by province abbreviation
    
    AB: Alberta
    BC: British Columbia
    MB: Manitoba
    NB: New Brunswick
    NF: Newfoundland
    NS: Nova Scotia
    NT: Northwest Territory
    ON: Ontario
    PE: Prince Edward Island
    QC: Quebec
    SK: Saskatchewan
    YT: Yukon Territory
    
    '''
    
    # Filtering table for Canadian cities
    return cities_df[cities_df['State/Province'].str.contains("AB|BC|MB|NB|NF|NS|NT|ON|PE|QC|SK|YT", case = False)]

def read_city_to_cities_distance(filepath : str, canadian_cities_df : pd.DataFrame, index_numbers :  List) -> pd.DataFrame:
    '''
    
    Reads a .txt file containing distances to each city from each city.

    
    '''
  
    # Reading all the distances seperately 
    ctc_distance_df = pd.read_csv(filepath, sep = r'\s+', header = None, comment = '#')
    column_ctc_distance_df = ctc_distance_df.stack()

    # Turning into a 312x312 matrix. Distances to each city. Flatten causes conversion to NumPy array, so turning back into DataFrame
    distance_matrix_df = pd.DataFrame(column_ctc_distance_df.values.reshape(312, 312))

    # Drop rows with NaN 
    distance_matrix_df = distance_matrix_df.dropna()

    #Removing rows and columns corresponding to American cities because Canada on top
    canadian_distance_matrix_df = distance_matrix_df.loc[index_numbers, index_numbers]

    # Creating a dictionary to map indices to city names
    index_to_city_dict = dict(zip(canadian_distance_matrix_df.index, canadian_cities_df['City']))

    # Replacing indices and column names in city_distances_df
    canadian_city_distances_df = canadian_distance_matrix_df.rename(index=index_to_city_dict, columns=index_to_city_dict)
    
    return canadian_city_distances_df

def read_lat_long_coordinates(filepath, canadian_cities_df: pd.DataFrame, index_numbers : List[int]) -> pd.DataFrame:

    '''

    Reads a .txt file containing city name and returns a DataFrame.
    
    '''

    dms_coordinates_df = pd.read_csv(filepath,  sep = r'\s+', header = None, names = ["Latitude Degrees", "Latitude Minutes", "Latitude Seconds", "Latitude Cardinal", "Longitude Degrees",  "Longitude Minutes", "Longitude Seconds", "Longitude Cardinal"], comment = '#')

    filtered_dms_coordinates_df = dms_coordinates_df.loc[index_numbers]

    index_to_city_dict = dict(zip(filtered_dms_coordinates_df.index, canadian_cities_df['City']))

    canadian_dms_coordinates_df = filtered_dms_coordinates_df.rename(index=index_to_city_dict, columns=index_to_city_dict)

    return canadian_dms_coordinates_df

def dms_to_dd(degrees, minutes, seconds, hemisphere='N'):
    
    # Converting DMS to decimal
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    
    # Fliping sign based on hemisphere
    if hemisphere.upper() in ['S', 'W']:
        dd = -dd
    
    return dd

@app.command("process")
def process_all():
    
    """
    
    Read external files, filter Canadian cities, build distance/coords, and write CSVs.
    
    """

    # 1) Load and filter cities
    city_filepath = EXTERNAL_DATA_DIR / "city_names.txt"
    cities_df = read_cities(city_filepath)
    canadian_cities_df = get_canadian_cities(cities_df)
    canadian_cities_df.to_csv(INTERIM_DATA_DIR / "canadian_cities.csv", index=False)

    # 2) Extract indices of Canadian cities from full dataset
    canadian_city_indices = canadian_cities_df.index.tolist()

    # 3) Build distance matrix
    distance_filepath = EXTERNAL_DATA_DIR / "ctc_distance.txt"
    dist_matrix_df = read_city_to_cities_distance(distance_filepath, canadian_cities_df, canadian_city_indices)
    dist_matrix_df.to_csv(PROCESSED_DATA_DIR / "distance_matrix.csv")

    # 4) Read and convert coordinates
    dms_filepath = EXTERNAL_DATA_DIR / "dms_coordinates.txt"
    dms_df = read_lat_long_coordinates(dms_filepath, canadian_cities_df, canadian_city_indices)

    # Insert decimal degree columns
    dms_df.insert(loc=0, column="Latitude Decimal Degrees", value=0.0)
    dms_df.insert(loc=5, column="Longitude Decimal Degrees", value=0.0)
    dms_df["Latitude Decimal Degrees"] = dms_df.apply(
        lambda row: dms_to_dd(row["Latitude Degrees"], row["Latitude Minutes"], row["Latitude Seconds"], row["Latitude Cardinal"]), axis=1
    )
    dms_df["Longitude Decimal Degrees"] = dms_df.apply(
        lambda row: dms_to_dd(row["Longitude Degrees"], row["Longitude Minutes"], row["Longitude Seconds"], row["Longitude Cardinal"]), axis=1
    )
    
    dms_df.index = canadian_cities_df["City"].values 
    dms_df.to_csv(PROCESSED_DATA_DIR / "coords_dms.csv", index=True)

    typer.echo("All data processed and saved.")

if __name__ == "__main__":
    app()