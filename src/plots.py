from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import folium
import pandas as pd
import json 

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

def visualize_map_with_points_and_paths(dms_coordinates_df, best_route, total_distance, label="Metaheuristic", color="red"):
    
    # Building a list of coordinates for all cities
    coordinates = []
    for city in dms_coordinates_df.index:
        lat = dms_coordinates_df.loc[city, 'Latitude Decimal Degrees']
        lon = dms_coordinates_df.loc[city, 'Longitude Decimal Degrees']
        coordinates.append((lat, lon))
    
    # Creating the Folium map
    folium_map = folium.Map(zoom_start=5, scrollWheelZoom=True)
    
    # Adding markers for each city
    for city in dms_coordinates_df.index:
        lat = dms_coordinates_df.loc[city, 'Latitude Decimal Degrees']
        lon = dms_coordinates_df.loc[city, 'Longitude Decimal Degrees']
        folium.Marker([lat, lon], popup=city).add_to(folium_map)
    
    # Fit map to all points
    if coordinates:
        folium_map.fit_bounds(coordinates)
    
    # Build polyline route coordinates
    route_coordinates = []
    for city in best_route:
        if city in dms_coordinates_df.index:
            lat = dms_coordinates_df.loc[city, 'Latitude Decimal Degrees']
            lon = dms_coordinates_df.loc[city, 'Longitude Decimal Degrees']
            route_coordinates.append((lat, lon))
        else:
            print(f"Warning: City '{city}' not found in DataFrame.")
    
    # Add the route to the map
    if route_coordinates:
        folium.PolyLine(locations=route_coordinates, color=color, weight=2, opacity=0.7, popup=label).add_to(folium_map)

    # Add legend
    legend_html = f'''
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 250px; height: 90px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color: white;
                 opacity: 0.8;
                 padding: 10px;">
     <b>{label} (Distance: {total_distance:.2f} km)</b>
     </div>
     '''
    folium_map.get_root().html.add_child(folium.Element(legend_html))

    return folium_map


def load_best_result(algorithm_name):
    path = PROCESSED_DATA_DIR / algorithm_name / "best_result.json"
    with open(path, "r") as f:
        result = json.load(f)
    return result["best_route"], result["best_distance"]

if __name__ == "__main__":
    filepath = PROCESSED_DATA_DIR / "coords_dms.csv"
    dms_coordinates_df = pd.read_csv(filepath, index_col=0)  # make sure city names are index

    # Load each algorithm's best result
    ga_best_route, ga_best_distance = load_best_result("genetic_algorithm")
    sa_best_route, sa_best_distance = load_best_result("simulated_annealing")
    ts_best_route, ts_best_distance = load_best_result("tabu_search")
    mh_best_route, mh_best_distance = load_best_result("metropolis_hastings")

    results = {
    "ga": {"route": ga_best_route, "distance": ga_best_distance},
    "sa": {"route": sa_best_route, "distance": sa_best_distance},
    "ts": {"route": ts_best_route, "distance": ts_best_distance},
    "mh": {"route": mh_best_route, "distance": mh_best_distance},
    }
    
    # Generate and save or display the map
    for i in ["ga", "sa", "ts", "mh"]:
        route = results[i]["route"]
        distance = results[i]["distance"]
        
        map_obj = visualize_map_with_points_and_paths(dms_coordinates_df, route, distance)
        output_path = FIGURES_DIR / f"{i}_tsp_routes_map.html"
        map_obj.save(str(output_path))