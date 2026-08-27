import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

def get_test_data(predictions_file):
    test_df_columns = ["user_id", "true_label"]
    pred_df_columns = ["user_id", "pred_1"]

    test_df = pd.read_csv(predictions_file)[test_df_columns]
    pred_df = pd.read_csv(predictions_file)[pred_df_columns]
    return test_df, pred_df

def prepare_locations_data(locations_file):
    # Load location data
    locs_df = pd.read_csv(locations_file)
    
    # Convert WKT geometry strings to Shapely geometries
    locs_df['geometry'] = locs_df['geometry'].apply(wkt.loads)
    
    # Convert to GeoDataFrame
    locs_gdf = gpd.GeoDataFrame(locs_df, geometry='geometry')
    
    # Set coordinate reference system (assuming WGS84)
    locs_gdf.crs = "EPSG:4326"
    
    # Convert to Web Mercator for better visualization with contextily
    locs_gdf = locs_gdf.to_crs("EPSG:3857")
    
    return locs_gdf

def get_switzerland_boundaries(geojson_path):
    """Load Switzerland boundary data from GeoJSON file."""
    try:
        # Load the GeoJSON file
        ch_boundaries = gpd.read_file(geojson_path)
        
        # Ensure it's in the correct CRS for the basemap
        if ch_boundaries.crs is None:
            ch_boundaries.crs = "EPSG:4326"
        
        # Convert to the same CRS as the other data (Web Mercator)
        ch_boundaries = ch_boundaries.to_crs("EPSG:3857")
        return ch_boundaries
    except Exception as e:
        print(f"Error loading Switzerland boundaries: {e}")
        return None

def visualize_trajectories(user_id, test_df, pred_df, locs_gdf, switzerland_gdf=None, output_dir=None, fig_size=(12, 10), fixed_extent=None):

    # Filter data for the specified user_id
    user_test = test_df[test_df['user_id'] == user_id]
    user_pred = pred_df[pred_df['user_id'] == user_id]
    
    if user_test.empty or user_pred.empty:
        print(f"No data found for user_id: {user_id}")
        return
    
    # Get true and predicted trajectories
    true_locations = locs_gdf[locs_gdf['location_id'].isin(user_test['true_label'])]
    pred_locations = locs_gdf[locs_gdf['location_id'].isin(user_pred['pred_1'])]
    
    # Create figure with consistent size
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot Switzerland boundary if available
    if switzerland_gdf is not None and not switzerland_gdf.empty:
        switzerland_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, zorder=1)
        
    # Plot true trajectory
    true_locations.plot(ax=ax, color='blue', markersize=50, alpha=0.7, label='True Trajectory')
    
    # Plot predicted trajectory
    pred_locations.plot(ax=ax, color='red', markersize=50, alpha=0.7, label='Predicted Trajectory')
    
    # Add basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        print(f"Could not add basemap: {e}")
        print("Make sure you have internet connection and contextily installed.")
    
    # Add connecting lines for true trajectory
    if len(true_locations) > 1:
        true_x = true_locations.geometry.x.values
        true_y = true_locations.geometry.y.values
        ax.plot(true_x, true_y, 'b-', linewidth=2, alpha=0.5)
        
        # Add arrows to indicate direction
        for i in range(len(true_x) - 1):
            plt.arrow(true_x[i], true_y[i], 
                     (true_x[i+1] - true_x[i]) * 0.9, 
                     (true_y[i+1] - true_y[i]) * 0.9,
                     head_width=3000, head_length=6000, 
                     fc='blue', ec='blue', alpha=0.6)
    
    # Add connecting lines for predicted trajectory
    if len(pred_locations) > 1:
        pred_x = pred_locations.geometry.x.values
        pred_y = pred_locations.geometry.y.values
        ax.plot(pred_x, pred_y, 'r-', linewidth=2, alpha=0.5)
        
        # Add arrows to indicate direction
        for i in range(len(pred_x) - 1):
            plt.arrow(pred_x[i], pred_y[i], 
                     (pred_x[i+1] - pred_x[i]) * 0.9, 
                     (pred_y[i+1] - pred_y[i]) * 0.9,
                     head_width=3000, head_length=6000, 
                     fc='red', ec='red', alpha=0.6)
    
    # Set fixed extent for consistent visualization if provided
    if fixed_extent:
        ax.set_xlim(fixed_extent[0], fixed_extent[2])
        ax.set_ylim(fixed_extent[1], fixed_extent[3])
    
    # Add title and legend
    plt.title(f"Trajectory Comparison for User {user_id}", fontsize=16)
    plt.legend(loc='upper right')
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Save figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/user_{user_id}_trajectory.png", dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_dir}/user_{user_id}_trajectory.png")
    
    # Show figure
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize user trajectories')
    parser.add_argument('--predictions', type=str, default='runs/dtepr_mhsa/dtepr_test_predictions.csv', help='Path to predictions CSV file')
    parser.add_argument('--locations', type=str, default='data/simulation/locs.csv', help='Path to locations CSV file')
    parser.add_argument('--user_id', type=int, default=700, help='Specific user ID to visualize')
    parser.add_argument('--num_users', type=int, default=5, help='Number of random users to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations', help='Output directory for saved visualizations')
    parser.add_argument('--switzerland_geojson', type=str, default='data/geospatial/switzerland.geojson', help='Path to Switzerland GeoJSON file')
    parser.add_argument('--fig_width', type=float, default=12.0, help='Figure width in inches')
    parser.add_argument('--fig_height', type=float, default=10.0, help='Figure height in inches')
    
    args = parser.parse_args()
    
    # Check if the files exist
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(args.locations):
        print(f"Error: Locations file not found: {args.locations}", file=sys.stderr)
        sys.exit(1)
    
    # Check for Switzerland GeoJSON
    switzerland_gdf = None
    if os.path.exists(args.switzerland_geojson):
        try:
            switzerland_gdf = get_switzerland_boundaries(args.switzerland_geojson)
            print(f"Successfully loaded Switzerland boundary from {args.switzerland_geojson}")
        except Exception as e:
            print(f"Warning: Could not load Switzerland boundary: {e}")
    else:
        print(f"Warning: Switzerland GeoJSON file not found at {args.switzerland_geojson}")
    
    try:
        # Load the data
        test_df, pred_df = get_test_data(args.predictions)
        locs_gdf = prepare_locations_data(args.locations)
        
        # Get the extent of all locations to ensure consistent visualization
        all_geometries = locs_gdf.geometry.values
        if len(all_geometries) > 0:
            # Calculate the bounding box of all geometries
            bounds = gpd.GeoSeries(all_geometries).total_bounds
            # Add some padding
            padding = 0.1  # 10% padding
            x_range = bounds[2] - bounds[0]
            y_range = bounds[3] - bounds[1]
            fixed_extent = (
                bounds[0] - x_range * padding,
                bounds[1] - y_range * padding,
                bounds[2] + x_range * padding,
                bounds[3] + y_range * padding
            )
        else:
            fixed_extent = None
        
        if args.user_id:
            # Visualize a specific user
            visualize_trajectories(
                args.user_id, 
                test_df, 
                pred_df, 
                locs_gdf, 
                switzerland_gdf, 
                args.output_dir,
                fig_size=(args.fig_width, args.fig_height),
                fixed_extent=fixed_extent
            )
        else:
            # Visualize random users
            unique_users = test_df['user_id'].unique()
            if len(unique_users) == 0:
                print("Error: No users found in the predictions file", file=sys.stderr)
                sys.exit(1)
                
            selected_users = np.random.choice(unique_users, min(args.num_users, len(unique_users)), replace=False)
            
            for user_id in tqdm(selected_users, desc="Visualizing user trajectories"):
                print(f"\nVisualizing trajectory for User {user_id}...")
                visualize_trajectories(
                    user_id, 
                    test_df, 
                    pred_df, 
                    locs_gdf, 
                    switzerland_gdf, 
                    args.output_dir,
                    fig_size=(args.fig_width, args.fig_height),
                    fixed_extent=fixed_extent
                )
                plt.close()  # Close the figure to free memory
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        print("\nFor help, run: python analysis/visualize_individuals.py --help", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
