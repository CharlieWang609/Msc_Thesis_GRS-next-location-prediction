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

def visualize_trajectories(user_id, test_df, pred_df, locs_gdf, output_dir=None):
  
    # Filter data for the specified user_id
    user_test = test_df[test_df['user_id'] == user_id]
    user_pred = pred_df[pred_df['user_id'] == user_id]
    
    if user_test.empty or user_pred.empty:
        print(f"No data found for user_id: {user_id}")
        return
    
    # Get true and predicted trajectories
    true_locations = locs_gdf[locs_gdf['location_id'].isin(user_test['true_label'])]
    pred_locations = locs_gdf[locs_gdf['location_id'].isin(user_pred['pred_1'])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot true trajectory
    true_locations.plot(ax=ax, color='blue', markersize=50, alpha=0.7, label='True Trajectory')
    
    # Plot predicted trajectory
    pred_locations.plot(ax=ax, color='red', markersize=50, alpha=0.7, label='Predicted Trajectory')
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

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
    
    # No ID labels as per request
    
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
    parser.add_argument('--predictions', type=str, default='runs/dtepr_mhsa/dtepr_test_predictions.csv',help='Path to predictions CSV file')
    parser.add_argument('--locations', type=str, default='data/simulation/locs.csv', help='Path to locations CSV file')
    parser.add_argument('--user_id', type=int, default=200,help='Specific user ID to visualize')
    parser.add_argument('--num_users', type=int, default=5, help='Number of random users to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations', help='Output directory for saved visualizations')
    
        
    args = parser.parse_args()
    
    # Check if predictions file is provided
    if not args.predictions:
        print("Error: --predictions argument is required", file=sys.stderr)
        sys.exit(1)
        
    # Check if the files exist
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(args.locations):
        print(f"Error: Locations file not found: {args.locations}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load the data
        test_df, pred_df = get_test_data(args.predictions)
        locs_gdf = prepare_locations_data(args.locations)
        
        if args.user_id:
            # Visualize a specific user
            visualize_trajectories(args.user_id, test_df, pred_df, locs_gdf, args.output_dir)
        else:
            # Visualize random users
            unique_users = test_df['user_id'].unique()
            if len(unique_users) == 0:
                print("Error: No users found in the predictions file", file=sys.stderr)
                sys.exit(1)
                
            selected_users = np.random.choice(unique_users, min(args.num_users, len(unique_users)), replace=False)
            
            for user_id in tqdm(selected_users, desc="Visualizing user trajectories"):
                print(f"\nVisualizing trajectory for User {user_id}...")
                visualize_trajectories(user_id, test_df, pred_df, locs_gdf, args.output_dir)
                plt.close()  # Close the figure to free memory
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        print("\nFor help, run: python analysis/visualize_predictions.py --help", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
