
# CONDA ENV: gis_watershed_tools

# This script extracts the Livneh-Lusu dataset for the 671 basins in the CAMELS dataset.


import os
import sys
import glob
import io
import contextlib
# Create a dummy stream to capture stdout.
dummy_stream = io.StringIO()

# Import required libraries
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from whitebox.whitebox_tools import WhiteboxTools

from tqdm import tqdm
import tempfile


sys.path.append('../dem_getter')
from dem_getter.dem_getter import dem_getter



def process_watershed_flow_length(basin_shapefile, wbt_work_dir, path_flowlen_csv=None,
                                  path_figures=None, keep_dem_raster=False):

    """
    Process the watershed flow length for a given basin shapefile.
    Calculates the maximum flow length for each basin using WhiteboxTools.
    The calculated flow length represents the maximum distance water would flow from the furthest point
    in the basin to the outlet. This is useful for hydrological modeling and analysis.
    
    Parameters:
    - basin_shapefile (GeoDataFrame): The GeoDataFrame containing the basin shapefiles.
            The shapefile must contain a column 'hru_id' with unique basin identifiers and geometries.
            The geometries should be in WGS84 (EPSG:4326) coordinate system.
            As this script is designed to work with the CAMELS dataset, it assumes the hru_id is a unique 
            identifier for each basin, and it is zero-padded to 8 digits (e.g., '00000001', '00000002', etc.).
    - wbt_work_dir (str): The working directory for WhiteboxTools operations. This path must 
            be an absolute path where the temporary files will be stored. WhiteboxTools will not work if 
            a relative path is provided.
    - path_flowlen_csv (str): The path to the CSV file where the flow length results will be saved.
            If the file already exists, it will be overwritten. If not provided, the results will be saved 
            if the same directory as the script and the name of the file will be 'flow_length.csv'.
    - path_figures (str): The path to the directory where the figures will be saved. If not provided,
            the figures will be saved in a '../figures/DEM' and a '../figures/FLOWLEN' directories.

    Returns:
    flow_length (dataframe): A DataFrame containing the maximum flow length for each basin.
    The DataFrame will have the following columns:
    - 'hru_id': The unique identifier for each basin, zero-padded to 8 digits.
    - 'max_flow_length': The maximum flow length for the basin in meters.
    - 'max_elevation': The maximum elevation in the basin in meters.
    - 'min_elevation': The minimum elevation in the basin in meters.
    - 'mean_elevation': The mean elevation in the basin in meters.
    """

    if path_flowlen_csv is None:
        path_flowlen_csv = '../data/flow_length.csv'
    # Check if the output directory exists, if not create it
    output_dir = os.path.dirname(path_flowlen_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if path_figures is None:
        path_figures = '../figures'
    # Check if the figures directory exists, if not create it
    figures_dem_dir = os.path.join(path_figures, 'DEM')
    if not os.path.exists(figures_dem_dir):
        os.makedirs(figures_dem_dir)
    figures_flowlen_dir = os.path.join(path_figures, 'FLOWLEN')
    if not os.path.exists(figures_flowlen_dir):
        os.makedirs(figures_flowlen_dir)

    # Create a copy of the basin shapefile to calculate the basin area in an equal area projection
    basin_equal_area = basin_shapefile.copy()
    basin_equal_area = basin_equal_area.to_crs(epsg=6933)

    basin_skipped = []
    for index, basin in tqdm(basin_shapefile.iterrows(), 
                            total=len(basin_shapefile), 
                            desc='Processing basins', 
                            unit='basin'):
        
        # Area of the basin in square meters
        basin_area = basin_equal_area.loc[index, 'geometry'].area

        # Get the bounding box of the selected basin
        xMin, yMin, xMax, yMax = basin.geometry.bounds
        basin_id = str(basin_shapefile.loc[index,'hru_id']).zfill(8)   # Ensure the basin ID is zero-padded to 8 digits
        print(f'Processing basin {basin_id} with bounding box: {xMin}, {yMin}, {xMax}, {yMax}')

        watershed = gpd.GeoDataFrame({'id':[basin_id]}, 
                                    geometry=[basin['geometry']], 
                                    crs=basin_shapefile.crs)

        # Use dem_getter to download and clip the DEM for the basin defined by the bounding box 'bounds'
        try:
            paths = dem_getter.get_aws_paths(
                dataset='NED_1as',
                xMin=xMin,
                yMin=yMin,
                xMax=xMax,
                yMax=yMax,
                filePath=None,
                inputEPSG=4326,
                doExcludeRedundantData=True
            )
        except:
            print(f"Error retrieving DEM paths for basin {basin_id}. Skipping this basin.")
            basin_skipped.append(basin_id)
            continue

        dem_getter.batch_download(
            dlList=paths,
            folderName=wbt_work_dir,
            doForceDownload=True
        )

        # Collect DEM file paths
        dem_files = [
            os.path.join(wbt_work_dir, f)
            for f in os.listdir(wbt_work_dir) if f.endswith('.tif')
        ]

        # Project basin geometry to match raster CRS
        projected_basin = basin_shapefile.copy()

        # List for temporary clipped rasters
        clipped_tempfiles = []

        for file in dem_files:
            with rasterio.open(file) as src:
                # Ensure CRS matches
                if projected_basin.crs != src.crs:
                    projected_basin = projected_basin.to_crs(src.crs)
                
                try:
                    # Attempt to clip with basin polygon
                    clipped, clipped_transform = mask(
                        src,
                        [projected_basin.iloc[index].geometry],
                        crop=True
                    )

                    # Prepare metadata for each clipped tile
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": clipped.shape[1],
                        "width": clipped.shape[2],
                        "transform": clipped_transform
                    })

                    # Write clipped tile to a temporary file
                    temp_fp = tempfile.NamedTemporaryFile(
                        suffix=".tif", delete=False
                    ).name
                    with rasterio.open(temp_fp, "w", **out_meta) as dst:
                        dst.write(clipped)
                    clipped_tempfiles.append(temp_fp)

                except ValueError:
                    # Skip files with no overlap
                    continue

        # Raise an error if no tiles were usable
        if not clipped_tempfiles:
            raise RuntimeError("No DEM tiles intersect the basin.")

        # Reopen temp files for merging
        srcs_to_merge = [rasterio.open(f) for f in clipped_tempfiles]
        mosaic, out_transform = merge(srcs_to_merge)

        # Update metadata
        out_meta = srcs_to_merge[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform
        })

        # Write final merged DEM
        merged_clipped_dem = f'{wbt_work_dir}/dem_{basin_id}.tif'
        with rasterio.open(merged_clipped_dem, "w", **out_meta) as dest:
            dest.write(mosaic)

        # Clean up temporary files
        for src in srcs_to_merge:
            src.close()
        for f in clipped_tempfiles:
            os.remove(f)

        # Optionally clean up original DEM tiles too
        for f in dem_files:
            os.remove(f)

        # Plot the clipped DEM to check if it looks correct
        fig, ax = plt.subplots(figsize=(10, 10))
        mosaic = np.where(mosaic == out_meta['nodata'], np.nan, mosaic)  # Replace nodata with NaN
        plt.imshow(mosaic[0], cmap='terrain',
                    extent=(out_transform[2],
                                out_transform[2] + out_transform[0] * mosaic.shape[2],
                                out_transform[5] + out_transform[4] * mosaic.shape[1],
                                out_transform[5]),
                    aspect='auto')
        gpd.GeoSeries(projected_basin.iloc[index].geometry.boundary).plot(ax=ax, color='red', linewidth=1)
        plt.colorbar(label='Elevation (m)')
        plt.title(f'DEM for Basin {basin_id} ({basin_area/1e6:.1f} km²)')
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{figures_dem_dir}/dem_{basin_id}.png')
        #plt.show()
        plt.close()

        # Get the elevation statistics for the clipped DEM
        min_elevation = np.nanmin(mosaic)
        max_elevation = np.nanmax(mosaic)
        mean_elevation = np.nanmean(mosaic)


        # Project the clipped DEM to the WGS84 coordinate system (EPSG:6933) in meters
        # This is necessary for the WhiteboxTools operations that follow.
        with rasterio.open(merged_clipped_dem) as src:
            # Calculate the transform and metadata for the reprojected DEM
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:6933', src.width, src.height, *src.bounds)
            
            # Update the metadata for the projected DEM
            # Note: 'EPSG:6933' is the WGS84 coordinate system in meters, suitable for distance calculations.
            out_meta = src.meta.copy()
            out_meta.update({
                'crs': 'EPSG:6933',
                'transform': transform,
                'width': width,
                'height': height
            })

            # Create a new file for the reprojected DEM
            projected_dem_file = f'{wbt_work_dir}/dem_{basin_id}_clipped_projected.tif'
            
            with rasterio.open(projected_dem_file, 'w', **out_meta) as dest:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dest, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs='EPSG:6933',
                        resampling=Resampling.nearest
                        )


        # Set up the WhiteboxTools instance and set the working directory
        wbt = WhiteboxTools()
        wbt.set_working_dir(wbt_work_dir)

        with contextlib.redirect_stdout(dummy_stream):
            
            # Fill the depressions in the DEM to ensure that there are no sinks
            wbt.breach_depressions_least_cost(
                dem=f'dem_{basin_id}_clipped_projected.tif',
                output=f'dem_{basin_id}_filled.tif',
                dist=1000,  # Set the distance to fill depressions
            )
            
            wbt.max_upslope_flowpath_length(
                f'dem_{basin_id}_filled.tif',
                f'flow_length_{basin_id}.tif'
            )


        # Plot the flow length raster to check if it looks correct
        with rasterio.open(f'{wbt_work_dir}/flow_length_{basin_id}.tif') as src:
            flow_length = src.read(1)
            flow_length_without_nodata = np.where(flow_length == src.nodata, np.nan, flow_length)  # Replace nodata with NaN
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.imshow(flow_length_without_nodata, cmap='viridis',
                        extent=(src.transform[2],
                                src.transform[2] + src.transform[0] * src.width,
                                src.transform[5] + src.transform[4] * src.height,
                                src.transform[5]),
                        aspect='auto')
            gpd.GeoSeries(basin_equal_area.iloc[index].geometry.boundary).plot(ax=ax, color='red', linewidth=1)
            plt.colorbar(label='Flow Length (m)')
            plt.title(f'Flow Length for Basin {basin_id}')
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{figures_flowlen_dir}/flow_length_{basin_id}.png')
            #plt.show()
            plt.close()


            # Extract the maximum flow length
            with rasterio.open(f'{wbt_work_dir}/flow_length_{basin_id}.tif') as src:
                flow_length = src.read(1)
                max_flow_length = np.nanmax(flow_length)


            # Create a DataFrame to store the results
            flow_length_basin = pd.DataFrame({
                'hru_id': [basin_id],
                'max_flow_length': [max_flow_length],
                'max_elevation': [max_elevation],
                'min_elevation': [min_elevation],
                'mean_elevation': [mean_elevation],
                'area': [basin_area]  # Area in square meters
            })

        
        # Concatenate the results into a single DataFrame
        if index == 0:
            flow_length_df = flow_length_basin
        else:
            flow_length_df = pd.concat([flow_length_df, flow_length_basin], ignore_index=True)

        # Remove the temporary files created during the process
        for filepath in glob.glob(os.path.join(wbt_work_dir, '*')):
            if os.path.isfile(filepath):
                os.remove(filepath)
    

    # Write the results to a CSV file
    flow_length_df.to_csv(path_flowlen_csv, index=False)
    # Save the 'basin_skipped' list to a text file
    with open('../data/basins_skipped.txt', 'w') as f:
        for basin_id in basin_skipped:
            f.write(f"{basin_id}\n")

    
if __name__ == "__main__":

    # WHitebox working directory (looks like the absolute path is required)
    #wbt_work_dir = 'D:/17_TOVA/DPL_robustness/temporary_working_dir'

    # Load the shapefile of the 671 basins
    basin_shapefile_file = '../data/shapefiles/CAMELS/HCDN_nhru_final_671.shp'
    # Read the shapefile using geopandas
    basin_shapefile = gpd.read_file(basin_shapefile_file)
    # Convert crs to 6933 to equal area projection for area calculations
    # basin_equal_area = basin_shapefile.to_crs(epsg=6933)
    # Convert crs to 4326 (WSG84); required for dem_getter
    basin_shapefile = basin_shapefile.to_crs(epsg=4326)

    # Does require an absolute path
    # wbt_work_dir = 'D:/18_My_Python_packages/GIS_watershed_tools/temporary_working_dir' 

    # Process the watershed flow length for the basins
    flow_length_df = process_watershed_flow_length(
        basin_shapefile=basin_shapefile,
        wbt_work_dir=wbt_work_dir,
        keep_dem_raster=False
    )

