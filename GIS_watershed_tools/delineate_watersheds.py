

import os
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from importlib.resources import path
from pysheds.grid import Grid
from pysheds.view import Raster

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import rasterio
from rasterio.merge import merge
from rasterio.features import shapes
from rasterio.transform import xy
import numpy as np
import matplotlib.pyplot as plt
import tarfile




def find_merit_tile(lon, lat, path, variable, temp_dir='temp_files'):
    """
    Find the correct MERIT Hydro tile for a given longitude and latitude.
    Returns the file path to the extracted raster file.
    - lon: longitude of the point
    - lat: latitude of the point
    - path: path to the MERIT Hydro data directory
    - variable: 'flow_accumulation' or 'flow_direction'
    - temp_dir: temporary directory to extract files to
    """

    # Placeholder function: implement logic to find the correct tile
    if variable == 'flow_accumulation':
        var = 'upa'
    elif variable == 'flow_direction':
        var = 'dir'
    else:
        raise ValueError('Variable not recognized')

    # The HYDRO Merit archive naming convention is {var}_{lat_orientation}{lat}{lon_orientation}{lon}.tif 
    # where lat and lon are integers that represent the lower-left corner of the tile. 
    # e.g., upa_n30w150.tar
    # One archive covers 30 degrees by 30 degrees.
    archive_lon = int(np.floor(lon / 30) * 30)
    archive_lat = int(np.floor(lat / 30) * 30)

    # Inside each archive, the file are named {lat_orientation}{lat}{lon_orientation}{lon}_{var}.tif
    # e.g., n30w120_upa.tif
    # One file covers 5 degrees by 5 degrees.
    file_lon = int(np.floor(lon / 5) * 5)
    file_lat = int(np.floor(lat / 5) * 5)

    if archive_lon >= 0:
        archive_lon_str = 'e{:03d}'.format(archive_lon)
        file_lon_str = 'e{:03d}'.format(file_lon)
    else:
        archive_lon_str = 'w{:03d}'.format(abs(archive_lon))
        file_lon_str = 'w{:03d}'.format(abs(file_lon))
    if archive_lat >= 0:
        archive_lat_str = 'n{:02d}'.format(archive_lat)
        file_lat_str = 'n{:02d}'.format(file_lat)
    else:
        archive_lat_str = 's{:02d}'.format(abs(archive_lat))
        file_lat_str = 's{:02d}'.format(abs(file_lat))

    archive_name = f'{var}_{archive_lat_str}{archive_lon_str}.tar'
    file_name = f'{file_lat_str}{file_lon_str}_{var}.tif'
    internal_path = f'{archive_name.replace(".tar", "")}/{file_name}'
    
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, file_name)

    if not os.path.exists(output_path):
        with tarfile.open(os.path.join(path, archive_name), 'r') as tar:
            member = tar.getmember(internal_path)
            tar.extract(member, path=temp_dir)

    return os.path.join(temp_dir, internal_path), (file_lat_str, file_lon_str)


def catchment_touches_edge(catch_mask, edge_buffer=8):
    """
    Returns dict of boolean tests and their reasons. If touching edge, returns the direction of
     the next tile to process and the indices of the edge touched.
    - catch_mask: 2D boolean numpy (True for catchment cells)
    - edge_buffer: number of pixels from border considered "near edge"
    """
    h, w = catch_mask.shape

    # Is within near-edge buffer
    if edge_buffer > 0:
        r0 = edge_buffer
        c0 = edge_buffer
        near_edge_mask = np.zeros_like(catch_mask, dtype=bool)
        near_edge_mask[:r0, :] = True
        near_edge_mask[-r0:, :] = True
        near_edge_mask[:, :c0] = True
        near_edge_mask[:, -c0:] = True
        if (catch_mask & near_edge_mask).any():
            # Determine which edge is touched
            next_tiles = []

            if (catch_mask[:r0, :] & near_edge_mask[:r0, :]).any():
                # Top edge
                next_tiles.append(('top', (r0, 0)))
            if (catch_mask[-r0:, :] & near_edge_mask[-r0:, :]).any():
                # Bottom edge
                next_tiles.append(('bottom', (-r0, 0)))
            if (catch_mask[:, :c0] & near_edge_mask[:, :c0]).any():
                # Left edge
                next_tiles.append(('left', (0, c0)))
            if (catch_mask[:, -c0:] & near_edge_mask[:, -c0:]).any():
                # Right edge
                next_tiles.append(('right', (0, -c0)))

            return {'touches': True, 'reason': f'within_{edge_buffer}_pixels_of_edge', 'next_tiles': next_tiles}

    return {'touches': False, 'reason': 'inside_tile_safely', 'next_tiles': None}


def merge_tiles(raster_ini, origin_ini, next_tiles, path, grid_shape, transform, variable='flow_direction'):
    """
    Merges multiple catchment masks into one.
    - raster_ini: initial raster file path
    - origin_ini: list of (lat_str, lon_str) tuples for the initial tile
    - next_tiles: list of list of tile identifiers (e.g., 'top', 'bottom', 'left', 'right') and their offsets
    - path: path to the MERIT Hydro data directory
    - grid_shape: shape of the raster grid (rows, cols)
    - transform: affine transform of the raster
    - variable: 'flow_direction' or 'flow_accumulation'
    Returns a single merged catchment mask.
    """
    # Load initial raster
    rasters = [raster_ini]
    # Keep track of origin tiles
    origins_new_tiles = origin_ini

    # Loop over next tiles to load
    for tile, (idx_edge, idy_edge) in next_tiles:

        # Determine which tile to load based on the direction and the lat/lon of the edge touched
        grid_x, grid_y = range(grid_shape[1]), range(grid_shape[0])
        lon_edge, lat_edge = xy(transform, grid_y[idx_edge], grid_x[idy_edge])

        # Load top tile and merge
        if tile == 'top':
            raster_file, (lat_str, lon_str) = find_merit_tile(lon_edge, lat_edge + 5, path, variable=variable)
            rasters.append(raster_file)
            origins_new_tiles.append((lat_str, lon_str))

        # Load bottom tile and merge
        elif tile == 'bottom':
            raster_file, (lat_str, lon_str) = find_merit_tile(lon_edge, lat_edge - 5, path, variable=variable)
            rasters.append(raster_file)
            origins_new_tiles.append((lat_str, lon_str))
        
        # Load left tile and merge
        elif tile == 'left':
            raster_file, (lat_str, lon_str) = find_merit_tile(lon_edge - 5, lat_edge, path, variable=variable)
            rasters.append(raster_file)
            origins_new_tiles.append((lat_str, lon_str))
        
        # Load right tile and merge
        elif tile == 'right':
            raster_file, (lat_str, lon_str) = find_merit_tile(lon_edge + 5, lat_edge, path, variable=variable)
            rasters.append(raster_file)
            origins_new_tiles.append((lat_str, lon_str))

    # Identify if the addition of new tiles will create hole(s) in the grid. If it does, add more tiles as needed.
    lats = [int(lat_str[1:]) * (1 if lat_str[0] == 'n' else -1) for (lat_str, lon_str) in origins_new_tiles]
    lons = [int(lon_str[1:]) * (1 if lon_str[0] == 'e' else -1) for (lat_str, lon_str) in origins_new_tiles]
    lat_uniq = set(lats)
    lon_uniq = set(lons)
    expected_lat_lon_combinations = set((lat, lon) for lat in lat_uniq for lon in lon_uniq)
    actual_lat_lon_combinations = set(zip(lats, lons))
    missing_combinations = expected_lat_lon_combinations - actual_lat_lon_combinations

    for (lat_missing, lon_missing) in missing_combinations:
        lat_str = 'n{:02d}'.format(abs(lat_missing)) if lat_missing >= 0 else 's{:02d}'.format(abs(lat_missing))
        lon_str = 'e{:03d}'.format(abs(lon_missing)) if lon_missing >= 0 else 'w{:03d}'.format(abs(lon_missing))
        raster_file, _ = find_merit_tile(lon_missing, lat_missing, path, variable=variable)
        rasters.append(raster_file)
        origins_new_tiles.append((lat_str, lon_str))

    # Merge rasters
    src_files_to_mosaic = [rasterio.open(r) for r in rasters]
    mosaic, out_transform = merge(src_files_to_mosaic)

    # Metadata
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        with rasterio.open(tmp.name, "w", **out_meta) as dest:
            dest.write(mosaic)

    # Close the opened files
    for src in src_files_to_mosaic:
        src.close()

    return tmp.name, origins_new_tiles



def main(path_outlets, 
         path_MERIT_Hydro,
         path_figures='../figures/',
         path_shapefiles='../data/shapefiles/ResOpsUS_catchments/',
         show_figures=False
    ):


    """Delineate watersheds for given outlet coordinates using MERIT Hydro data.
    - path_outlets: path to CSV file with outlet coordinates (columns 'LONG' and 'LAT')
    - path_MERIT_Hydro: path to the MERIT Hydro data directory. The directory should contain the 
        `.tar` files that can be found at https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/.
        The flow direction (dir_*.tar) and flow accumulation (upa_*.tar) files are needed for the 
        regions of interest.
    - path_figures: path to save output figures
    - path_shapefiles: path to save output shapefiles
    """

    # Create output directories if they don't exist
    os.makedirs(path_figures, exist_ok=True)
    os.makedirs(path_shapefiles, exist_ok=True)

    # Extract outlet coordinates from the CSV
    outlet_attrs = pd.read_csv(path_outlets)
    outlet_coords = [(lon, lat) for lon, lat in zip(outlet_attrs['LONG'], outlet_attrs['LAT'])]

    # Define the direction mapping for D8 flow direction (given in the MERIT Hydro dataset)
    dirmap = (
        64, # North
        128, # Northeast
        1, # East
        2, # Southeast
        4, # South
        8, # Southwest
        16, # West
        32 # Northwest
    )


    # Loop over outlet coordinates and delineate watersheds
    for i, (lon, lat) in enumerate(outlet_coords):

        # Find the archive `.tar` file where the flow accumulation is stored
        # The HYRO1k MERIT Hydro data is organized in tiles with 30 arc-second resolution
        # You need to implement a function to find the correct tile based on lon/lat
        # For simplicity, let's assume we have a function `find_merit_tile(lon, lat)` that returns the file path
        flow_acc_path, (lat_origin, lon_origin) = \
            find_merit_tile(lon, lat, path_MERIT_Hydro, variable='flow_accumulation')
        flow_dir_path, (lat_origin, lon_origin) = \
            find_merit_tile(lon, lat, path_MERIT_Hydro, variable='flow_direction')

        # Load flow direction and accumulation rasters
        grid = Grid.from_raster(flow_acc_path)
        acc = grid.read_raster(flow_acc_path, dirmap=dirmap)
        fdir = grid.read_raster(flow_dir_path, dirmap=dirmap)

        # Snap the point to the nearest high accumulation cell
        x_snap, y_snap = grid.snap_to_mask(acc > 1000, (lon, lat))
        
        # Delineate the catchment
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')

        outlet = catchment_touches_edge(catch, edge_buffer=8)

        # If `catchment_touches_edge` indicates touching, open the next tile(s) and merge catchments as needed
        k = 0
        path_dir_raster = flow_dir_path
        origin_tiles = [(lat_origin, lon_origin)]
        while outlet['touches']:

            next_tiles = outlet['next_tiles']

            # Merge the necessary tiles
            path_dir_raster, origin_tiles = merge_tiles(
                path_dir_raster,
                origin_tiles,
                next_tiles,
                path_MERIT_Hydro, 
                grid_shape=grid.shape,
                transform=grid.affine,
                variable='flow_direction'
            )

            # Recalculate catchment on merged raster
            grid = Grid.from_raster(path_dir_raster)
            merged_catch = grid.read_raster(path_dir_raster, dirmap=dirmap)
            catch = grid.catchment(
                x=x_snap, 
                y=y_snap, 
                fdir=merged_catch, 
                dirmap=dirmap, 
                xytype='coordinate')

            outlet = catchment_touches_edge(catch, edge_buffer=8)

            k += 1
            if k > 10:
                print(f'Exceeded maximum iterations for point ({lon}, {lat}). Skipping.')
                break


        # Extract shapes from the raster catchment mask
        shape_gen = shapes(catch.astype(np.uint8), mask=catch.astype(bool), transform=grid.affine)
        geometries = [shape(geom) for geom, value in shape_gen if value == 1]
        # Save the catchment as a shapefile
        gdf = gpd.GeoDataFrame(geometry=geometries, crs='EPSG:4326')
        gdf.to_file(f'{path_shapefiles}/DAMID{outlet_attrs["DAM_ID"][i]}_{outlet_attrs["DAM_NAME"][i].replace(" ", "_")}.shp')


        # Plot the catchment (x and y should be in the raster's coordinate reference system)
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.imshow(catch, cmap='Blues', extent=grid.extent)
        ax.plot(x_snap, y_snap, 'ro')  # Mark the snapped outlet point
        ax.set_title(f'{outlet_attrs["DAM_NAME"][i]} ({lon}, {lat})\nTouch Edge: {outlet["touches"]}, Reason: {outlet["reason"]}')
        fig.savefig(f'{path_figures}/DAMID_{outlet_attrs["DAM_ID"][i]}_{outlet_attrs["DAM_NAME"][i].replace(" ", "_")}.png')
        if show_figures:
            plt.show()
        plt.close()

        if outlet['touches']:
            print(f'Warning: Catchment for {outlet_attrs["DAM_NAME"][i]} may be incomplete due to edge touching.')
    
    # Clean up temporary files
    temp_dir = 'temp_files'
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            shutil.rmtree(os.path.join(temp_dir, f), ignore_errors=True)
        os.rmdir(temp_dir)

if __name__ == "__main__":
    
    path_outlets = 'D:/17_TOVA/DPL_ABCD-cc-robustness/data/ResOpsUS/attributes/reservoir_attributes.csv'
    path_MERIT_Hydro = '../data/MERIT_Hydro/'
    path_figures='../figures/'
    path_shapefiles='../data/shapefiles/ResOpsUS_catchments/'   
    show_figures = False 
    
    main(path_outlets, path_MERIT_Hydro, path_figures, path_shapefiles)

