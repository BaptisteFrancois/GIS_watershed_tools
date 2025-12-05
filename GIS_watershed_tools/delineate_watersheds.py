# conda: gis_watershed

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
    elif variable == 'elevation':
        var = 'elv'
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


def find_optimal_snap_point(grid, acc, fdir, lon, lat, target_area_km2, dirmap, 
                            area_tolerance=0.1, min_threshold=1, max_threshold=100000, 
                            max_iterations=50):
    """
    Find the optimal accumulation threshold to snap to, such that the resulting 
    catchment area matches the target area from GDW.
    
    Parameters:
    - grid: pysheds Grid object
    - acc: flow accumulation array
    - fdir: flow direction array
    - lon, lat: outlet coordinates
    - target_area_km2: target catchment area from GDW (km²)
    - dirmap: direction mapping
    - area_tolerance: acceptable relative error (e.g., 0.1 = 10%)
    - min_threshold: minimum accumulation threshold to try
    - max_threshold: maximum accumulation threshold to try
    - max_iterations: maximum number of iterations
    
    Returns:
    - x_snap, y_snap: snapped coordinates
    - threshold_used: the accumulation threshold that produced the best match
    - catch: the catchment mask
    - actual_area_km2: the actual area of the delineated catchment
    """
    
    #print(f"    [DEBUG] Entering find_optimal_snap_point for ({lon}, {lat}), target={target_area_km2:.2f} km²")
    
    # Validate inputs first
    if acc is None or fdir is None:
        print(f"    [ERROR] acc or fdir is None")
        raise ValueError("acc or fdir is None")
    
    if not np.any(np.isfinite(acc)):
        print(f"    [ERROR] acc contains no finite values")
        raise ValueError("acc contains no finite values")
    
    if not np.any(np.isfinite(fdir)):
        print(f"    [ERROR] fdir contains no finite values")
        raise ValueError("fdir contains no finite values")
    
    if target_area_km2 <= 0:
        print(f"    [ERROR] Invalid target area: {target_area_km2}")
        raise ValueError(f"Invalid target area: {target_area_km2}")
    
    # print(f"    [DEBUG] Input validation passed")
    
    # For MERIT Hydro, flow accumulation (upa) is already in km² of upstream area
    # So we can use the target area directly as the threshold estimate
    estimated_threshold = target_area_km2
    
    # Start with estimated threshold
    initial_threshold = max(min_threshold, min(max_threshold, estimated_threshold))
    best_threshold = initial_threshold
    best_area = None
    best_error = float('inf')
    best_catch = None
    best_snap = None
    
    # print(f"    [DEBUG] Initial threshold: {initial_threshold:.0f} (estimated from target area)")
    
    # Try the estimated threshold first
    try:
        # print(f"    [DEBUG] Attempting snap with initial threshold {initial_threshold:.0f}...")
        
        # Check if any cells meet the threshold BEFORE calling snap_to_mask
        mask = acc > initial_threshold
        if not np.any(mask):
            print(f"    [DEBUG] No cells meet initial threshold {initial_threshold:.0f}, skipping initial attempt")
            raise ValueError("No cells meet threshold")
        
        # print(f"    [DEBUG] Calling snap_to_mask...")
        x_snap, y_snap = grid.snap_to_mask(mask, (lon, lat))
        # print(f"    [DEBUG] Snapped to ({x_snap}, {y_snap})")
        
        # print(f"    [DEBUG] Calling grid.catchment...")
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
        # print(f"    [DEBUG] Catchment delineated")
        
        # Validate catchment result
        if catch is None:
            # print(f"    [DEBUG] Catchment is None")
            raise ValueError("Catchment is None")
        
        if not np.any(catch):
            # print(f"    [DEBUG] Catchment is empty")
            raise ValueError("Catchment is empty")
        
        # print(f"    [DEBUG] Converting catchment to shapes...")
        shape_gen = shapes(catch.astype(np.uint8), mask=catch.astype(bool), transform=grid.affine)
        geometries = [shape(geom) for geom, value in shape_gen if value == 1]
        
        if len(geometries) > 0:
            # print(f"    [DEBUG] Found {len(geometries)} geometries, calculating area...")
            gdf = gpd.GeoDataFrame(geometry=geometries, crs='EPSG:4326')
            actual_area_km2 = gdf.to_crs(epsg=6933).geometry.area.sum() / 1e6
            relative_error = abs(actual_area_km2 - target_area_km2) / target_area_km2 if target_area_km2 > 0 else float('inf')
            best_error = relative_error
            best_area = actual_area_km2
            best_catch = catch
            best_snap = (x_snap, y_snap)
            best_threshold = initial_threshold
            
            # print(f"    [DEBUG] Initial attempt: area={actual_area_km2:.2f} km², error={relative_error*100:.1f}%")
            
            # If we're already within tolerance, return early
            if relative_error <= area_tolerance:
                # print(f"    [DEBUG] Within tolerance, returning early")
                return x_snap, y_snap, initial_threshold, catch, actual_area_km2
            
            # Set intelligent search bounds based on the initial result
            # Add safety factor of 50% to be conservative
            safety_factor = 1.5
            
            if actual_area_km2 < target_area_km2:
                # Underestimating - need LOWER threshold (to include more area)
                # Current threshold is too high, set as upper bound
                # Estimate how much lower to go based on area ratio
                area_ratio = target_area_km2 / actual_area_km2
                estimated_new_threshold = initial_threshold / area_ratio
                
                low_threshold = max(min_threshold, estimated_new_threshold / safety_factor)
                high_threshold = initial_threshold
                #  print(f"    [DEBUG] Underestimating: search range [{low_threshold:.0f}, {high_threshold:.0f}]")
            else:
                # Overestimating - need HIGHER threshold (to include less area)
                # Current threshold is too low, set as lower bound
                area_ratio = actual_area_km2 / target_area_km2
                estimated_new_threshold = initial_threshold * area_ratio
                
                low_threshold = initial_threshold
                high_threshold = min(max_threshold, estimated_new_threshold * safety_factor)
                # print(f"    [DEBUG] Overestimating: search range [{low_threshold:.0f}, {high_threshold:.0f}]")
                
    except Exception as e:
        # print(f"    [DEBUG] Initial attempt failed: {type(e).__name__}: {e}")
        # If initial estimate fails, use default wide search bounds
        low_threshold = min_threshold
        high_threshold = max_threshold
        # print(f"    [DEBUG] Using default search range [{low_threshold:.0f}, {high_threshold:.0f}]")
    
    # print(f"    [DEBUG] Starting binary search...")
    # Continue with binary search
    for iteration in range(max_iterations):
        # Try middle threshold
        threshold = (low_threshold + high_threshold) / 2
        
        # print(f"    [DEBUG] Iteration {iteration}: trying threshold {threshold:.0f}")
        
        # Check if any cells meet this threshold
        mask = acc > threshold
        if not np.any(mask):
            # print(f"    [DEBUG] No cells meet threshold {threshold:.0f}")
            high_threshold = threshold
            continue
        
        # Snap to accumulation cells above threshold
        try:
            x_snap, y_snap = grid.snap_to_mask(mask, (lon, lat))
        except (ValueError, IndexError, RuntimeError) as e:
            # If no cells found, try lower threshold
            # print(f"    [DEBUG] snap_to_mask failed: {e}")
            high_threshold = threshold
            continue
        
        # Delineate catchment
        try:
            catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
            
            # Validate catchment
            if catch is None or not np.any(catch):
                # print(f"    [DEBUG] Empty/None catchment at threshold {threshold:.0f}")
                high_threshold = threshold
                continue
            
            # Calculate area
            shape_gen = shapes(catch.astype(np.uint8), mask=catch.astype(bool), transform=grid.affine)
            geometries = [shape(geom) for geom, value in shape_gen if value == 1]
            if len(geometries) == 0:
                # No catchment found, try lower threshold
                high_threshold = threshold
                continue
                
            gdf = gpd.GeoDataFrame(geometry=geometries, crs='EPSG:4326')
            actual_area_km2 = gdf.to_crs(epsg=6933).geometry.area.sum() / 1e6
            
            # Calculate relative error
            if target_area_km2 > 0:
                relative_error = abs(actual_area_km2 - target_area_km2) / target_area_km2
            else:
                relative_error = float('inf')
            
            # print(f"    [DEBUG] Iteration {iteration}: area={actual_area_km2:.2f} km², error={relative_error*100:.1f}%")
            
            # Check if this is the best match so far
            if relative_error < best_error:
                best_error = relative_error
                best_threshold = threshold
                best_area = actual_area_km2
                best_catch = catch
                best_snap = (x_snap, y_snap)
            
            # Check if we're within tolerance
            if relative_error <= area_tolerance:
                # print(f"    [DEBUG] Within tolerance at iteration {iteration}")
                return x_snap, y_snap, threshold, catch, actual_area_km2
            
            # Adjust search range
            if actual_area_km2 < target_area_km2:
                # Area too small, need LOWER threshold (more inclusive)
                high_threshold = threshold
            else:
                # Area too large, need HIGHER threshold (less inclusive)
                low_threshold = threshold
                
            # Check convergence
            if high_threshold - low_threshold < 1:
                # print(f"    [DEBUG] Converged at iteration {iteration}")
                break
                
        except Exception as e:
            # If catchment delineation fails, try lower threshold
            # print(f"    [DEBUG] Catchment delineation failed: {type(e).__name__}: {e}")
            high_threshold = threshold
            continue
    
    # Return best match found
    if best_catch is not None:
        # print(f"    [DEBUG] Returning best match: threshold={best_threshold:.0f}, area={best_area:.2f} km²")
        return best_snap[0], best_snap[1], best_threshold, best_catch, best_area
    else:
        # Fallback to original method with low threshold
        # print(f"    [DEBUG] No good match found, using fallback with min_threshold={min_threshold}")
        
        mask = acc > min_threshold
        if not np.any(mask):
            # print(f"    [ERROR] No cells meet minimum threshold")
            raise ValueError("No cells meet minimum threshold")
            
        x_snap, y_snap = grid.snap_to_mask(mask, (lon, lat))
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
        shape_gen = shapes(catch.astype(np.uint8), mask=catch.astype(bool), transform=grid.affine)
        geometries = [shape(geom) for geom, value in shape_gen if value == 1]
        if len(geometries) > 0:
            gdf = gpd.GeoDataFrame(geometry=geometries, crs='EPSG:4326')
            actual_area_km2 = gdf.to_crs(epsg=6933).geometry.area.sum() / 1e6
        else:
            actual_area_km2 = 0
        return x_snap, y_snap, min_threshold, catch, actual_area_km2



def get_gdw_catchment_area(dam_id, gdw_dict):
    """
    Get catchment area from GDW database using DAM_ID (which matches GRAND_ID).
    
    Parameters:
    - dam_id: DAM_ID from reservoir attributes CSV (matches GRAND_ID in GDW)
    - gdw_dict: dictionary mapping GRAND_ID to CATCH_SKM
    
    Returns:
    - catch_area_km2: catchment area from GDW, or None if not found
    """
    if dam_id in gdw_dict:
        catch_area = gdw_dict[dam_id]
        # Handle missing values
        if pd.isna(catch_area) or catch_area <= 0:
            return None
        return float(catch_area)
    return None


def main(path_outlets, 
         path_MERIT_Hydro,
         path_figures='../figures/',
         path_shapefiles='../data/shapefiles/ResOpsUS_catchments/',
         show_figures=False,
         path_GDW=None,
         area_tolerance=0.1,
         use_gdw_area=True
    ):

    """Delineate watersheds for given outlet coordinates using MERIT Hydro data.
    - path_outlets: path to CSV file with outlet coordinates (columns 'LONG' and 'LAT')
    - path_MERIT_Hydro: path to the MERIT Hydro data directory. The directory should contain the 
        `.tar` files that can be found at https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/.
        The flow direction (dir_*.tar) and flow accumulation (upa_*.tar) files are needed for the 
        regions of interest.
    - path_figures: path to save output figures
    - path_shapefiles: path to save output shapefiles
    - path_GDW: path to GDW reservoirs shapefile (optional, required if use_gdw_area=True)
    - area_tolerance: acceptable relative error for catchment area matching (default 0.1 = 10%)
    - use_gdw_area: if True, use GDW catchment areas to optimize snapping (default True)
    """

    # Create output directories if they don't exist
    os.makedirs(path_figures, exist_ok=True)
    os.makedirs(path_shapefiles, exist_ok=True)

    # Extract outlet coordinates from the CSV
    outlet_attrs = pd.read_csv(path_outlets)
    outlet_coords = [(lon, lat) for lon, lat in zip(outlet_attrs['LONG'], outlet_attrs['LAT'])]

    # Load GDW data if provided and create ID-based lookup dictionary
    gdw_dict = {}
    if use_gdw_area and path_GDW is not None:
        if os.path.exists(path_GDW):
            gdw_gdf = gpd.read_file(path_GDW)
            # Create dictionary mapping GRAND_ID to CATCH_SKM
            gdw_dict = dict(zip(gdw_gdf['GRAND_ID'], gdw_gdf['CATCH_SKM']))
            print(f"Loaded {len(gdw_dict)} reservoirs from GDW database")
        else:
            print(f"Warning: GDW shapefile not found at {path_GDW}. Proceeding without GDW area matching.")
            use_gdw_area = False

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


    area_df = pd.DataFrame(columns=['DAM_ID', 'DAM_NAME', 'GDW_AREA_KM2', 'Delineated_AREA_KM2'])
    area_df['DAM_ID'] = outlet_attrs['DAM_ID']
    area_df['DAM_NAME'] = outlet_attrs['DAM_NAME']
    area_df['GDW_AREA_KM2'] = None
    area_df['Delineated_AREA_KM2'] = None
    area_df['Relative_Error'] = None


    # Loop over outlet coordinates and delineate watersheds
    for i, (lon, lat) in enumerate(outlet_coords):
        
        try:
            # Get target catchment area from GDW if available (using DAM_ID which matches GRAND_ID)
            target_area_km2 = None
            if use_gdw_area and gdw_dict:
                dam_id = outlet_attrs['DAM_ID'][i]
                target_area_km2 = get_gdw_catchment_area(dam_id, gdw_dict)
                if target_area_km2 is not None:
                    print(f"Reservoir {i+1}/{len(outlet_coords)} (DAM_ID: {dam_id}): Target area from GDW = {target_area_km2:.2f} km²", end='')
                    area_df.loc[i, 'GDW_AREA_KM2'] = target_area_km2
                else:
                    print(f"Reservoir {i+1}/{len(outlet_coords)} (DAM_ID: {dam_id}): No GDW match found, using default threshold", end='')
                    area_df.loc[i, 'GDW_AREA_KM2'] = None
            
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
            if use_gdw_area and target_area_km2 is not None:
                # Find cell with accumulation matching target area
                #x_snap, y_snap = find_snap_point_by_target_area(
                #    grid, acc, lon, lat, target_area_km2, search_radius_km=25
                #)
                x_snap, y_snap = grid.snap_to_mask(acc > target_area_km2, (lon, lat))
            else:
                # Use original method with low threshold
                x_snap, y_snap = grid.snap_to_mask(acc > 1, (lon, lat))
            
            # Delineate the catchment
            catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
            
            outlet = catchment_touches_edge(catch, edge_buffer=8)
            
            # If catchment touches edge, merge tiles and re-delineate
            # (existing tile merging code stays the same)
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
                
                # Re-snap if we had a target area
                if use_gdw_area and target_area_km2 is not None:
                    # Need to reload accumulation for merged area
                    # For now, use the original snapped point
                    catch = grid.catchment(
                        x=x_snap, 
                        y=y_snap, 
                        fdir=merged_catch, 
                        dirmap=dirmap, 
                        xytype='coordinate')
                else:
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

            # Print the number of geometries and their areas    
            #print(f'   [DEBUG] Number of geometries: {len(gdf)}')
            #print(f'   [DEBUG] Areas of geometries: {gdf.geometry.to_crs(epsg=6933).area / 10**6}')

            # Check if there are multiple geometries and dissolve them into one if so
            if len(gdf) > 1:
                # Calculate area for each geometry
                areas = gdf.geometry.to_crs(epsg=6933).area / 10**6  # Convert from m² to km²
                
                # Keep only the largest geometry
                idx_to_keep = areas.idxmax()
                largest_area = areas.iloc[idx_to_keep]
                
                # Check if there are other large geometries (> 10% of the largest)
                other_large = areas[areas > largest_area * 0.1]
                
                if len(other_large) > 1:
                    # Multiple large geometries found
                    print(f'Warning: Multiple large geometries found for reservoir {i+1} (DAM_ID: {outlet_attrs["DAM_ID"][i]}, {outlet_attrs["DAM_NAME"][i]}). Keeping largest ({largest_area:.2f} km²).')
                
                # Always keep the largest geometry
                gdf = gdf.loc[[idx_to_keep]]

            gdf.to_file(f'{path_shapefiles}/DAMID_{outlet_attrs["DAM_ID"][i]}_{outlet_attrs["DAM_NAME"][i].replace(" ", "_")}.shp')


            # Calculate final area for reporting
            final_area_km2 = gdf.to_crs(epsg=6933).geometry.area.sum() / 1e6
            area_df.loc[i, 'Delineated_AREA_KM2'] = final_area_km2
            
            # Calculate error percentage
            if use_gdw_area and target_area_km2 is not None:
                error_pct = abs(final_area_km2 - target_area_km2) / target_area_km2 * 100
                print(f"  Delineated area: {final_area_km2:.2f} km² (Error: {error_pct:.1f}%)")
            else:
                print(f"  Delineated area: {final_area_km2:.2f} km²")

            # Plot the catchment (x and y should be in the raster's coordinate reference system)
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.imshow(catch, cmap='Blues', extent=grid.extent)
            ax.plot(x_snap, y_snap, 'ro')  # Mark the snapped outlet point
            title = f'{outlet_attrs["DAM_NAME"][i]} ({lon}, {lat})\n'
            title += f'Area: {final_area_km2:.2f} km²'
            if use_gdw_area and target_area_km2 is not None:
                title += f' (Target: {target_area_km2:.2f} km²)'
            title += f'\nTouch Edge: {outlet["touches"]}, Reason: {outlet["reason"]}'
            ax.set_title(title)
            fig.savefig(f'{path_figures}/DAMID_{outlet_attrs["DAM_ID"][i]}_{outlet_attrs["DAM_NAME"][i].replace(" ", "_")}.png')
            if show_figures:
                plt.show()
            plt.close()

            if outlet['touches']:
                print(f'Warning: Catchment for {outlet_attrs["DAM_NAME"][i]} may be incomplete due to edge touching.')
            
            # Force garbage collection periodically to prevent memory issues
            if (i + 1) % 10 == 0:
                import gc
                gc.collect()
                print(f"Processed {i+1} reservoirs, memory cleaned")

        
        except KeyboardInterrupt:
            print(f'\n\nInterrupted by user at reservoir {i+1}/{len(outlet_coords)}')
            print('Saving progress...')
            # Could save a checkpoint here if needed
            raise  # Re-raise to allow clean exit
        
        except MemoryError as e:
            dam_id = outlet_attrs.get('DAM_ID', [None])[i] if i < len(outlet_attrs) else None
            dam_name = outlet_attrs.get('DAM_NAME', ['Unknown'])[i] if i < len(outlet_attrs) else 'Unknown'
            print(f'\n\nMEMORY ERROR: Out of memory at reservoir {i+1}/{len(outlet_coords)} (DAM_ID: {dam_id}, {dam_name})')
            print('This may be due to large raster operations. Try processing in smaller batches.')
            import gc
            gc.collect()
            raise  # Re-raise as this is a critical error
        
        except SystemExit:
            print(f'\n\nSystem exit requested at reservoir {i+1}/{len(outlet_coords)}')
            raise  # Re-raise to allow clean exit
        
        except BaseException as e:
            # Catch everything else, including C extension crashes if possible
            dam_id = outlet_attrs.get('DAM_ID', [None])[i] if i < len(outlet_attrs) else None
            dam_name = outlet_attrs.get('DAM_NAME', ['Unknown'])[i] if i < len(outlet_attrs) else 'Unknown'
            print(f'\n\nCRITICAL ERROR: Failed to process reservoir {i+1}/{len(outlet_coords)} (DAM_ID: {dam_id}, {dam_name}) at ({lon}, {lat})')
            print(f'Error type: {type(e).__name__}')
            print(f'Error message: {str(e)}')
            import traceback
            traceback.print_exc()
            print('\nAttempting to continue to next reservoir...\n')
            
            # Try to clean up any open files/resources
            try:
                import gc
                gc.collect()
            except:
                pass
            
            # For truly fatal errors, we might want to stop, but try to continue
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
            continue  # Continue to next reservoir

    area_df['Relative_Error'] = (area_df['Delineated_AREA_KM2'] - area_df['GDW_AREA_KM2']) / area_df['GDW_AREA_KM2']
    area_df.to_csv(f'{path_shapefiles}/area_comparison_GDW_delineated.csv', index=False)

    # Clean up temporary files (with error handling)
    try:
        temp_dir = 'temp_files'
        if os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                try:
                    shutil.rmtree(os.path.join(temp_dir, f), ignore_errors=True)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
    except Exception as e:
        print(f'Warning: Error during cleanup: {e}')

if __name__ == "__main__":

    path_outlets = 'D:/17_TOVA/DPL_ABCD-cc-robustness/data/ResOpsUS/attributes/reservoir_attributes.csv'
    path_MERIT_Hydro = '../data/MERIT_Hydro/'
    path_figures='../figures/'
    path_shapefiles='../data/shapefiles/ResOpsUS_catchments/'   
    path_GDW = '../data/GDW_v1_0_shp/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp'
    show_figures = False 
    area_tolerance = 0.05  # 10% error tolerance

    main(path_outlets, path_MERIT_Hydro, path_figures, path_shapefiles, 
         show_figures=show_figures, path_GDW=path_GDW, area_tolerance=area_tolerance)

