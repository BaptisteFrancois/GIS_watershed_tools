# conda: gis_watershed

import os
import glob
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from scipy import ndimage
import math

# Utility: find MERIT tile path (user-provided function)
from delineate_watersheds import find_merit_tile  # keep as you had

# 3x3 Horn kernels for dz/dx and dz/dy
Kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=float) / 8.0
Ky = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]], dtype=float) / 8.0

def deg_to_m_per_deg(lat_deg):
    # 111132.954: baseline value (meters) for one degree of latitude near the equator, 
    # derived from the WGS84 ellipsoid parameters (semi-major axis and flattening).
    #  This is the dominant term.
    
    # −559.822 · cos(2φ): first latitude-dependent correction that accounts for the ellipsoidal shape
    #  (the fact that meridians converge slightly differently as latitude changes). 
    # It uses cos(2φ) because meridional radius variations are periodic with half the period of latitude.

    # +1.175 · cos(4φ): small higher-order correction to improve accuracy across latitudes
    # (period 90° in latitude).

    lat_rad = math.radians(lat_deg)
    meters_per_deg_lat = 111132.954 - 559.822 * math.cos(2*lat_rad) + 1.175 * math.cos(4*lat_rad)
    meters_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3*lat_rad)
    return meters_per_deg_lat, meters_per_deg_lon

def compute_slope_aspect(dem, transform, dem_crs_is_geographic=True, nodata=None):
    """
    dem: 2D ndarray with nan for nodata
    transform: affine transform from rasterio
    dem_crs_is_geographic: bool, True if dem CRS is geographic (degrees)
    returns slope_deg, aspect_deg arrays (same shape), nodata preserved as np.nan
    """
    px_x_deg = transform.a
    px_y_deg = -transform.e

    # convert degrees->meters if geographic CRS
    if dem_crs_is_geographic:
        h, w = dem.shape
        center_col = w // 2
        center_row = h // 2
        center_x = transform.c + (center_col + 0.5) * transform.a + (center_row + 0.5) * transform.b
        center_y = transform.f + (center_col + 0.5) * transform.d + (center_row + 0.5) * transform.e
        # center_x=center_lon, center_y=center_lat
        m_per_deg_lat, m_per_deg_lon = deg_to_m_per_deg(center_y)
        px_x = abs(px_x_deg) * m_per_deg_lon
        px_y = abs(px_y_deg) * m_per_deg_lat
    else:
        px_x = abs(px_x_deg)
        px_y = abs(px_y_deg)

    # mask nodata as nan
    arr = dem.copy()
    # replace masked values with nan already assumed
    # compute dz/dx and dz/dy using convolution; ignore nan by filling with local mean using ndimage
    nan_mask = np.isnan(arr)

    # Fill small nan holes to allow convolution; larger masks remain nan after restoring
    # use nearest or local mean: here we temporarily fill nan with 0 and account for kernel normalization
    arr_f = np.nan_to_num(arr, nan=0.0)

    # Create a weight array that is 1 where data present, 0 where nan; convolve same kernel to get normalization
    weight = (~nan_mask).astype(float)

    dzdx_num = ndimage.convolve(arr_f, Kx, mode='nearest')
    weight_x = ndimage.convolve(weight, np.abs(Kx), mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        dzdx = dzdx_num / weight_x
    dzdy_num = ndimage.convolve(arr_f, Ky, mode='nearest')
    weight_y = ndimage.convolve(weight, np.abs(Ky), mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        dzdy = dzdy_num / weight_y

    # Convert to slope in radians: slope = arctan(sqrt( (dzdx/px_x)^2 + (dzdy/px_y)^2 ))
    sx = dzdx / px_x
    sy = dzdy / px_y
    slope_rad = np.arctan(np.hypot(sx, sy))
    slope_deg = np.degrees(slope_rad)

    # Aspect: atan2(dzdy, -dzdx) per common definition, then convert to degrees clockwise from north
    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = np.degrees(aspect_rad)
    # convert range to 0-360 and clockwise-from-north
    aspect_deg = (90.0 - aspect_deg) % 360.0

    # restore nodata
    slope_deg[nan_mask] = np.nan
    aspect_deg[nan_mask] = np.nan

    return slope_deg, aspect_deg

def raster_stats_from_array(arr, mask=None):
    """
    arr: 2D array with np.nan for nodata
    mask: boolean mask True where valid (optional)
    Returns dict of stats (mean, median, std, min, max, pct10, pct90)
    """
    if mask is None:
        valid = ~np.isnan(arr)
    else:
        valid = mask & ~np.isnan(arr)
    if not np.any(valid):
        return {k: np.nan for k in ['mean','median','std','min','max','pct10','pct90']}
    vals = arr[valid].ravel()
    return {
        'mean': float(np.nanmean(vals)),
        'median': float(np.nanmedian(vals)),
        'std': float(np.nanstd(vals)),
        'min': float(np.nanmin(vals)),
        'max': float(np.nanmax(vals)),
        'pct10': float(np.nanpercentile(vals, 10)),
        'pct90': float(np.nanpercentile(vals, 90))
    }


def process_dem_from_basin_shapefile(paths):

    """Process DEMs for basins defined in shapefiles, compute static attributes,
    and save to CSV.
    """

    os.makedirs(paths['dem'], exist_ok=True)
    # Iterate shapefiles
    shapefiles = [f for f in os.listdir(paths['shapefiles']) if f.endswith('.shp')]
    rows = []


    for shp in tqdm(shapefiles, desc="Basins", unit="basin", ncols=80, total=len(shapefiles)):
        basin = gpd.read_file(os.path.join(paths['shapefiles'], shp))
        # ensure single geometry (dissolve / unary_union if multiple parts)
        basin = basin.to_crs(epsg=4326)  # use lat/lon to find tiles; change if your shapefiles differ

        # bounding box for MERIT tile selection (5x5 degree tiles)
        minx, miny, maxx, maxy = basin.total_bounds
        minx_tile = int(np.floor(minx / 5.0)) * 5
        miny_tile = int(np.floor(miny / 5.0)) * 5
        maxx_tile = int(np.floor(maxx / 5.0)) * 5
        maxy_tile = int(np.floor(maxy / 5.0)) * 5

        merit_tiles_coords = []
        for x in range(minx_tile, maxx_tile + 5, 5):
            for y in range(miny_tile, maxy_tile + 5, 5):
                merit_tiles_coords.append((x, y))

        # find tile file paths
        path_tiles = []
        for (lon, lat) in merit_tiles_coords:
            p, ok = find_merit_tile(lon, lat, paths['MERIT_Hydro'], variable='elevation')
            if ok and os.path.exists(p):
                path_tiles.append(p)
        if len(path_tiles) == 0:
            warnings.warn(f"No MERIT tiles found for {shp}, skipping")
            continue

        # open and merge rasters
        srcs = [rasterio.open(p) for p in path_tiles]
        mosaic, out_transform = merge(srcs)
        # mosaic shape = (bands, height, width) ; for DEM we expect a single band
        dem = mosaic[0, :, :].astype(float)

        # build a MemoryFile dataset for masking
        meta = srcs[0].meta.copy()
        meta.update({
            'driver': 'GTiff',
            'height': dem.shape[0],
            'width': dem.shape[1],
            'transform': out_transform,
            'count': 1
        })
        for s in srcs:
            s.close()

        # mask to basin geometry (use basin geometry in the DEM CRS)
        # reproject basin to DEM CRS if needed
        dem_crs = meta.get('crs')
        if dem_crs is None:
            raise RuntimeError("DEM CRS unknown")
        if basin.crs != dem_crs:
            basin = basin.to_crs(dem_crs)

        basin_geom = [feature["geometry"] for feature in basin.__geo_interface__['features']]
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(dem, 1)
                out_image, out_transform2 = mask(dataset=dataset, shapes=basin_geom, crop=True, nodata=np.nan)
                clipped = out_image[0].astype(float)

        # Save clipped DEM
        out_meta = meta.copy()
        out_meta.update({
            'height': clipped.shape[0],
            'width': clipped.shape[1],
            'transform': out_transform2,
            'nodata': np.nan
        })
        outpath = os.path.join(paths['dem'], f"clipped_dem_{os.path.splitext(shp)[0]}.tif")
        with rasterio.open(outpath, 'w', **out_meta) as dst:
            dst.write(np.where(np.isnan(clipped), out_meta['nodata'], clipped), 1)

        # compute stats for elevation
        elev_stats = raster_stats_from_array(clipped)

        # compute slope and aspect
        slope_deg, aspect_deg = compute_slope_aspect(clipped, 
                                                    out_transform2, 
                                                    dem_crs_is_geographic=basin.crs.is_geographic, 
                                                    nodata=np.nan)
        slope_stats = raster_stats_from_array(slope_deg)
        aspect_stats = raster_stats_from_array(aspect_deg)

        # compute watershed area in km2 using geometry area (reproject to equal-area if desired)
        # If basin crs is geographic (degrees), reproject to an equal-area projection for accurate area
        if basin.crs.is_geographic:
            # use EPSG:6933 (NSIDC EASE-Grid 2.0 Global) or other appropriate equal-area
            basin_area = basin.to_crs(epsg=6933).geometry.area.sum() / 1e6
        else:
            basin_area = basin.geometry.area.sum() / 1e6

        # collect results
        DAMID = os.path.splitext(shp)[0]
        row = {
            'DAMID': DAMID,
            'area_km2': float(basin_area),
            'lat_cen': float(basin.geometry.centroid.y.mean()),
            'lon_cen': float(basin.geometry.centroid.x.mean()),
            'mean_elevation_m': elev_stats['mean'],
            'median_elevation_m': elev_stats['median'],
            'std_elevation_m': elev_stats['std'],
            'min_elevation_m': elev_stats['min'],
            'max_elevation_m': elev_stats['max'],
            'elev_pct10': elev_stats['pct10'],
            'elev_pct90': elev_stats['pct90'],
            'slope_mean': slope_stats['mean'],
            'slope_median': slope_stats['median'],
            'slope_std': slope_stats['std'],
            'aspect_mean': aspect_stats['mean'],
            'aspect_median': aspect_stats['median'],
            'aspect_std': aspect_stats['std'],
            'clipped_dem_path': outpath
        }
        rows.append(row)
        print(f"Processed {shp}: area={basin_area:.2f} km2, elev_mean={elev_stats['mean']:.1f} m")

        # Delete the clipped DEM from memory
        del clipped, slope_deg, aspect_deg

        # Remove the clipped DEM file if not needed
        os.remove(outpath)

    # Save results dataframe
    df = pd.DataFrame(rows)
    df.to_csv(paths['out_csv'], index=False)
    print("Done. Results saved to", paths['out_csv'])

if __name__ == "__main__":

    # Paths (edit to your layout)
    

    paths = {
        'MERIT_Hydro': '../data/MERIT_Hydro/',
        'shapefiles': '../data/shapefiles/ResOpsUS_catchments/',
        'dem': '../data/DEM_clipped/',
        'out_csv': '../data/ResOpsUS_watershed_topo_attributes.csv'
    }
    process_dem_from_basin_shapefile(paths)