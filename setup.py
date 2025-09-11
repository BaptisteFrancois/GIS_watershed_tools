from setuptools import setup, find_packages

setup(
    name='GIS_watershed_tools',
    version='0.0.1',
    description='A set of tools for extracting features from shapefiles and DEM raster files',
    url='https://github.com/BaptisteFrancois/GIS_watershed_tools.git',
    author='BaptisteFrancois',
    author_email='BaptisteFrancois51@gmail.com',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'seaborn',
        'geopandas',
        'cdsapi',
        'xarray',
        'shapely',
        'rasterio',
        'netCDF4', 
        'whitebox'
    ],
    keywords=['python', 'GIS', 'watershed', 'shapefile', 'DEM', 'raster', 'geospatial'],
)


# The following package needs to be installed manually
# 'git+https://code.usgs.gov/gecsc/dem_getter.git@main#egg=dem_getter',