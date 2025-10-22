
# ResOpsUS Catchment Shapefiles

This repository contains shapefiles (ResOpsUS_catchments.tar) representing catchments delineated around reservoirs in the ResOpsUS dataset. Each shapefile corresponds to a reservoir and outlines the upstream catchment area that drains into it.

Catchments were delineated using the GIS_watershed_tools package, which utilizes flow direction and accumulation rasters from the MERIT Hydro dataset. Outlet points from the ResOpsUS dataset were used to ensure accurate watershed representation.

These shapefiles are intended for hydrological modeling, water resource management, and environmental analysis related to the reservoirs in the ResOpsUS dataset.

# Licensing & Attribution

The MERIT Hydro dataset is licensed under the Open Database License (ODbL) v1.0, which permits commercial use. However, any derived data—such as these shapefiles—must be made publicly available under the same ODbL license.
In compliance with this requirement, the shapefiles generated from MERIT Hydro data are shared openly in this repository. Users must provide proper attribution to:

- MERIT Hydro dataset
- GIS_watershed_tools package

Please ensure you review and follow the full terms of the ODbL license when using these shapefiles.

# Disclaimer

Users should verify catchment boundaries before use. Accuracy may vary depending on the resolution of the underlying DEM and the delineation methods used. Errors may also arise from unidentified issues in the processing workflow.