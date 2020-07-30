"""
Common constants for the loader functions
"""
import os
import gcsfs

# Data locations
GCP_PROJECT = "solar-pv-nowcasting"
GCP_FS = gcsfs.GCSFileSystem(project=GCP_PROJECT, token=None)

# Grid projection info of the satellite and NWPs
# - This is the natibve grid of the UKV NWP
# - The satellite data has been projected onto this grid
DST_CRS = 'EPSG:27700'
WEST=-239_000
SOUTH=-185_000
EAST=857_000
NORTH=1_223_000