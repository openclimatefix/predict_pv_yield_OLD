"""
Common constants for the loader functions
"""
import os
import gcsfs

# data locations
GCP_PROJECT = "solar-pv-nowcasting"
GCP_FS = gcsfs.GCSFileSystem(project=GCP_PROJECT, token=None)

DST_CRS = 'EPSG:27700'
WEST=-239_000
SOUTH=-185_000
EAST=857_000
NORTH=1223_000