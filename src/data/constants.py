"""
Common constants for the loader functions
"""
import os
import gcsfs

# data locations
GCP_PROJECT = "solar-pv-nowcasting"
GCP_FS = gcsfs.GCSFileSystem(project=GCP_PROJECT, token=None)