"""
Common constants for the loader functions
"""
import os

# data locations
GCP_PROJECT = "solar-pv-nowcasting"
LOCAL_DATA_DIRECTORY = os.path.expanduser('~/repos/predict_pv_yield/data')
GCP_FS = gcsfs.GCSFileSystem(project=GCP_PROJECT, token=None)


