#!/usr/bin/env python
# coding: utf-8

"""
This script performs the same function as notebook 
data_analysis/004-sat_and_nwp_norm_values.ipynb

This is to calculate useful aggregate statictics for the satellite and NWP data
so that it can be normalised, min-max scaled, or log transformed and then 
normalised or min-max scaled.
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask

import os
import gcsfs

import src
from src.data.constants import GCP_FS

# set up local path to save to
agg_dir = os.path.expanduser("~/agg_data")
os.makedirs(agg_dir, exist_ok=True)


def aggregates(ds, dim):
    """Function to calculate various aggregates to be used in pre-processing data."""
    agg = {}
    agg['mean'] = ds.mean(dim=dim)
    agg['std'] = ds.std(dim=dim)
    agg['min'] = ds.min(dim=dim)
    agg['max'] = ds.max(dim=dim)
    # need min for next section so do first round of computation now
    (agg['mean'],agg['std'], 
     agg['min'], agg['max']) = dask.compute(*[agg[k] for k in ['mean','std','min','max']])
    
    log_ds = np.log(ds - agg['min'] + 1)
    agg['mean_log'] = log_ds.mean(dim=dim)
    agg['std_log'] = log_ds.std(dim=dim)
    agg['min_log'] = log_ds.min(dim=dim)
    agg['max_log'] = log_ds.max(dim=dim)
    
    # combine and compute
    agg_ds = xr.concat([ds.assign_coords(aggregate_statistic=[k]) 
                            for k, ds in agg.items()], 
                       dim='aggregate_statistic').compute()
    agg_ds.attrs = ds.attrs
    agg_ds.attrs['log_calculation_note'] = "Calculated from log(x-x.min()+1)"
    return agg_ds


################################################################################
# Satellite aggrgate stats
print('Starting satellite aggregate statistics')

# At time of last running only June 2018 to Dec 2019 is available. So I just use
# 2019 here so that samples are even over season.
all_sat_channels = list(src.data.sat_loader.AVAILABLE_CHANNELS.index)
sat_ds = src.data.sat_loader.SatelliteLoader(channels=all_sat_channels).dataset
sat_ds = sat_ds.sel(time=slice('2019-01-01', '2019-12-31'))

sat_time_message="""
    Aggregate statistics (min,max, logmin etc) used for 
    preprocessing calculated over date range 2019-01-01 : 2019-12-31. Sample 
    every ~25mins
"""

sl = slice(None, None, 5)
sat_ds_filtered = sat_ds.isel(time=sl)
sat_aggs = aggregates(sat_ds_filtered, dim=('x', 'y', 'time'))
sat_aggs.attrs['time-range'] = sat_time_message


# save locally just to avoid losing the calculaion somehow
print('Saving satellite aggregate statistics locally')
sat_aggs.to_netcdf(f"{agg_dir}/sat_aggs.nc")

################################################################################
# NWP aggrgate stats
print('Starting NWP aggregate statistics')

# At time of last running this is over a 2 year period from Jan 2018 to Dec 2019
all_nwp_channels = list(src.data.nwp_loader.AVAILABLE_CHANNELS.index)
nwp_ds = src.data.nwp_loader.NWPLoader(channels=all_nwp_channels).dataset

# Don't need 37 hours per 3-hourly forecast. Take every n-th forecast and m-th valid time
nwp_time_message="""
    Aggregate statistics (min,max, logmin etc) used for 
    preprocessing calculated over date range 2018-01-01 : 2019-01-01. Samples 
    are one forecast every 9 hours and take forecast step every 2 hours.
"""

nwp_sl_time = slice(None, None, 3)
nwp_sl_step = slice(None, None, 2)

nwp_ds_filtered = nwp_ds.isel(time=nwp_sl_time, step=nwp_sl_step)

nwp_aggs = aggregates(nwp_ds_filtered, dim=('x', 'y', 'time', 'step'))
nwp_aggs.attrs['time-range'] = nwp_time_message

print('Saving NWP aggregate statistics')
nwp_aggs.to_netcdf(f"{agg_dir}/nwp_aggs.nc")

################################################################################
# Upload to zarr store

SATELLITE_AGG_PATH = 'solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/aggregate'
SATELLITE_AGG_STORE = gcsfs.mapping.GCSMap(SATELLITE_AGG_PATH, gcs=GCP_FS, check=True, create=True)
sat_aggs.to_zarr(store=SATELLITE_AGG_STORE, consolidated=True)

NWP_AGG_PATH = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/aggregate'
NWP_AGG_STORE = gcsfs.mapping.GCSMap(NWP_AGG_PATH, gcs=GCP_FS, check=True, create=True)
nwp_aggs.to_zarr(store=NWP_AGG_STORE, consolidated=True)






