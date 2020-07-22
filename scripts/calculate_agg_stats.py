#!/usr/bin/env python
# coding: utf-8

# In this notebook we calculate some values which will be useful in preprocessing the EUMETSAT and UKV data (min, max, mean, std etc) and upload them to the GCP bucket. 
# 
# Currently these aggregate statistic values are calculated over less than a full year, so ideally this will be rerun once more data is available.


from src.data.constants import GCP_FS
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask

import os
import gcsfs


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


agg_dir = os.path.expanduser("~/agg_data")
os.makedirs(agg_dir, exist_ok=True)


# ## Satellite aggrgate stats
# 
# Note this is only calculated from Jan, Feb and June 2019. See graph further below for times


#SATELLITE_ZARR_PATH = 'solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/zarr'
#SATELLITE_STORE = gcsfs.mapping.GCSMap(SATELLITE_ZARR_PATH, gcs=GCP_FS, 
#                                       check=True, create=False)

#sat_ds = xr.open_zarr(store=SATELLITE_STORE, consolidated=True)
#sat_time_message="""Dates from 2019 Jan, Feb and June. Sample every ~25mins"""
#sl = slice(None, None, 5)
#sat_ds_filtered = sat_ds.isel(time=sl)
#sat_aggs = aggregates(sat_ds_filtered, dim=('x', 'y', 'time'))
#sat_aggs.attrs['time-range'] = sat_time_message

# save locally just to avoid losing the calculaion somehow
#sat_aggs.to_netcdf(f"{agg_dir}/sat_aggs.nc")

# upload to zarr store
#SATELLITE_AGG_PATH = 'solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/aggregate'
#SATELLITE_AGG_STORE = gcsfs.mapping.GCSMap(SATELLITE_AGG_PATH, gcs=GCP_FS, check=True, create=True)
#sat_aggs.to_zarr(store=SATELLITE_AGG_STORE, consolidated=True)


# ## NWP aggrgate stats
# Note this is only calculated for 2019 for all the year

NWP_ZARR_PATH1 = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/2019_1-6'
NWP_STORE1 = gcsfs.mapping.GCSMap(NWP_ZARR_PATH1, gcs=GCP_FS, 
                                       check=True, create=False)

NWP_ZARR_PATH2 = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/2019_7-12'
NWP_STORE2 = gcsfs.mapping.GCSMap(NWP_ZARR_PATH2, gcs=GCP_FS, 
                                       check=True, create=False)

nwp_ds1 = xr.open_zarr(store=NWP_STORE1, consolidated=True)
nwp_ds2 = xr.open_zarr(store=NWP_STORE2, consolidated=True)
nwp_ds = xr.concat([nwp_ds1, nwp_ds2], dim='time')

# Don't need 37 hours per 3-hourly forecast. Take every n-th forecast and m-th valid time
nwp_time_message="""Dates from all 2019. Took one forecast every 9 hours and take forecast step every 2 hours"""

nwp_sl_time = slice(None, None, 3)
nwp_sl_step = slice(None, None, 2)

nwp_ds_filtered = nwp_ds.isel(time=nwp_sl_time, step=nwp_sl_step)

nwp_aggs = aggregates(nwp_ds_filtered, dim=('x', 'y', 'time', 'step'))
nwp_aggs.attrs['time-range'] = nwp_time_message
nwp_aggs.to_netcdf(f"{agg_dir}/nwp_aggs.nc")

# upload to zarr store
NWP_AGG_PATH = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr/aggregate'
NWP_AGG_STORE = gcsfs.mapping.GCSMap(NWP_AGG_PATH, gcs=GCP_FS, check=True, create=True)
nwp_aggs.to_zarr(store=NWP_AGG_STORE, consolidated=True)






