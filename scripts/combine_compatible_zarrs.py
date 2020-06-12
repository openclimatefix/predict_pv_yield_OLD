"""assume we have x, y, time, isobar.
This only combines in time."""

import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
import zarr
import subprocess
import os
import gcsfs
import json
import glob
import shutil

################################################################################
# user defined
################################################################################

masterzarr = os.path.expanduser("~/tmp/newzarr")
stagingdir = os.path.expanduser("~/staging")

os.makedirs(masterzarr, exist_ok=True)
os.makedirs(stagingdir, exist_ok=True)

zarrfiles = [os.path.expanduser(f"~/tmp/{f}") for f in 
             ['myzarr1', 'myzarr2', 'myzarr3']]

# here I download data as an example
# bucket and save path
fs = gcsfs.GCSFileSystem(project='solar-pv-nowcasting', token=None)
gcssavepath = 'solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_zarr'
gcsmap = gcsfs.mapping.GCSMap(gcssavepath, gcs=fs, check=False, create=False)

ds = xr.open_zarr(store=gcsmap)
ds = ds[['t', 'lcc']]
ds.isel(x=slice(0,10), y=slice(0, 10), time=slice(5, 10)).to_zarr(zarrfiles[0], mode='w')
ds.isel(x=slice(0,10), y=slice(0, 10), time=slice(10, 20)).to_zarr(zarrfiles[1], mode='w')
ds.isel(x=slice(0,10), y=slice(0, 10), time=slice(0, 5)).to_zarr(zarrfiles[2], mode='w')
del ds


################################################################################
# derived
################################################################################

all_ds = [xr.open_zarr(z) for z in zarrfiles]
combined_ds = xr.merge(all_ds).sortby('time')
time_shape = combined_ds.time.shape[0]
coordzarr = os.path.join(stagingdir, 'coordszarr')

################################################################################
# function library
################################################################################

def decode_command(command_list, split=True):
    out = subprocess.Popen(command_list,
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.STDOUT)
    stdout, _ = out.communicate()
    result = stdout.decode()
    if split:
        result = result.split('\n')
        result = [r for r in result if r!='']
    return result


def set_up_empty_zarr():
    
    # copy top level attributes
    print('setting up')
    os.system(f"cp {zarrfiles[0]}/.z* {masterzarr}/")
    
    subdirs = glob.glob(f"{zarrfiles[0]}/*")
    
    # copy and update data level atributes
    for sd in subdirs:
        sd = sd.split('/')[-1]
        inpath = os.path.join(zarrfiles[0], sd)
        outpath = os.path.join(masterzarr, sd)
        os.system(f"mkdir {outpath}")
        
        with open(f"{inpath}/.zarray", 'r') as json_file:
            data = json.load(json_file)
            data['shape'][0]=time_shape
        with open(f"{outpath}/.zarray", 'w') as outfile:
            json.dump(data, outfile)
        os.system(f"cp {inpath}/.zattrs {outpath}/.zattrs")
    combined_ds.coords.to_dataset().to_zarr(coordzarr, mode='w')
    os.system(f"cp -r {coordzarr}/* {masterzarr}/")
    shutil.rmtree(coordzarr)
        

def copy_files_to_new():
    for ti, time in enumerate(combined_ds.time.values):
        for i, ds in enumerate(all_ds):
            if time in ds.time:
                file_ti = int(np.argwhere(ds.time.values.flatten()==time))
                for key in combined_ds.keys():
                    paths = glob.glob(f"{zarrfiles[i]}/{key}/{file_ti}.*")
                    for path in paths:
                        
                        file = '.'.join([str(ti)]+path.split('/')[-1].split('.')[1:])
                        print(f"{path} -> {masterzarr}/{key}/{file}")
                        os.system(f"cp {path} {masterzarr}/{key}/{file}")
                break

################################################################################
# do the work
################################################################################
                
if __name__=='__main__':
    set_up_empty_zarr()
    copy_files_to_new()
    dz = xr.open_zarr(masterzarr).load()
    
    # run check to verify all values in right place in combined zarr
    for ds in all_ds:
        x1 = ds.load()
        x2 = dz.sel(time=ds.time).load()
        print(x1.equals(x2))
    
                

