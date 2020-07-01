# TO DO
# many fields in NWP data have same value over huge area. Take single values 
# rather than rectangular patch.

import numpy as np
import pandas as pd
import numba as nb

import torch
import copy
from sklearn.utils import shuffle

import pvlib
from pvlib.location import Location

import threading
from multiprocessing import Value


def compute_clearsky(times, latitudes, longitudes):
    clearsky = np.full(shape=(len(times), len(latitudes), 3), 
                       fill_value=np.NaN, dtype=np.float32)

    
    for i, (lat, lon) in enumerate(zip(lat, lon)):
        loc = Location(
                latitude=lat,
                longitude=lon,
                tz='UTC').get_clearsky(times)
        clearsky[:,i,:] = clearsky_for_location.values    
    
    return clearsky

def data_source_intersection(pv_output, clearsky=None, sat_loader=None, nwp_loader=None, 
                             lead_time=pd.Timedelta('0min')):
    """Return the datetimes where the pv data, the available satellite data 
    (optional), and the available numerical weather prediction data (optional)
    overlap. This is considered with a forecast lead time."""

    intersect_times = pv_output.index.values
    
    if clearsky is not None:
        intersect_times = np.intersect1d(clearsky.index,values, intersect_times)
    
    if sat_loader is not None:
        sat_times = sat_loader.dataset.time.values + pd.Timedelta('1min') - lead_time
        intersect_times = np.intersect1d(sat_times, intersect_times)
        
    if nwp_loader is not None:
        nwp_times = nwp_loader.dataset.time.values
        forecast_time = pd.to_datetime(intersect_times - lead_time)
        in_nwp = np.in1d(forecast_time.floor('180min'), nwp_times)
        intersect_times = intersect_times[in_nwp]
        
    return pd.DatetimeIndex(intersect_times)

@nb.jit()
def _shuffled_indexes_for_pv(pv_output_values, consec_samples_per_datetime):
    """
    Parameters
    ----------
    pv_output_values : array_like,
        The pv-output values
    consec_samples_per_datetime: int
        number of samples per each datetime row. Value -1 means take all pv 
        systems fotr each randomly selected datetime. 
    """
    if consec_samples_per_datetime<1 and consec_samples_per_datetime!=-1:
        raise ValueError("invalid samples_per_datetime")
    rowchunks = []
    n = consec_samples_per_datetime
    i_max, j_max = pv_output_values.shape
    
    # get indices without nans
    for i in range(i_max):
        row=[]
        for j in range(j_max):
            if not np.isnan(pv_output_values[i,j]):
                row.append((i,j))
                
        if len(row)>0:
            # shuffle row
            row = [row[k] for k in np.random.permutation(np.arange(len(row)))]
            if consec_samples_per_datetime==-1: n = len(row)
            rowchunks.extend([row[m:m+n] for m in range(0, len(row), n)])
    
    # shuffle the order of the same-time-different-pv-system chunks
    rowchunks = [rowchunks[k] for k in np.random.permutation(np.arange(len(rowchunks)))]
    
    # fill with value so can be converted to numpy array
    for i, rc in enumerate(rowchunks):
        if consec_samples_per_datetime==-1: n = j_max
        rowchunks[i] = rc+[(-1,-1)]*(n-len(rc))
        
    return np.array(rowchunks, dtype=np.int32)

    
class cross_processor_batch:
    """
    A superbatch data generator object for CPU or GPU.
    Loading is done on CPU and ported to GPU if required.
    ...
    
    Parameters
    ----------
    y : pandas.DataFrame 
        PV prediction data. Index must be 
        pandas.DatetimeIndex type and columns must be unique system identifiers.
    y_meta : pandas.DataFrame
        PV metadata with columns which include the x and y coordinates of the PV 
        systems. Index must be unique system identifiers which cover all systems 
        in `y`.
    clearsky : pandas.DataFrame, optional
        Clearsky GHI data for each time and position as the data from the 
        systems in `y`.
    sat_loader : sat_loader.SatelliteLoader, optional
        Satellite image loader object which returns data with any required
        preprocessing already implemented.
    nwp_loader : nwp_loader.NWPLoader, optional
    batch_size : int, default 256
        Number of samples to use in batch
    batches_per_superbatch : int, default 16
        number of batches of size `batch_size` to load and store at once.
    n_superbatches : int, default 1
        Limit of number of superbatches to load in generator functionality.
    n_epochs : int, optional
        Overrides n_superbatches and we loop through the entire dataset this 
        many times.
    gpu : bool, default False
        Whether to load the data into CUDA GPU. Will initially load in CPU then
        transfer over.
    shuffle_datetime bool, default True (strongly recommended)
        Whether to shuffle the datetimes before loading so that all samples in
        batch are not in consecutive order.
    

    Attributes
    ----------
    cpu_superbatch : dict of numpy.array
    gpu_superbatch : dict of numpy.array, only if `gpu` is True.

    Methods
    -------
    next()
        Generate the next batch of data and load a new superbatch if required.
    """
    # slots for faster attribute access
    # on speed test this contributes maybe 2% speedup
    __slots__ = ['datetime', 'lead_time', 'y', 'y_meta', 'clearsky', 
              'sat_loader','nwp_loader', 'batch_size', 'batches_per_superbatch', 
              'superbatch_size','n_superbatches', 'n_epochs', 'gpu', 
              'batch_index','superbatch_index', 'epoch', 
              'consec_samples_per_datetime', 'indexes', 'index_number',
              'extinguished', 'reshuffle_required', 'parallel_loading_cores', 
              '_parallel_loading_cache', 'cpu_superbatch', 'gpu_superbatch']
    
    def __init__(self, y, y_meta, 
                 clearsky=None,
                 sat_loader=None,
                 nwp_loader=None,
                 lead_time=pd.Timedelta('0min'),
                 batch_size=256, batches_per_superbatch=16, 
                 n_superbatches=1, n_epochs=None, 
                 gpu=False,
                 consec_samples_per_datetime=10,
                 parallel_loading_cores = 2,
                ):
        
        # remove UTC timezone info
        y.index = y.index.values
        
        # make sure we only keep datetimes where we have all data
        datetime = data_source_intersection(y, clearsky=clearsky,
                                            sat_loader=sat_loader, 
                                            nwp_loader=nwp_loader, 
                                            lead_time=lead_time)
        if len(datetime)==0:
            raise ValueError('Data sources do not overlap in time')
                        
        self.datetime = datetime
        self.lead_time = lead_time
        
        # store datasets
        self.y = y.reindex(datetime).astype(np.float32)
        self.y_meta = y_meta.reindex(y.columns)
        if clearsky is None:
            self.clearsky = None
        else:
            self.clearsky = clearsky.reindex(datetime).astype(np.float32)
        
        # store loaders
        self.sat_loader = sat_loader
        self.nwp_loader = nwp_loader
        
        # store options
        self.batch_size = batch_size
        self.batches_per_superbatch = batches_per_superbatch
        self.superbatch_size = batch_size*batches_per_superbatch
        self.n_superbatches = n_superbatches
        self.n_epochs = n_epochs
        self.gpu = gpu

        # Initiate these indices for looping through and loading the data
        self.batch_index = -1
        self.superbatch_index = -1
        self.epoch = 0
        self.consec_samples_per_datetime = consec_samples_per_datetime
        self.indexes = _shuffled_indexes_for_pv(self.y.values, 
                                               consec_samples_per_datetime)
        self.index_number = 0
        self.extinguished = False
        self.reshuffle_required = False
        self.parallel_loading_cores = parallel_loading_cores
        # Make loader copies for parallel loading
        # This makes better use of caching for speed
        self._parallel_loading_cache = {i:{
                'nwp_loader': nwp_loader if i==0 else copy.deepcopy(nwp_loader),
                'sat_loader': sat_loader if i==0 else copy.deepcopy(sat_loader),
                'thread_current_index':-1,
                'thread_subindex':-1
            } for i in range(self.parallel_loading_cores)}
        self._instantiate_superbatch('cpu')
        if self.gpu: self._instantiate_superbatch('gpu')
            
            
    def __next__(self):
        """Return the next batch of data and load a new superbatch if 
        required"""
        
        if self.extinguished:
            raise StopIteration
            
        self.batch_index += 1

        # load if very first batch or we have already looped though loaded data
        first_item = (self.batch_index -1) == self.superbatch_index == -1
        load_required = (first_item or (
            self.batch_index >= self.batches_per_superbatch
        ))
        
        none_greater = lambda x, y: (y is not None) and x>=y
        if load_required:
            # Stop iteration if
            # if the number of epochs has reached it's limit or
            # if the number of superbatches has reached it's limit
            if (none_greater(self.epoch, self.n_epochs) or 
                none_greater(self.superbatch_index+1, self.n_superbatches)):
                self.extinguished = True
                raise StopIteration
            
            # load more data otherwise
            
            # if we have reached the end of an epoch then reshuffle
            if self.reshuffle_required:
                self.indexes = _shuffled_indexes_for_pv(self.y.values, 
                                               self.consec_samples_per_datetime)
                self.index_number = 0
            
            self.load_next_superbatch_to_cpu()
            if self.gpu:
                self.transfer_superbatch_to_gpu()
            self.batch_index = 0

        batch = self.return_batch(self.batch_index, gpu=self.gpu)
        
        return batch
    
    def __iter__(self):
            return self
    
    def _instantiate_superbatch(self, where):
        """Instantiate space for data in memory"""
        def new_array(size):
            if where=='gpu':
                return torch.full(size=size, 
                                  fill_value=0., 
                                  dtype=torch.float16, 
                                  device='cuda')
            elif where=='cpu':
                return np.full(shape=size, 
                                  fill_value=0., 
                                  dtype=np.float32)
        
        superbatch = {}
        superbatch['y'] = new_array((self.superbatch_size, 1))
        superbatch['day_fraction'] =  new_array((self.superbatch_size, 1))
        
        if self.clearsky is not None:
            superbatch['clearsky'] = new_array((self.superbatch_size, 1))
        
        if self.sat_loader is not None:
            superbatch['satellite'] = new_array((self.superbatch_size,) 
                                        + self.sat_loader.sample_shape
            )
        if self.nwp_loader is not None:
            superbatch['nwp'] = new_array((self.superbatch_size,) 
                                        + self.nwp_loader.sample_shape
            )
        
        if where=='cpu':
            self.cpu_superbatch = superbatch
        elif where=='gpu':
            self.gpu_superbatch = superbatch
        return
    
    def shuffle_cpu_superbatch(self):
        rand_state = np.random.get_state()
        for key in self.cpu_superbatch.keys():
                np.random.set_state(rand_state)
                np.random.shuffle(self.cpu_superbatch[key])
    
    def load_next_superbatch_to_cpu(self):
        
        # share this value between threads
        index_n = Value('i', self.index_number)
        
        # store whether we run over end of data
        newepoch = Value('i', 0)

        def single_thread_data_gather(n_start, n_stop, cache_dict):
            
            thread_current_index = cache_dict['thread_current_index']
            thread_subindex = cache_dict['thread_subindex']
            sat_loader = cache_dict['sat_loader']
            nwp_loader = cache_dict['nwp_loader']
            
            
            
            # if first use of this function get an initial index
            if thread_current_index==-1:
                with index_n.get_lock():
                    thread_current_index = index_n.value
                    index_n.value += 1
                thread_subindex = 0
                
            
            for n in range(n_start, n_stop):
                
                # loop until we find valid sample
                completed_new=False
                while not completed_new:
                    
                    # assume this next sample will be okay for now
                    completed_new = True
                    
                    i, j = self.indexes[thread_current_index, thread_subindex]
                    
                    # To make array regular shaped nan values were filled with
                    # -1. Nan values always occur at end of index rows
                    if i==j==-1:
                        with index_n.get_lock():
                            thread_current_index = index_n.value
                            index_n.value += 1
                        thread_subindex = 0
                        completed_new = False
                        continue
                        
                    
                    self.cpu_superbatch['y'][n] = self.y.values[i, j]
                    day_frac = lambda x: ((x.hour+x.minute/60)/24.)
                    self.cpu_superbatch['day_fraction'][n] = day_frac(self.datetime[i])
                    
                    if self.clearsky is not None:
                        if np.isnan(self.clearsky.values[i, j]):
                            completed_new = False
                            continue
                        else:
                            self.cpu_superbatch['clearsky'][n] = self.clearsky.values[i, j]
                    
                    dt = pd.Timestamp(self.datetime[i])-self.lead_time
                    
                    if self.sat_loader is not None:
                        sat = sat_loader.get_rectangle_array(
                                dt.floor('5min') - pd.Timedelta('1min'),
                                *self.y_meta.loc[self.y.columns[j]][['x', 'y']]
                                ).astype(np.float32)[..., np.newaxis]
                        # check for nans and shape
                        completed_new = (completed_new and 
                            not np.any(np.isnan(sat)) and 
                            self.cpu_superbatch['satellite'][n].shape==sat.shape
                        )
                        if not completed_new: 
                            continue
                        else:
                            self.cpu_superbatch['satellite'][n] = sat
                    
                    if self.nwp_loader is not None:
                        nwp = nwp_loader.get_rectangle_array(dt, self.datetime[i], 
                                *self.y_meta.loc[self.y.columns[j]][['x', 'y']]
                                ).astype(np.float32)[..., np.newaxis]
                        # check for nans and shape
                        completed_new = (completed_new and 
                            not np.any(np.isnan(nwp)) and 
                            self.cpu_superbatch['nwp'][n].shape==nwp.shape
                        )
                        if not completed_new: 
                            continue
                        else:
                            self.cpu_superbatch['nwp'][n] = nwp
                    
                    # update thread indices and global next index
                    thread_subindex+=1
                    if thread_subindex>=self.indexes.shape[1]:
                        with index_n.get_lock():
                            thread_current_index = index_n.value
                            index_n.value += 1
                            if index_n.value >= self.indexes.shape[0]:
                                index_n.value = 0
                                newepoch.value += 1
                        thread_subindex = 0
                        
            # store the current loading location
            cache_dict['thread_current_index'] = thread_current_index
            cache_dict['thread_subindex'] = thread_subindex
            
            return
        
        if self.parallel_loading_cores>1:
        
            sample_nums = np.linspace(0, self.superbatch_size, 
                                      self.parallel_loading_cores+1).astype(int)

            chunk_args = [[sample_nums[i], sample_nums[i+1], cache_dict] 
                                  for i, cache_dict in self._parallel_loading_cache.items()]

            # Spawn one thread per chunk
            threads = [threading.Thread(target=single_thread_data_gather, args=args)
                                   for args in chunk_args]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        
        else:
            single_thread_data_gather(0, self.superbatch_size, 
                                      self._parallel_loading_cache[0])

        # store where we loaded up to
        self.index_number = index_n.value
        self.epoch += newepoch.value
        
        # delay reshuffling in case we have hit epoch limit
        if newepoch.value >0:
            self.reshuffle_required = True
        self.superbatch_index+=1
        self.shuffle_cpu_superbatch()
        return
    

    def transfer_superbatch_to_gpu(self):
        if not self.gpu:
            raise TypeError('gpu support was set to False')
        for k, v in self.cpu_superbatch.items():
            try:
                self.gpu_superbatch[k].copy_(torch.HalfTensor(v))
            except:
                raise Exception('Problem with', k)
    
    def return_batch(self, batch_index, gpu=False):
        if gpu:
            superbatch = self.gpu_superbatch
        else:
            superbatch = self.cpu_superbatch
        i1 = batch_index*self.batch_size
        i2 = i1 + self.batch_size
        dict_view = {k:v[i1:i2] for k, v in superbatch.items()}
        return dict_view