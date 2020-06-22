from sklearn.utils import shuffle
import numba as nb
from numba import njit, objmode
import numpy as np
import torch

import pvlib
from pvlib.location import Location


def compute_clearsky(times, latitudes, longitudes):
    clearsky = np.full(shape=(len(times), len(latitudes), 3), fill_value=np.NaN, dtype=np.float32)

    
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
        _, ind, _ = np.intersect1d(forecast_time.floor('180min'), 
                                   nwp_loader.dataset.time, return_indices=True)
        intersect_times = intersect_times[ind]
        
    return intersect_times

@nb.jit()
def shuffled_indexes_for_pv(pv_output_values, consec_samples_per_datetime):
    """
    Parameters
    ----------
    pv_output_values : array_like 
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
            row = [row[i] for i in np.random.permutation(np.arange(len(row)))]
            if consec_samples_per_datetime==-1: n = len(row)
            rowchunks.extend([row[m:m+n] for m in range(0,len(row), n)])
    
    rowchunks = [rowchunks[i] for i in np.random.permutation(np.arange(len(rowchunks)))]
    
    indexes = []
    for rowchunk in rowchunks:
        for ind in rowchunk:
            indexes.append(ind)
        
    return np.array(indexes, dtype=np.int32)


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
    satellite_loader : sat_loader.SatelliteLoader, optional
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
    def __init__(self, y, y_meta, 
                 clearsky=None,
                 satellite_loader=None,
                 nwp_loader=None,
                 lead_time=pd.Timedelta('0min'),
                 batch_size=256, batches_per_superbatch=16, 
                 n_superbatches=1, n_epochs=None, 
                 gpu=False,
                 consec_samples_per_datetime=10,
                ):
        
        # make sure we only keep datetimes where we have all data
        datetime = data_source_intersection(pv_output, clearsky=clearsky,
                                            sat_loader=satellite_loader, 
                                            nwp_loader=nwp_loader, 
                                            lead_time=lead_time)
        if len(datetime)==0:
            raise ValueError('Data sources do not overlap in time')
            
        assert np.all(y_meta.index == y.columns), "metadata doesn't match y"
            
        self.datetime = datetime
        self.lead_time = lead_time
        
        # store datasets
        self.y = y.reindex(datetime).astype(np.float32)
        self.y_meta = y_meta
        if clearsky is None:
            self.clearsky = None
        else:
            self.clearsky = clearsky.reindex(datetime).astype(np.float32)
        
        # store loaders
        self.satellite_loader = satellite_loader
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
        self.indexes = shuffled_indexes_for_pv(self.y.values, 
                                               consec_samples_per_datetime)
        self.index_number = 0
        self.extinguished = False
        self.reshuffle_required = False

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
                self.indexes = shuffled_indexes_for_pv(self.y.values, 
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
        
        if self.clearksy is not None:
            superbatch['clearsky'] = new_array((self.superbatch_size, 1))
        
        if self.satellite_loader is not None:
            superbatch['satellite'] = new_array((self.superbatch_size,) 
                                        + self.satellite_loader.sample_shape
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
        if not hasattr(self, '_load_next_superbatch_to_cpu'):
            # generate numba function for fast loading
            self._load_next_superbatch_to_cpu = self._cpu_load_function_factory(
                        method=self.method)

        day_fraction_in = ((self.datetime.hour.values +
                            self.datetime.minute.values/60)
                           /24).astype(np.float32)
        
        #if including clearsky
        is_cs = np.int32(self.clearsky is not None)
        if is_cs:
            clearsky_in = self.clearsky.values
            clearsky_out = self.cpu_superbatch['clearsky']
        else:
            clearsky_in = clearsky_out = np.empty((0,0), dtype=np.float32)
        
        # if including satellite imagery
        is_sat = np.int32(self.satellite_loader is not None)
        if is_sat:
            satellite_image_out = self.cpu_superbatch['satellite']
        else:
            satellite_image_out = np.empty((0,0,0,0,0), dtype=np.float32)
            
        # if including nwp data
        is_nwp = np.int32(self.nwp_loader is not None)
        if is_nwp:
            nwp_image_out = self.cpu_superbatch['satellite']
        else:
            nwp_image_out = np.empty((0,0,0,0,0), dtype=np.float32)

        args = [self.y.values, # y_in
                self.cpu_superbatch['y'], # y_out
                is_cs, # include clearksy
                clearsky_in, 
                clearsky_out,
                day_fraction_in, 
                self.cpu_superbatch['day_fraction'], # day_fraction_out
                is_sat, # include sat
                satellite_image_out, 
                is_nwp, # include NWP
                nwp_image_out, 
                np.int32(self.superbatch_size), 
                np.int32(self.index_number),
                self.indexes]

        # load data
        out = self._load_next_superbatch_to_cpu(*args)

        # store where we loaded up to
        self.index_number, newepoch = out
        self.epoch += newepoch
        # delay reshuffling in case we have hit epoch limit
        if newepoch >0:
            self.reshuffle_required = True
        self.superbatch_index+=1
        self.shuffle_cpu_superbatch()
        return
        
    def _cpu_load_function_factory(self):
        """generates numba function which loads data much faster than
        the base python."""
                
        def get_sat(i,j):
            dt = pd.Timestamp(self.datetime[i])-self.lead_time
            dt = dt.floor('5min') - pd.Timedelta('1min')
            return self.satellite_loader.get_rectangle_array(dt
                                *self.y_meta.loc[self.y.columns[j]][['x', 'y']]
                                ).values.astype(np.float32)
        
        def get_nwp(i,j):
            dt = pd.Timestamp(self.datetime[i])-self.lead_time
            return self.nwp_loader.get_rectangle_array(dt, self.datetime[i], 
                                *self.y_meta.loc[self.y.columns[j]][['x', 'y']]
                                ).values.astype(np.float32)
        

        @nb.jit("""int32[:](float32[:,:], float32[:,:],
                            int32, float32[:,:], float32[:,:],
                            float32[:], float32[:,:],
                            int32, float32[:,:,:,:,:],
                            int32, float32[:,:,:,:,:],
                            int32, int32, float32[:,:])""", 
                nopython=True, nogil=True)
        def numba_data_gather(y_in, y_out, 
                              is_cs, clearsky_in, clearsky_out, 
                              day_fraction_in, day_fraction_out,
                              is_sat, satellite_image_out, 
                              is_nwp, nwp_image_out,
                              superbatch_size, index_n, indexes):

            newepoch = 0
            for n in range(superbatch_size):

                completed_new=0
                while completed_new==0:
                    i, j = indexes[index_n]
                    
                    y_out[n] = y_in[i, j]
                    day_fraction_out[n] = day_fraction_in[i]
                    
                    if is_cs:
                        clearsky_out[n] = clearsky_in[i, j]
                    
                    if is_sat:
                        with objmode(sat='float32[:,:,:]'):
                                sat = get_sat(i,j)
                        satellite_image_out[n] = sat
                        
                    if is_nwp:
                        with objmode(nwp='float32[:,:,:]'):
                                nwp = get_sat(i,j)
                        nwp_image_out[n] = nwp

                    # sat data sometimes has NaNs. Check for this
                    completed_new = 1-int(np.any(np.isnan(sat)) or 
                                          np.any(np.isnan(nwp)))
                    
                    index_n+=1
                    if index_n>=len(indexes):
                        index_n = 0
                        newepoch+=1
                    if newepoch>=2:
                        raise Exception('Not enough data for superbatch size')

            return np.array([index_n, int(newepoch)], dtype=np.int32)
        
        return numba_data_gather
        

    def transfer_superbatch_to_gpu(self):
        if not self.gpu:
            raise TypeError('gpu support was set to False')
        for k, v in self.cpu_superbatch.items():
            try:
                self.gpu_superbatch[k].copy_(torch.HalfTensor(v))
            except:
                print('Problem with', k)
                raise
    
    def return_batch(self, batch_index, gpu=False):
        if gpu:
            superbatch = self.gpu_superbatch
        else:
            superbatch = self.cpu_superbatch
        i1 = batch_index*self.batch_size
        i2 = i1 + self.batch_size
        dict_view = {k:v[i1:i2] for k, v in superbatch.items()}
        return dict_view
    
