import numpy as np

def filter_to_power_metadata_match(pv_metadata_df, pv_power_df):
    """
    Modify the PV output metadata and PV output power dataframes in place to
    only keep systems which are present in both data sources.
    
    Parameters
    ----------
    pv_metadata_df : pandas.DataFrame of dimension (system_id, [any])
    pv_power_df : pandas.DataFrame of dimension ([any], system_id)
    
    Returns
    -------
    None
    """
    non_matched_index = np.setdiff1d(pv_metadata_df.index, pv_power_df.columns)
    non_matched_column = np.setdiff1d(pv_power_df.columns, pv_metadata_df.index)
    pv_power_df.drop(columns=non_matched_column, inplace=True)
    pv_metadata_df.drop(index=non_matched_index, inplace=True)
    return

def train_test_split_day(pv_power_df, test_size, shuffle=True, seed=None):
    '''Split the PV output power dataframe into train and test sets based on 
    date.
    
    Parameters
    ----------
    pv_power_df : pandas.DataFrame of dimension (datetime, system_id),
        PV output power with pandas.DatetimeIndex type index.
    test_size : float, int,
        If float, should be between 0.0 and 1.0 and represent the proportion of 
        the dataset to include in the test split. If int, represents the 
        absolute number of test samples. 
    shuffle : bool, optional
        Whether to split days randomly. Else test days are latest in dataset
        
    Returns
    -------
    train, test : (pandas.DataFrame, pandas.DataFrame) of dimension 
                  (datetime, system_id)
        Views of the original DataFrame
    '''
    if test_size<0:
        raise ValueError("test_size can't be negative")
        
    days = np.unique(pv_power_df.index.date)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        days = np.random.permutation(days)
    
    if isinstance(test_size, float):
        if test_size>1:
            raise ValueError("test_size fraction can't be greater than 1")
        else:
            test_size = int(len(days)*test_size)
    
    n = len(days)-test_size
    train_bool = np.isin(pv_power_df.index.date, days[:n])
    test_bool = np.isin(pv_power_df.index.date, days[n:])
    
    return pv_power_df.loc[train_bool], pv_power_df.loc[test_bool]