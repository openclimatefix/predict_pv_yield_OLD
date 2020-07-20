import numpy as np

def filter_to_power_metadata_match(pv_metadata_df, pv_power_df):
    non_matched_index = np.setdiff1d(pv_metadata_df.index, pv_power_df.columns)
    non_matched_column = np.setdiff1d(pv_power_df.columns, pv_metadata_df.index)
    pv_power_df.drop(columns=non_matched_column, inplace=True)
    pv_metadata_df.drop(index=non_matched_index, inplace=True)
    return

def train_test_split_day(pv_output, test_size, shuffle=True, seed=None):
    '''
    Parameters
    ----------
    pv_output : pandas.DataFrame,
        DataFrame must have timestamp index
    test_size : float, int,
        If float, should be between 0.0 and 1.0 and represent the proportion of 
        the dataset to include in the test split. If int, represents the 
        absolute number of test samples. 
    shuffle : bool, optional
        Whether to split days randomly. Else test days are latest in dataset
        
    Returns
    -------
    train, test : (pandas.DataFrame, pandas.DataFrame)
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
    
    return pv_output.loc[train_bool], pv_output.loc[test_bool]