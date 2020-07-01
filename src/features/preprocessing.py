import numpy as np

def filter_to_power_metadata_match(pv_metadata_df, pv_power_df):
    non_matched_index = np.setdiff1d(pv_metadata_df.index, pv_power_df.columns)
    non_matched_column = np.setdiff1d(pv_power_df.columns, pv_metadata_df.index)
    pv_power_df.drop(columns=non_matched_column, inplace=True)
    pv_metadata_df.drop(index=non_matched_index, inplace=True)
    return 