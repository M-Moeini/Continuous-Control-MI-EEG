import os
import datastream
from joblib import dump, load

CACHE_FILE = 'cached_data.joblib'

def read_data_once(path, p_num_list, block_list):
    # Check if the cache file exists
    if os.path.exists(CACHE_FILE):
        # Load from cache if the file exists
        cached_data_dict_list = load(CACHE_FILE, mmap_mode='r')
        return cached_data_dict_list

    # If the cache file doesn't exist, read the data and save it to the cache
    cached_data_dict_list = []
    for p_num in p_num_list:
        print("Reading", p_num)
        data_dict = datastream.pickle_reader(path, p_num, block_list)
        cached_data_dict_list.append(data_dict)

    # Save to cache
    dump(cached_data_dict_list, CACHE_FILE)
    
    return cached_data_dict_list
