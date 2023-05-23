import seaborn as sns
import pandas as pd
import time



def compare_dt_formats(dataset):
    # Load the tips dataset
    tips = sns.load_dataset(dataset)

    formats = ['csv', 'json', 'pickle', 'parquet', 'feather']

    for format in formats:
        # Time the save operation
        start_save = time.time()
        if format == 'csv':
            tips.to_csv('tips.' + format)
        elif format == 'json':
            tips.to_json('tips.' + format)
        elif format == 'pickle':
            tips.to_pickle('tips.' + format)
        elif format == 'parquet':
            tips.to_parquet('tips.' + format)
        elif format == 'feather':
            tips.to_feather('tips.' + format)
        end_save = time.time()
        save_time = end_save - start_save

        # Time the load operation
        start_load = time.time()
        if format == 'csv':
            pd.read_csv('tips.' + format)
        elif format == 'json':
            pd.read_json('tips.' + format)
        elif format == 'pickle':
            pd.read_pickle('tips.' + format)
        elif format == 'parquet':
            pd.read_parquet('tips.' + format)
        elif format == 'feather':
            pd.read_feather('tips.' + format)
        end_load = time.time()
        load_time = end_load - start_load

        # Print the results
        print(f"{format} - load time: {load_time:.6f} / save time: {save_time:.6f}")

compare_dt_formats('tips')


