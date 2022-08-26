"""" Utility functions for notebooks."""
import pandas as pd
from datetime import datetime as dt
from functools import wraps

class FuncUtils(object):
    def log_shapetime(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args,**kwargs)
            print(func.__name__, "Shape:", result.shape, "Columns:", result.columns)
            return result
        return wrapper

    def check_df(self, dataframe, head=5):
        print(f'{" Info ":-^39}')
        print(dataframe.info())
        print(f'{" Head ":-^53}')
        print(dataframe.head(head))
        print(f'{" Tail ":-^58}')
        print(dataframe.tail(head))
        print(f'{" Quantiles ":-^97}')
        print(dataframe.describe([0.25, 0.50, 0.95, 0.99]).T)

    @log_shapetime
    def add_timeseries_to_data(self, dataf, date_col):
        dataf = dataf.copy()
        dataf[date_col] = pd.date_range(dt.today(), periods=1500).tolist()
        return dataf

    @log_shapetime
    def rename_cols(self, dataf, mapping_dict: dict):
        dataf = dataf.copy()
        dataf = dataf.rename(columns=mapping_dict)
        return dataf

    @log_shapetime
    def reverse_col_order(self, dataf):
        dataf = dataf.copy()
        dataf = dataf.iloc[:, ::-1]
        return dataf