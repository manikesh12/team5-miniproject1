import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
import seaborn as sns
import xgboost as xgb
import streamlit as st



def create_df(start_date,end_date):
    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date)

    # Initialize empty lists for dataframe columns
    dates = []
    stores = []
    items = []

    # Generate data for the dataframe

    for store in range(1, 11):  # Store values from 1 to 10
        for item in range(1, 51):  # 50 items for each store
            for date in date_range:
                dates.append(date)
                stores.append(store)
                items.append(item)

    # Create the dataframe
    data = {
        'date': dates,
        'store': stores,
        'item': items
    }

    df = pd.DataFrame(data)

    return df

