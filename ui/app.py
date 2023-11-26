import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
import seaborn as sns
import xgboost as xgb

from helper import create_df


def split_date(df):
    # split date
    df['date'] = pd.to_datetime(df.date, format="%Y-%m-%d")

    # adding columns for date identifiers
    df['day'] = df.date.dt.day.astype(int)
    df['month'] = df.date.dt.month.astype(int)
    df['year'] = df.date.dt.year.astype(int)
    df['day_of_week'] = df.date.dt.dayofweek.astype(int)  # Mon:0, Sun: 6
    df['week_of_month'] = (df['date'].dt.isocalendar().week.astype(int) - 1) % 4 # 0 to 4
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int) # 0 to 52
    df['quarter'] = df['date'].dt.quarter.astype(int) # 1 to 4
    # df = df.drop("date", axis=1)
    # df = df.drop("id", axis=1)

    return df


def sales_pred(data):
    # data=pd.read_csv('D:/project/team5-miniproject1/test.csv')
    data=split_date(data)
    temp=data
    data = data.drop("date", axis=1)
    # print(temp.dtypes)
    dtest = xgb.DMatrix(data, enable_categorical=True)
    # print(data)
    pred = loaded_model.predict(dtest)
    temp['sales']=pred
    # print(temp)

    return temp

def graph_sales(df):
    daily_sales= df.groupby('date', as_index=False)['sales'].sum()
    fig = plt.figure(figsize=(15, 10))
    plt.plot(daily_sales['date'], daily_sales['sales'])
    # # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Daily sales')
    # Display the figure
    st.pyplot(fig)

def graph_sales_by_store(df):
    fig = plt.figure(figsize=(15, 10))
    store_daily_sales= df.groupby(['store','date'],as_index=False)['sales'].sum()
    # Loop through unique stores
    for store in store_daily_sales['store'].unique():
        current_store_daily_sales = store_daily_sales[store_daily_sales['store'] == store]
        plt.plot(current_store_daily_sales['date'], current_store_daily_sales['sales'], label=f'Store {store}')

    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Store daily sales')
    plt.legend()
    st.pyplot(fig)

def graph_sales_by_item(df):
    fig = plt.figure(figsize=(15, 10))
    item_daily_sales = df.groupby(['item', 'date'], as_index=False)['sales'].sum()
    for item in item_daily_sales['item'].unique():
        current_item_daily_sales = item_daily_sales[item_daily_sales['item'] == item]
        plt.plot(current_item_daily_sales['date'], current_item_daily_sales['sales'], label=f'Item {item}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Item daily sales')
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)


def graph_sales_year(df,from_date,to_date):
    fig = plt.figure(figsize=(15, 10))
    from_year=from_date.year
    to_year=to_date.year
    for year in range(from_year , to_year):
        monthly_sales = df.loc[df.year == year].groupby('month').sales.mean()
        # print(monthly_sales)
        plt.plot(range(1 , 13), monthly_sales, label = year)
        plt.xlabel('month')

    monthly_sales = df.loc[df.year == to_year].groupby('month').sales.mean()
    # print(to_date.month)
    # print(monthly_sales)
    plt.plot(range(1 , to_date.month+1), monthly_sales, label = to_year)
    plt.xlabel('month')
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)

def graph_sales_quarter(df,from_date,to_date):
    quarterly_sales = df.groupby(["year", "quarter"]).mean()['sales']

    fig = plt.figure(figsize=(15, 10))

    quarterly_sales.plot(kind = "bar", label = "average sales")

    plt.title("average sales of all stores by quarter")
    plt.ylabel("average sales")
    plt.xlabel("year, quarter")
    plt.xticks(rotation = 30)
    st.pyplot(fig)
    




loaded_model=pickle.load(open("ui/premodel.pkl",'rb'))




# Title and description
st.title('Store Item Demand Forecasting')
st.write('select the date range')

# Date input
from_date = st.date_input('From Date', pd.Timestamp('2018-01-01'), min_value=datetime.date(2018, 1, 1),max_value=datetime.date(2019, 12, 31))
to_date = st.date_input('To Date', pd.Timestamp('2018-03-01'),min_value=datetime.date(2018, 1, 1),max_value=datetime.date(2019, 12, 31))


# print(from_date.year)

# Check if from_date is after 2017-12-31
if from_date <= datetime.date(2017,12,31):
    st.error('From date should be after 2017-12-31')
if from_date> to_date:
    st.error('From date should be smaller than To date')
else:
    st.success('Dates are valid!')

    data=create_df(from_date,to_date)

    graph_data=sales_pred(data)

    option = st.selectbox(
        'select a graph',
        ('daily sales', 'sales by stores', 'sales by items','sales by year','sales by quarter'),
        index=None,
        placeholder="Select a graph...",)


    if option=="daily sales":
        graph_sales(graph_data)

    if option=="sales by stores":
        graph_sales_by_store(graph_data)

    if option=="sales by items":
        graph_sales_by_item(graph_data)
    
    if option=="sales by year":
        graph_sales_year(graph_data,from_date,to_date)
        
    if option=="sales by quarter":
        graph_sales_quarter(graph_data,from_date,to_date)

    # graph_sales(sales_pred())
