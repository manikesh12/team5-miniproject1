import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
import seaborn as sns

# loaded_model=pd.read_pickle(open("D:/project/team5-miniproject1/trained_model.sav",'rb'))

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

    return df


def sales_pred():
    data=pd.read_csv('D:/project/team5-miniproject1/test.csv')
    data=split_date(data)
    pred = loaded_model.predict(data)
    print(pred)
    # return pred
print(sales_pred())

def graph_sales(df):
    fig, ax = plt.subplots()

    daily_sales= df.groupby('date', as_index=False)['sales'].sum()
    ax.figure(figsize=(10, 6))
    sns.set(style="darkgrid")

    # Plot the daily sales

    ax.plot(daily_sales['date'], daily_sales['sales'])

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Daily sales')

    st.pyplot(fig)

    

    # Show the plot
    # plt.show(fig)


# Title and description
st.title('Graph Plotter')
st.write('Enter two dates and see the graph of y=x+1')

# Date input
from_date = st.date_input('From Date', pd.Timestamp('2018-01-01'), min_value=datetime.date(2013, 1, 1))
to_date = st.date_input('To Date', pd.Timestamp('2023-11-22'),min_value=datetime.date(2013, 1, 1))

# Check if from_date is after 2017-12-31
if from_date <= datetime.date(2017,12,31):
    st.error('From date should be after 2017-12-31')
if from_date> to_date:
    st.error('From date should be smaller than To date')
else:
    st.success('Dates are valid!')





# Function to plot graphs
def plot_graph(equation):
    x = np.linspace(-10, 10, 100)
    if equation == 'y=x+1':
        y = x + 1
        title = 'Graph of y=x+1'
    elif equation == 'y=x+2':
        y = x + 2
        title = 'Graph of y=x+2'

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    st.pyplot(fig)

# Button to display y=x+1 graph
if st.button('Display y=x+1 graph'):
    graph_sales(sales_pred())
    # plot_graph('y=x+1')

# Button to display y=x+2 graph
if st.button('Display y=x+2 graph'):
    plot_graph('y=x+2')