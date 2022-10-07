
from tkinter.ttk import Separator
from turtle import onclick
from matplotlib.pyplot import text
import pandas as pd
netflix_titles = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv')
df = netflix_titles

# so we go from 0 to n-1
len(df.columns)


# funciton to adjust data types
df.dtypes

df.date_added = pd.to_datetime(df.date_added)
df.release_year = pd.DatetimeIndex(pd.to_datetime(df.release_year, format='%Y')).year # format='%y%m%d'

pd.to_numeric(df.duration)

# creating new columns

add new column based on
- condition applied to 1 column
- text split using separator
- extracting part of a text field

def new_value(row, separator, extracted_value):
    return row.split(separator)[extracted_value]

df['duration_length'] = df.apply(lambda row: new_value(row.duration, ' ', 0), axis=1)
df['duration_type'] = df.apply(lambda row: new_value(row.duration, ' ', 1), axis=1)

def new_value1(row, a):
    return row.partition('s')[a]

df['slow_id_only'] = df.apply(lambda row: new_value1(row.show_id, 2), axis=1)

def new_value2(row, a, b):
    return row[a:b]

df['desc_short'] = df.apply(lambda row: new_value2(row.description, 5, 10), axis=1)

df
a='123456s'
a.partition('s')[0]


a = ['s', 1,2,3,4]
a[1:]

s='mango'
list(s)[1:]

a = '1:'
a.strip()[1:-1].split("'")

for i in df.columns:
    print(df[i].unique)

df.iloc[0].unique


df.columns

def vp_summ(df):
    print('#columns:', df.shape[1]) # number of columns
    print('#rows:', df.shape[0]) # number of rows
    for r in df.columns:
        print(r, ':', # column name
        df[r].unique().shape[0], # number of unique elements in the column
        '| example:', df[r][0]) # example of the first element in the column
vp_summ(stations)


for c in df.columns:


df['Breed'].dtype



