import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from minisom import MiniSom
from PIL import Image

# Page layout
st.set_page_config(layout="wide")

# Title of the app
st.title('Wine Analysis')

# Load the dataset
data1 = pd.read_csv('winequality-red.csv', delimiter=';')
data1['wine type'] = 'R'
data1['wine'] = 'Red'

data2 = pd.read_csv('winequality-white.csv', delimiter=';')
data2['wine type'] = 'W'
data2['wine'] = 'White'

df = pd.concat([data1, data2], ignore_index=True)

X = df.iloc[1:, :-3]  # Features
Y = df.iloc[:, -2]   # Wine Type

X = X.to_dict(orient='split')
X = X['data']

Y = Y.to_dict()
Y = np.array(list(Y.values()))



###########################
########### 1A ############
st.subheader('1A. Grouping of red and white wines')
st.markdown('<span style="color: red; font-size: 16px;">Red Wine</span> <span style="color: white; font-size: 16px;">  |  </span> <span style="color: blue; font-size: 16px;">White Wine</span>', unsafe_allow_html=True)

image = Image.open("som_wine.png")


# Display the plot in Streamlit

col1, _ = st.columns([6, 10])  # Adjust the first number to control the width ratio
with col1:
#    st.pyplot(fig)
    st.image(image, use_column_width=True)


###########################
########### 1C ############
st.write("<br>" * 2, unsafe_allow_html=True)
st.subheader('1C. Predicting Quality Values')

Wine_filter = st.radio('Select Wine', options=list(df['wine'].unique()))
filtered_df = df[df['wine'] == Wine_filter]

#st.write(desc_df)

# Red Wine
df1 = data1.drop(['wine type'], axis = 1)
fig1 = px.parallel_coordinates(df1, color="quality", labels={
        'fixed acidity': 'Fixed Acidity', 'volatile acidity': 'Volatile Acidity', 'citric acid': 'Citric Acid',
        'residual sugar': 'Residual Sugar', 'chlorides': 'Chlorides', 'free sulfur dioxide': 'Free Sulfur Dioxide',
        'total sulfur dioxide': 'Total Sulfur Dioxide', 'density': 'Density', 'pH': 'pH', 'sulphates': 'Sulphates',
        'alcohol': 'Alcohol', 'quality': 'Quality'},
        color_continuous_scale=px.colors.sequential.Plasma, template = 'plotly_white', height = 700)

fig1.update_layout(
    font=dict(
        family="Arial, sans-serif",
        size=14,
        color="white"
    ),
    paper_bgcolor='rgb(30, 30, 30)',
    plot_bgcolor='rgb(30, 30, 30)',
    margin=dict(l=100)
)

for dim in fig1.data[0]['dimensions']:
    if dim['label'] == 'Quality':
        dim['tickvals'] = list(range(int(df1['quality'].min()), int(df1['quality'].max()) + 1))
        dim['ticktext'] = [str(x) for x in dim['tickvals']]

# White Wine

df2 = data2.drop(['wine type'], axis = 1)
fig2 = px.parallel_coordinates(df2, color="quality", labels={
        'fixed acidity': 'Fixed Acidity', 'volatile acidity': 'Volatile Acidity', 'citric acid': 'Citric Acid',
        'residual sugar': 'Residual Sugar', 'chlorides': 'Chlorides', 'free sulfur dioxide': 'Free Sulfur Dioxide',
        'total sulfur dioxide': 'Total Sulfur Dioxide', 'density': 'Density', 'pH': 'pH', 'sulphates': 'Sulphates',
        'alcohol': 'Alcohol', 'quality': 'Quality'},
        color_continuous_scale=px.colors.sequential.Plasma, template = 'plotly_dark', height = 700)

fig2.update_layout(
    font=dict(
        family="Arial, sans-serif",
        size=14,
        color="white"
    ),
    paper_bgcolor='rgb(30, 30, 30)',
    plot_bgcolor='rgb(30, 30, 30)',
    margin=dict(l=100)
)

for dim in fig2.data[0]['dimensions']:
    if dim['label'] == 'Quality':
        dim['tickvals'] = list(range(int(df2['quality'].min()), int(df2['quality'].max()) + 1))
        dim['ticktext'] = [str(x) for x in dim['tickvals']]


if Wine_filter == 'Red':
    st.plotly_chart(fig1)
elif Wine_filter == 'White':
    st.plotly_chart(fig2)