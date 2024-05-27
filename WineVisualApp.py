import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from minisom import MiniSom
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
########### 1B ############
st.subheader('1B. Association between Features and Quality')

df = data1.drop(['wine type'], axis = 1)
df = df.drop(['wine'], axis = 1)

# Define a custom color map for quality
color_map = {
    3: '#0E078B',  
    4: '#6501A6',
    5: '#A63989',
    6: '#D06967',  
    7: '#EFAD4F',
    8: '#F4FB58'   
}

# Melt the DataFrame to long format
df_melted = df.melt(id_vars='quality', var_name='attribute', value_name='value')

# Create subplots
num_attributes = df_melted['attribute'].nunique()
rows = (num_attributes // 4) + (num_attributes % 4 > 0)
fig = make_subplots(rows=rows, cols=4, subplot_titles=df_melted['attribute'].unique(), vertical_spacing=0.1)

# Add individual box plots to the subplots
for i, attribute in enumerate(df_melted['attribute'].unique()):
    row = i // 4 + 1
    col = i % 4 + 1
    
    # Create a box plot for the current attribute
    fig_box = px.box(df_melted[df_melted['attribute'] == attribute], 
                     x='quality', y='value', 
                     color='quality', 
                     color_discrete_map=color_map, 
                     template='plotly_dark')
    
    # Add traces from the box plot to the subplot
    for trace in fig_box['data']:
        fig.add_trace(trace, row=row, col=col)

# Define the tick values and tick labels for the x-axis
quality_values = df['quality'].unique()
quality_values.sort()
tickvals = list(quality_values)
ticktext = [str(val) for val in tickvals]

# Update layout to ensure all tick labels show up on x-axis
for i in range(1, rows + 1):
    for j in range(1, 5):
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, row=i, col=j)
        fig.update_xaxes(title_font=dict(family='Arial', size=12, color='black'), row=i, col=j)
        fig.update_yaxes(title_font=dict(family='Arial', size=12, color='black'), row=i, col=j)

# Update subplot titles to use Arial font
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(family='Arial', size=12, color='black')

fig.update_layout(
    title_text='Red Wine: Distribution of Attributes by Quality',
    title_font=dict(family='Arial', size=16, color='black'),
    showlegend=False,
    height=800,
    width=1500
)

# Show the plot
st.plotly_chart(fig)

st.write("<br>" * 1, unsafe_allow_html=True)

df = data2.drop(['wine type'], axis = 1)
df = df.drop(['wine'], axis = 1)

# Define a custom color map for quality
color_map = {
    3: '#0E078B',  
    4: '#5D02A5',
    5: '#8F2698',
    6: '#BE5377',  
    7: '#DE805B',
    8: '#F1B84E',
    9: '#F4FB58'   
}

# Melt the DataFrame to long format
df_melted = df.melt(id_vars='quality', var_name='attribute', value_name='value')

# Create subplots
num_attributes = df_melted['attribute'].nunique()
rows = (num_attributes // 4) + (num_attributes % 4 > 0)
fig = make_subplots(rows=rows, cols=4, subplot_titles=df_melted['attribute'].unique(), vertical_spacing=0.1)

# Add individual box plots to the subplots
for i, attribute in enumerate(df_melted['attribute'].unique()):
    row = i // 4 + 1
    col = i % 4 + 1
    
    # Create a box plot for the current attribute
    fig_box = px.box(df_melted[df_melted['attribute'] == attribute], 
                     x='quality', y='value', 
                     color='quality', 
                     color_discrete_map=color_map, 
                     template='plotly_dark')
    
    # Add traces from the box plot to the subplot
    for trace in fig_box['data']:
        fig.add_trace(trace, row=row, col=col)

# Define the tick values and tick labels for the x-axis
quality_values = df['quality'].unique()
quality_values.sort()
tickvals = list(quality_values)
ticktext = [str(val) for val in tickvals]

# Update layout to ensure all tick labels show up on x-axis
for i in range(1, rows + 1):
    for j in range(1, 5):
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, row=i, col=j)
        fig.update_xaxes(title_font=dict(family='Arial', size=12, color='black'), row=i, col=j)
        fig.update_yaxes(title_font=dict(family='Arial', size=12, color='black'), row=i, col=j)

# Update subplot titles to use Arial font
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(family='Arial', size=12, color='black')

fig.update_layout(
    title_text='White Wine: Distribution of Attributes by Quality',
    title_font=dict(family='Arial', size=16, color='black'),
    showlegend=False,
    height=800,
    width=1500
)

# Show the plot
st.plotly_chart(fig)




###########################
########### 1C ############
st.write("<br>" * 2, unsafe_allow_html=True)
st.subheader('1C. Predicting Quality Values')

df = pd.concat([data1, data2], ignore_index=True)

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

fig1.update_traces(unselected=dict(line=dict(color='white')))

fig1.update_layout(
    font=dict(
        family="Arial, sans-serif",
        size=14,
        color="black"
    ),
    #paper_bgcolor='rgb(30, 30, 30)',
    #plot_bgcolor='rgb(30, 30, 30)',
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

fig2.update_traces(unselected=dict(line=dict(color='white')))

fig2.update_layout(
    font=dict(
        family="Arial, sans-serif",
        size=14,
        color="black"
    ),
    #paper_bgcolor='rgb(30, 30, 30)',
    #plot_bgcolor='rgb(30, 30, 30)',
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