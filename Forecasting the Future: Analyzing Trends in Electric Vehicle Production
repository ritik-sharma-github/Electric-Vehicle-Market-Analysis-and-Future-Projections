#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression as LR


# ### Step 1: Preparation and Analyzing the Dataset.

# In[2]:


df = pd.read_csv("C:/Ritik Sharma/VIT  2nd SEMESTER/EDA J PROJECT/Electric_Vehicle_Population_Data.csv")


# In[3]:


df


# In[4]:


df1 = df.drop(columns=['VIN (1-10)', 'City', 'State', 'DOL Vehicle ID', 'Vehicle Location', '2020 Census Tract'])


# In[5]:


df1


# In[6]:


len(df1.columns)


# In[7]:


df1.shape


# In[8]:


df1.head()


# In[9]:


# Unique values of all the columns in the data set.

cols = df1.columns
def Unique_Values():
    for i in np.arange(0,len(cols)):
        print('There are {} nos of unique values in {} column out of {}'.format(df[cols[i]].nunique(), cols[i], len(df)))
Unique_Values()


# In[10]:


df1.info()


# In[11]:


# to view the statistical values of numerical columns

df1.describe().style.background_gradient(cmap='cividis')


# ### step 2 : Handling the missing values.

# In[12]:


df1.isnull().sum()


# In[13]:


# to view the missing percentages

missing_percentages=df1.isna().sum().sort_values(ascending=False)/len(df1)
missing_percentages


# In[14]:


missing_percentages[missing_percentages !=1]


# In[15]:


# To plot the missing percentage of the columns in the data set.

missing_percentages[missing_percentages != 1].plot(kind = 'barh')

Legislative District column has the more missing values.
# In[16]:


# Visualize the missing data in percentages using basic numerical equation.

data_missing = df1.isnull().sum()*100/len(df1)
data_missing


# In[17]:


plt.figure(figsize=(20,18))
sns.heatmap(df1.isnull(), yticklabels=False, cmap='viridis')

fill the null values of County, City and Postal Code column with the mode of County Column, City Column and Postal Code column respectively. Using the mean, mode, fill, drop method in Pandas, we can fill the missing values.
# In[18]:


df1['County'].fillna(df1['County'].mode()[0], inplace=True)


# In[19]:


# Use mean of Legislative District column to fill the null values of Legislative District Column.

df1['Legislative District'].fillna(df1['Legislative District'].mean(),inplace =True)


# In[20]:


# Using the forwardfill and backwardfill method in Pandas, we can fill all the rows with missing values

df1['Electric Utility'].fillna('ffill',inplace =True)
df1['Postal Code'].fillna('bfill',inplace =True)


# In[21]:


# Assuming 'Electric Range' is the column name
df1 = df1[df1['Electric Range'] != 0]


# In[22]:


# Assuming 'Base MSRP' is the column name
df1 = df1[df1['Base MSRP'] != 0]


# In[23]:


df1.isnull().sum()


# In[24]:


df1['Electric Range'].unique()


# In[25]:


df1['Model'].unique()


# In[26]:


df1['Legislative District'].unique()


# In[27]:


df1['Make'].unique()


# In[28]:


df1["Electric Range"].replace(0, pd.NA, inplace=True)
df1["Legislative District"].replace(0, pd.NA, inplace=True)


# In[29]:


print("\nCleaned Dataset:")
df1.head()


# #### Label Encoding for categorical variables with ordinality

# In[30]:


import pandas as pd

# Assuming df1 is the cleaned dataset

# Label Encoding for categorical variables with ordinality
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
# Example: Encoding 'Electric Vehicle Type' column
df1['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] = label_encoder.fit_transform(df1['Clean Alternative Fuel Vehicle (CAFV) Eligibility'])

# Viewing the modified dataset
df1.head()


# ### Feature Engineering:

# In[31]:


import pandas as pd


# 1. Create a new feature for the age of the vehicle
current_year = 2024  # Assuming the current year
df1['Vehicle Age'] = current_year - df1['Model Year']


# Viewing the modified dataset
df1.head()


# In[32]:


# 2. Create a binary feature indicating if the vehicle is a luxury brand
luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Porsche', 'KIA', 'VOLVO', 'TESLA', 'CHRYSLER', 'Bentley']
df1['Is Luxury Brand'] = df1['Make'].isin(luxury_brands).astype(int)

# Viewing the modified dataset
df1.head()


# In[33]:


df1


# ### Outlier Detection and Handling.

# In[34]:


# Save the DataFrame to a CSV file
df1.to_csv('clean_dataset.csv', index=False)


# In[35]:


import os

# Get the current working directory (where your Python script is located)
current_directory = os.getcwd()

print("Current Directory:", current_directory)


# In[36]:


df2=pd.read_csv("C:/Ritik Sharma/VIT  2nd SEMESTER/EDA J PROJECT/clean_dataset.csv")


# In[37]:


# Visual Inspection - Histograms
df2.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Visual Inspection - Boxplots
df2.boxplot(figsize=(12, 8))
plt.xticks(rotation=30)
plt.show()


# In[38]:


import pandas as pd
import dask.dataframe as dd
import numpy as np

# Define data types for columns
dtypes = {'Legislative District': 'float64', 'Postal Code': 'object'}

# Load the dataset using Dask and specify the data types
ddf = dd.read_csv("C:/Ritik Sharma/VIT  2nd SEMESTER/EDA J PROJECT/clean_dataset.csv", dtype=dtypes)

# Select only numeric columns
numeric_cols = ddf.select_dtypes(include=np.number)

# Calculate z-scores
z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()

# Identify outliers based on z-scores
outliers_zscore = ddf[((z_scores > 3) | (z_scores < -3)).any(axis=1)]

# Compute IQR
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

# Identify outliers based on IQR
outliers_iqr = ddf[((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

# Convert Dask DataFrame to Pandas DataFrame for printing
outliers_zscore_pd = outliers_zscore.compute()
outliers_iqr_pd = outliers_iqr.compute()

print("Potential Outliers based on Z-scores:")
print(outliers_zscore_pd)


# In[39]:


print("\nPotential Outliers based on IQR:")
print(outliers_iqr_pd)


# In[40]:


# Calculate the first quartile (Q1)
Q1 = df2['Base MSRP'].quantile(0.25)

# Calculate the third quartile (Q3)
Q3 = df2['Base MSRP'].quantile(0.75)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 4.5 * IQR
upper_bound = Q3 + 4.5 * IQR

print("Lower bound for outliers:", lower_bound)
print("Upper bound for outliers:", upper_bound)


# In[41]:


# Merge outlier detection results
merged_outliers = pd.concat([outliers_zscore_pd, outliers_iqr_pd])

# Remove duplicate rows if any
merged_outliers = merged_outliers.drop_duplicates()

# Optionally, add a column indicating the method used for outlier detection
merged_outliers['Outlier Detection Method'] = np.where(merged_outliers.index.isin(outliers_zscore_pd.index), 'Z-score', 'IQR')

# Reset index
merged_outliers.reset_index(drop=True, inplace=True)

# Print the merged outlier detection results
print("Merged Outliers:")
print(merged_outliers)


# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure df is loaded properly with your dataset

# Features more likely to be useful for predicting future sales of cars
features_for_prediction = ['Model Year', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 
                          'Electric Range', 'Base MSRP']

# Outlier analysis (box plot) for numerical features
for column in features_for_prediction:
    if df[column].dtype != 'object':
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.ylabel('Values')
        plt.show()


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure df2 is loaded properly with your dataset

# Features more likely to be useful for predicting future sales of cars
features_for_prediction = ['Model Year', 
                          'Electric Range', 'Base MSRP']

for column in features_for_prediction:
    if ddf[column].dtype != 'object':
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=column, data=ddf.compute())  # Using data=ddf.compute() to convert Dask DataFrame to Pandas for plotting
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.ylabel('Values')
        plt.show()


# In[44]:


from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: KNN-based outlier detection
def knn_outlier_detection(data, k=100):
    numeric_data = data.select_dtypes(include=np.number)  # Select only numeric columns
    nbrs = NearestNeighbors(n_neighbors=k).fit(numeric_data)
    distances, _ = nbrs.kneighbors(numeric_data)
    return np.mean(distances, axis=1)

# Compute outlier scores using KNN
outlier_scores = knn_outlier_detection(df2)
outlier_scores


# In[45]:


# Step 2: Dendrogram-based outlier detection
def dendrogram_outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)  # Select only numeric columns
    Z = linkage(numeric_data, method='ward')
    plt.figure(figsize=(20, 8))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

# Perform dendrogram-based outlier detection
dendrogram_outlier_detection(df2)

# You can further analyze the dendrogram to identify potential outliers visually.


# In[46]:


# Step 1: Define KNN-based outlier detection function
def knn_outlier_detection(data, column, k=5):
    column_data = data[column].values.reshape(-1, 1)  # Reshape to 2D array
    nbrs = NearestNeighbors(n_neighbors=k).fit(column_data)
    distances, _ = nbrs.kneighbors(column_data)
    return np.mean(distances, axis=1)

# Step 2: Define dendrogram-based outlier detection function
def dendrogram_outlier_detection(data, column):
    column_data = data[column].values.reshape(-1, 1)  # Reshape to 2D array
    Z = linkage(column_data, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(f'Hierarchical Clustering Dendrogram - {column}')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

# Step 3: Perform outlier analysis and plot for each column
columns_of_interest = ['Model Year', 'Electric Range', 'Base MSRP']
for column in columns_of_interest:
    # KNN-based outlier detection
    outlier_scores = knn_outlier_detection(df2, column)
    
    # Dendrogram-based outlier detection
    dendrogram_outlier_detection(df2, column)
    
    # Optionally, plot the KNN-based outlier scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(outlier_scores)), outlier_scores, marker='o', linestyle='-', color='b')
    plt.title(f'KNN Outlier Detection - {column}')
    plt.xlabel('Sample Index')
    plt.ylabel('Outlier Score')
    plt.grid(True)
    plt.show()


# In[47]:


# Step 1: Define KNN-based outlier detection function
def knn_outlier_detection(data, column, k=5):
    nbrs = NearestNeighbors(n_neighbors=k)
    distances = nbrs.fit(data[[column]]).kneighbors()[0]
    return np.mean(distances, axis=1)

# Step 2: Define dendrogram-based outlier detection function
def dendrogram_outlier_detection(data, column):
    Z = linkage(data[[column]], method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(f'Hierarchical Clustering Dendrogram - {column}')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

# Step 3: Perform outlier analysis and plot for each column
columns_of_interest = ['Model Year', 'Electric Range', 'Base MSRP']
for column in columns_of_interest:
    # KNN-based outlier detection
    outlier_scores = knn_outlier_detection(ddf, column)
    
    # Dendrogram-based outlier detection
    dendrogram_outlier_detection(ddf, column)
    
    # Optionally, plot the KNN-based outlier scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(outlier_scores)), outlier_scores, marker='o', linestyle='-', color='b')
    plt.title(f'KNN Outlier Detection - {column}')
    plt.xlabel('Sample Index')
    plt.ylabel('Outlier Score')
    plt.grid(True)
    plt.show()


# In[48]:


# Distribution of Electric Vehicles Types.

ev_types = df2["Electric Vehicle Type"].value_counts().reset_index()
ev_types.columns = (["Electric Vehicle Type", "Number Of Vehicles"])
ev_types


# In[49]:


figure, axes = plt.subplots(1, 2, figsize=(14, 6))

figure.suptitle('Distribution Of Electric Vehicle Types', fontweight='bold', fontsize=14)

barplot = sns.barplot(ev_types, x="Electric Vehicle Type", y="Number Of Vehicles", ax=axes[0], palette="copper")
axes[0].set_xlabel('Electric Vehicle Types', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Number Of Vehicles', fontweight='bold', fontsize=11)
axes[0].set_xticklabels(barplot.get_xticklabels(), fontsize=9)

for p in barplot.patches:
    barplot.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height()), va="bottom", ha="center", fontsize=10)

sizes1 = ev_types["Number Of Vehicles"]
labels1 = ev_types["Electric Vehicle Type"]
explode1 = [.02, .02]

axes[1].pie(sizes1, labels=labels1, autopct='%0.2f%%', shadow=True, explode=explode1, startangle=130)

plt.show()


# In[50]:


# Count of Cars by Make.

plt.figure(figsize=(12,6))
sns.countplot(data=df,x='Make')
plt.xticks(rotation=90)
plt.xlabel('Make')
plt.ylabel('Count')
plt.show()


# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns

# List of features
features = ['Make', 'Model', 'Electric Vehicle Type', 
            'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 
            'Electric Range', 'Base MSRP', 
            'Legislative District', 'Electric Utility', 
            'Vehicle Age', 'Is Luxury Brand']

for feature in features:
    # Histogram
    plt.figure(figsize=(20, 6))
    plt.hist(df2[feature], bins=20, edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
    plt.show()
    
    # Bar Graph
    if df2[feature].dtype == 'object':
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df2, x=feature)
        plt.xticks(rotation=45)
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.title(f'Count of {feature}')
        plt.show()
    
    # Pie Chart
    if df2[feature].dtype == 'object':
        plt.figure(figsize=(8,24))
        df2[feature].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'Distribution of {feature}')
        plt.ylabel('')
        plt.show()
    
    # Frequency Polygon
    plt.figure(figsize=(24, 6))
    sns.histplot(df2[feature], kde=True, color='skyblue')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Frequency Polygon of {feature}')
    plt.show()
    
    # Scatter Plot
    if df2[feature].dtype != 'object':
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df2, x=feature, y='Base MSRP')
        plt.xlabel(feature)
        plt.ylabel('Base MSRP')
        plt.title(f'Scatter Plot of {feature} vs Base MSRP')
        plt.show()
    
    # Heatmap
    if df2[feature].dtype != 'object':
        plt.figure(figsize=(10, 8))
        sns.heatmap(data=df2.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()


# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns

# List of features
features = ['Make', 'Model', 'Electric Vehicle Type', 
            'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 
            'Electric Range', 'Base MSRP', 
            'Legislative District', 'Electric Utility', 
            'Vehicle Age', 'Is Luxury Brand']

# Calculate the number of rows and columns for the subplots
num_features = len(features)
num_cols = 2
num_rows = (num_features - 1) // num_cols + 1

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(35, 10*num_rows))

# Flatten the axes array to iterate over the subplots
axes = axes.flatten()

for i, feature in enumerate(features):
    # Histogram
    axes[i].hist(df2[feature], bins=20, edgecolor='black')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()


# In[53]:


import matplotlib.pyplot as plt
import seaborn as sns

# List of features
features = ['Make', 'Model', 'Electric Vehicle Type', 
            'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 
            'Electric Range', 'Base MSRP', 
            'Legislative District', 'Electric Utility', 
            'Vehicle Age', 'Is Luxury Brand']

# Calculate the number of rows and columns for the subplots
num_features = len(features)
num_cols = 2
num_rows = (num_features - 1) // num_cols + 1

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(35, 10*num_rows))

# Flatten the axes array to iterate over the subplots
axes = axes.flatten()

for i, feature in enumerate(features):
    # Frequency Polygon
    sns.histplot(df2[feature], kde=True, color='skyblue', ax=axes[i])
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Frequency Polygon of {feature}')
    axes[i].grid(True)


# Adjust layout
plt.tight_layout()
plt.show()


# In[54]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset using Dask and specify the data types
dtypes = {'County': 'object', 
          'Postal Code': 'object', 
          'Model Year': 'float64', 
          'Make': 'object', 
          'Model': 'object', 
          'Electric Vehicle Type': 'object',
          'Clean Alternative Fuel Vehicle (CAFV) Eligibility': 'float64',
          'Electric Range': 'float64',
          'Base MSRP': 'float64',
          'Legislative District': 'float64',
          'Electric Utility': 'object',
          'Vehicle Age': 'float64',
          'Is Luxury Brand': 'float64'}

ddf = dd.read_csv("C:/Ritik Sharma/VIT  2nd SEMESTER/EDA J PROJECT/clean_dataset.csv", dtype=dtypes)

# Define the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Interactive Electric Vehicle Dashboard"),
    
    # Add dropdowns for selecting features
    html.Div([
        dcc.Dropdown(
            id='x-dropdown',
            options=[
                {'label': col, 'value': col} for col in ddf.columns
            ],
            value='Electric Range',
            clearable=False,
            style={'width': '48%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='y-dropdown',
            options=[
                {'label': col, 'value': col} for col in ddf.columns
            ],
            value='Base MSRP',
            clearable=False,
            style={'width': '48%', 'float': 'right', 'display': 'inline-block'}
        )
    ]),
    
    # Add scatter plot
    dcc.Graph(id='scatter-plot'),
    
    # Add histogram
    dcc.Graph(id='histogram'),
    
    # Add pie chart
    dcc.Graph(id='pie-chart'),

    # Add bar graph
    dcc.Graph(id='bar-graph'),

    # Add box plot
    dcc.Graph(id='box-plot'),

    # Add line plot
    dcc.Graph(id='line-plot'),

    # Add area plot
    dcc.Graph(id='area-plot'),

    # Add heatmap
    dcc.Graph(id='heatmap')
])

# Define callback to update scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value')]
)
def update_scatter_plot(x_feature, y_feature):
    fig = px.scatter(ddf.compute(), x=x_feature, y=y_feature, color='Electric Vehicle Type', 
                     hover_name='Make', hover_data=['Model Year', 'Electric Range', 'Base MSRP'],
                     title=f'Scatter Plot: {y_feature} vs {x_feature}',
                     labels={x_feature: x_feature, y_feature: y_feature})
    return fig

# Define callback to update histogram
@app.callback(
    Output('histogram', 'figure'),
    [Input('x-dropdown', 'value')]
)
def update_histogram(x_feature):
    fig = px.histogram(ddf.compute(), x=x_feature, nbins=20, title=f'Histogram: {x_feature}',
                       labels={x_feature: x_feature, 'count': 'Frequency'})
    return fig

# Define callback to update pie chart
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('y-dropdown', 'value')]
)
def update_pie_chart(y_feature):
    fig = px.pie(ddf.compute(), names=y_feature, title=f'Pie Chart: Distribution of {y_feature}',
                 hole=0.3)
    return fig

# Define callback to update bar graph
@app.callback(
    Output('bar-graph', 'figure'),
    [Input('x-dropdown', 'value')]
)
def update_bar_graph(x_feature):
    bar_data = ddf[x_feature].value_counts().compute()
    fig = px.bar(x=bar_data.index, y=bar_data.values, title=f'Bar Graph: {x_feature}', 
                 labels={x_feature: x_feature, 'y': 'Count'})
    return fig



# Define callback to update box plot
@app.callback(
    Output('box-plot', 'figure'),
    [Input('x-dropdown', 'value')]
)
def update_box_plot(x_feature):
    fig = px.box(ddf.compute(), x=x_feature, title=f'Box Plot: {x_feature}', 
                 labels={x_feature: x_feature, 'y': 'Value'})
    return fig

# Define callback to update line plot
@app.callback(
    Output('line-plot', 'figure'),
    [Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value')]
)
def update_line_plot(x_feature, y_feature):
    fig = px.line(ddf.compute(), x=x_feature, y=y_feature, title=f'Line Plot: {y_feature} vs {x_feature}', 
                  labels={x_feature: x_feature, y_feature: y_feature})
    return fig

# Define callback to update area plot
@app.callback(
    Output('area-plot', 'figure'),
    [Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value')]
)
def update_area_plot(x_feature, y_feature):
    fig = px.area(ddf.compute(), x=x_feature, y=y_feature, title=f'Area Plot: {y_feature} vs {x_feature}', 
                  labels={x_feature: x_feature, y_feature: y_feature})
    return fig

# Define callback to update heatmap
@app.callback(
    Output('heatmap', 'figure'),
    [Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value')]
)
def update_heatmap(x_feature, y_feature):
    fig = px.imshow(ddf.compute().pivot_table(index=x_feature, columns=y_feature), 
                    labels={'x': x_feature, 'y': y_feature})
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[55]:


# Dimensionality Reduction.


# In[56]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming df1 contains your dataset with numeric features

# Selecting numeric features for PCA
numeric_features = ddf.select_dtypes(include=['int64', 'float64']).columns

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df1[numeric_features])

# Applying PCA
pca = PCA(n_components=4)  # Specify the number of components you want to retain
pca_result = pca.fit_transform(scaled_data)

# Creating a DataFrame to visualize the results
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4'])
print(pca_df.head())  # Display the transformed data


# In[57]:


pca_df


# In[58]:


# Plotting the PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', data=pca_df)
plt.title('PCA Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[59]:


import seaborn as sns

# Plotting pairplot for all PCA components
sns.pairplot(pca_df)
plt.suptitle('Pairplot of PCA Components', y=1.02)
plt.show()


# In[60]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = pca_df.corr()

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of PCA Components')
plt.show()


# In[61]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming df1 contains your dataset with numeric features

# Selecting numeric features for PCA
numeric_features = ddf.select_dtypes(include=['int64', 'float64']).columns

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df1[numeric_features])

# Applying PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Calculate the correlation matrix between principal components
pca_corr = pd.DataFrame(pca_result).corr()

# Display the correlation matrix
print("Correlation Matrix between Principal Components:")
pca_corr


# In[62]:


# Noise Reduction:
# Assuming pca_result contains the result of PCA transformation

# Visualize the explained variance ratio
import matplotlib.pyplot as plt

plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Number of Components')
plt.show()


# In[63]:


# visualization
# Assuming pca_result contains the result of PCA transformation

# Plot PCA components in 2D or 3D
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization (2D)')
plt.show()

# For 3D visualization
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.5)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA Visualization (3D)')
plt.show()


# In[64]:


# Feature Selection

Filter Method (Correlation Coefficient and Chi-square Test):
# In[65]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Assuming df2 is your DataFrame
df2 = pd.read_csv("C:/Ritik Sharma/VIT  2nd SEMESTER/EDA J PROJECT/clean_dataset.csv")  # You should replace "your_dataset.csv" with your actual file path if loading from a CSV file

# Convert categorical variables into numerical labels
label_encoder = LabelEncoder()
df2['Electric Vehicle Type'] = label_encoder.fit_transform(df2['Electric Vehicle Type'])
df2['Make'] = label_encoder.fit_transform(df2['Make'])

# Select features and target
X = df2[['Model Year', 'Make', 'Electric Range', 'Base MSRP', 'Legislative District']]
y = df2['Clean Alternative Fuel Vehicle (CAFV) Eligibility']


# In[66]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SelectKBest with f_classif scoring function (ANOVA F-value between label/feature for classification tasks)
k_best = SelectKBest(score_func=f_classif, k=3)

# Fit SelectKBest to training data and transform it
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using selected features
knn.fit(X_train_kbest, y_train)

# Make predictions on the test data using selected features
y_pred = knn.predict(X_test_kbest)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with selected features:", accuracy)
print("Selected Features:", X.columns[k_best.get_support()])


# In[67]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

# Assuming df2 is your DataFrame
df2 = pd.read_csv("C:/Ritik Sharma/VIT  2nd SEMESTER/EDA J PROJECT/clean_dataset.csv")  # You should replace "your_dataset.csv" with your actual file path if loading from a CSV file

# One-hot encode the 'Make' column
df2 = pd.get_dummies(df2, columns=['Make'])

# Select features
X = df2[['Model Year', 'Electric Range', 'Base MSRP', 'Legislative District']]

# Calculate the distances between each pair of samples
Z = hierarchy.linkage(X, 'ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dn = hierarchy.dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()


# In[68]:


import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# Assuming df2 is your DataFrame containing the dataset

# Drop any non-numeric columns or handle them separately
numeric_columns = df2.select_dtypes(include=['number']).columns
X_numeric = df2[numeric_columns]

# Encode categorical variables
categorical_columns = df2.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    df2[col] = le.fit_transform(df2[col])

# Define your features (independent variables) and target variable
# For demonstration purposes, let's assume the target variable is 'Clean Alternative Fuel Vehicle (CAFV) Eligibility'
X = df2.drop(columns=['Clean Alternative Fuel Vehicle (CAFV) Eligibility'])
y = df2['Clean Alternative Fuel Vehicle (CAFV) Eligibility']

# Apply chi-square feature selection
best_features = SelectKBest(score_func=chi2, k='all')
fit = best_features.fit(X, y)

# Summarize scores
scores = pd.DataFrame(fit.scores_)
columns = pd.DataFrame(X.columns)
feature_scores = pd.concat([columns, scores], axis=1)
feature_scores.columns = ['Feature', 'Score']
print(feature_scores)

# Get selected features
selected_features = fit.transform(X)
print(selected_features.shape)

The feature 'County' has a chi-square score of 31.541272.
The feature 'Model' has a much higher score of 1265.951520.
The feature 'Base MSRP' has the highest score of 464565.444786.

These scores indicate the strength of the relationship between each feature and the target variable, as measured by the chi-square test. Higher scores suggest a stronger association with the target variable.


Based on the provided result, it appears that chi-square scores have been calculated for each feature in the dataset. Here's the interpretation:

1. Feature: This column lists the names of the features (independent variables) in the dataset.

2. Score: This column represents the chi-square scores calculated for each feature. The higher the chi-square score, the more significant the association between the feature and the target variable ('Clean Alternative Fuel Vehicle (CAFV) Eligibility'). Higher scores suggest that the feature is more relevant or informative for predicting the target variable.

3. (3363, 12): This indicates the shape of the dataset after feature selection. It means that after applying chi-square feature selection, the dataset now consists of 3363 samples (rows) and 12 selected features (columns).

In summary, the representation provides insights into the association between each feature and the target variable, helping in identifying important features for further analysis and modeling.It seems that each feature has been individually selected, and a simple linear regression model has been trained using only that feature. Then, the root mean squared error (RMSE) has been calculated for each model on a test dataset. Here's the interpretation:

Selected feature: This represents the feature that was selected for training the linear regression model.

Test RMSE: This represents the root mean squared error of the model when using only the selected feature for prediction. RMSE is a measure of how well the model performs in predicting the target variable. Lower RMSE values indicate better performance, meaning the model's predictions are closer to the actual target values.

The output provides insights into the performance of individual features when used in isolation for predicting the target variable. It helps assess the predictive power of each feature and identify which features contribute the most to reducing prediction error.
# In[69]:


import pandas as pd

# Assuming df2 is your DataFrame containing the dataset
# You may need to preprocess the data to convert categorical variables to numerical format if needed

# Define your features (independent variables) and target variable
# For demonstration purposes, let's assume the target variable is 'Clean Alternative Fuel Vehicle (CAFV) Eligibility'
X = df2.drop(columns=['Clean Alternative Fuel Vehicle (CAFV) Eligibility'])
y = df2['Clean Alternative Fuel Vehicle (CAFV) Eligibility']

# Calculate Pearson correlation coefficients between features and target variable
correlation_values = X.corrwith(y)

# Convert correlation values to DataFrame for better organization
correlation_df = pd.DataFrame({'Feature': correlation_values.index, 'Correlation with Target': correlation_values.values})

# Sort the DataFrame by absolute correlation values in descending order
correlation_df['Absolute Correlation'] = correlation_df['Correlation with Target'].abs()  # Absolute values for sorting
correlation_df.sort_values(by='Absolute Correlation', ascending=False, inplace=True)

# Print the correlation results
print(correlation_df)


# In[70]:


pip install tabulate


# In[71]:


import pandas as pd
from tabulate import tabulate

# Assuming df2 is your DataFrame containing the dataset
# Define your features (independent variables) and target variable
X = df2.drop(columns=['Vehicle Age', 'Is Luxury Brand', 'Electric Range'])  # Remove target variables from features
target_variables = ['Vehicle Age', 'Is Luxury Brand', 'Electric Range']  # List of target variables

correlation_tables = {}

for target_var in target_variables:
    correlation_values = X.corrwith(df2[target_var])
    correlation_df = pd.DataFrame({'Feature': correlation_values.index, 'Correlation with ' + target_var: correlation_values.values})
    correlation_df['Absolute Correlation'] = correlation_df['Correlation with ' + target_var].abs()
    correlation_df.sort_values(by='Absolute Correlation', ascending=False, inplace=True)
    correlation_tables[target_var] = correlation_df

# Print correlation tables for different target variables
for target_var, correlation_df in correlation_tables.items():
    print(f"Correlation table for {target_var}:")
    print(tabulate(correlation_df, headers='keys', tablefmt='grid'))
    print("\n")

Absolute Correlation: The absolute value of the correlation coefficient, which is used for sorting the features. This helps identify features with the highest magnitude of correlation with the target variable, regardless of the direction of the correlation.
# In[72]:


# Assuming X is your DataFrame containing all the features
correlation_matrix = df2.corr()

# Print correlation matrix in table form
print("Correlation Matrix:")
display(correlation_matrix.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2))

Wrapper Method (Forward Selection and Backward Elimination)Forward Selection with Simple Linear Regression:
# In[73]:


import statsmodels.api as sm

def forward_selection_simple_linear(X, y, significance_level=0.05):
    selected_features = []
    num_features = X.shape[1]
    for i in range(num_features):
        remaining_features = [feature for feature in X.columns if feature not in selected_features]
        p_values = []
        for feature in remaining_features:
            X_temp = X[selected_features + [feature]]
            X_temp = sm.add_constant(X_temp)
            model = sm.OLS(y, X_temp).fit()
            p_value = model.pvalues[feature]
            p_values.append((feature, p_value))
        best_feature, best_p_value = min(p_values, key=lambda x: x[1])
        if best_p_value < significance_level:
            selected_features.append(best_feature)
        else:
            break
    return selected_features

# Example usage:
selected_features_forward = forward_selection_simple_linear(X, y)
print("Selected features (Forward Selection with Simple Linear Regression):", selected_features_forward)


# In[74]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming df2 is your DataFrame containing the dataset
# Define your features (independent variables) and target variable
X = df2[['Base MSRP', 'Electric Range', 'Electric Utility']]  # Features
y = df2['Vehicle Age']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize variables to store the best feature and its corresponding MSE
best_feature = None
lowest_mse = float('inf')

# Loop through each feature and calculate MSE
for feature in X.columns:
    # Fit linear regression model using the current feature
    model = LinearRegression()
    model.fit(X_train[[feature]], y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test[[feature]])
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Feature: {feature}")
    print(f"MSE: {mse}\n")

    # Check if the current MSE is the lowest encountered so far
    if mse < lowest_mse:
        lowest_mse = mse
        best_feature = feature

# Print the best feature and its corresponding MSE
print("Best Feature:", best_feature)
print("MSE:", lowest_mse)


# In[75]:


import itertools

# Initialize variables to store the best features and its corresponding MSE
best_features = None
lowest_mse = float('inf')

# Loop through each combination of two features and calculate MSE
for feature_pair in itertools.combinations(X.columns, 2):
    # Fit linear regression model using the current features
    model = LinearRegression()
    model.fit(X_train[list(feature_pair)], y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test[list(feature_pair)])
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Features: {feature_pair}")
    print(f"MSE: {mse}\n")

    # Check if the current MSE is the lowest encountered so far
    if mse < lowest_mse:
        lowest_mse = mse
        best_features = feature_pair

# Print the best features and their corresponding MSE
print("Best Features:", best_features)
print("MSE:", lowest_mse)


# In[76]:


import itertools

# Initialize variables to store the best features and corresponding MSE
best_features = None
lowest_mse = float('inf')

# Loop through combinations of three features and calculate MSE
for features in itertools.combinations(X.columns, 3):
    # Fit linear regression model using the current set of features
    model = LinearRegression()
    model.fit(X_train[list(features)], y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test[list(features)])
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Features: {features}")
    print(f"MSE: {mse}\n")

    # Check if the current MSE is the lowest encountered so far
    if mse < lowest_mse:
        lowest_mse = mse
        best_features = features

# Print the best features and their corresponding MSE
print("Best Features:", best_features)
print("MSE:", lowest_mse)

Backward Elimination
# In[77]:


import statsmodels.api as sm

# Assuming X_train and y_train are your training features and target variable

# Step 1: Start with all features included
selected_features = X_train.columns.tolist()

# Step 6: Stopping criterion (e.g., continue until a predefined number of features is reached)
while len(selected_features) > 0:
    # Step 2: Fit a model using all the features
    X_train_with_const = sm.add_constant(X_train[selected_features])  # Add a constant for the intercept
    model = sm.OLS(y_train, X_train_with_const)
    results = model.fit()
    
    # Step 3: Determine the least significant feature
    p_values = results.pvalues[1:]  # Exclude the constant
    least_significant_feature = p_values.idxmax()  # Find the feature with the highest p-value
    
    # Step 4: Remove the least significant feature
    if p_values.max() > 0.05:  # Stopping criterion based on p-value (adjust as needed)
        selected_features.remove(least_significant_feature)
    else:
        break

# Step 5: Evaluate the model with the remaining features
X_train_with_const = sm.add_constant(X_train[selected_features])  # Add a constant for the intercept
final_model = sm.OLS(y_train, X_train_with_const)
final_results = final_model.fit()

# Print the final selected features and model summary
print("Selected Features:", selected_features)
print(final_results.summary())

Bayesian Neural Network with the help of Bayesian estiamator.
# In[78]:


import numpy as np

# Calculate the mean of non-zero values in 'Base MSRP'
mean_base_msrp = df2.loc[df2['Base MSRP'] != 0, 'Base MSRP'].mean()

# Replace 0 values with the mean
df2['Base MSRP'].replace(0, mean_base_msrp, inplace=True)


# In[79]:


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from sklearn.preprocessing import LabelEncoder

# DataFrame columns
columns = ['County', 'Postal Code', 'Model Year', 'Make', 'Model',
           'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility',
           'Electric Range', 'Base MSRP', 'Legislative District', 'Electric Utility']

# Create an empty DataFrame to store encoded data
encoded_df = pd.DataFrame()

# Initialize Bayesian model
model = BayesianModel()

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Loop through columns
for column in columns:
    # Check if the column is categorical
    if df[column].dtype == 'object':
        # Encode the categorical column
        encoded_df[column] = label_encoder.fit_transform(df[column])
        # Add the edge to the Bayesian model (assuming the first column is the parent)
        if encoded_df.columns[-1] != encoded_df.columns[0]:
            model.add_edge(encoded_df.columns[0], encoded_df.columns[-1])

# Fit the Bayesian network
model.fit(encoded_df, estimator=BayesianEstimator)

# Print the structure of the Bayesian network
print("Bayesian Network Structure:")
print(model.edges())


# In[80]:


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianModel

# Assuming df is your DataFrame containing the dataset
# Define your features (independent variables) and target variable
X = df[['County', 'Postal Code', 'Model Year', 'Make', 'Model', 'Electric Vehicle Type', 'Electric Range', 'Base MSRP', 'Legislative District', 'Electric Utility']]
y = df[['Clean Alternative Fuel Vehicle (CAFV) Eligibility']]  # Target variables

# Initialize Bayesian model
model = BayesianModel()

# Define the structure of the Bayesian network based on prior knowledge or assumptions
# For example, let's assume dependencies between variables as follows:
# County -> Clean Alternative Fuel Vehicle (CAFV) Eligibility
# Model Year -> Electric Vehicle Type
# Electric Range -> Electric Vehicle Type
# Base MSRP -> Electric Vehicle Type
# Legislative District -> Clean Alternative Fuel Vehicle (CAFV) Eligibility
# Electric Utility -> Electric Vehicle Type
# Note: This structure should be defined based on domain knowledge or data analysis.
model.add_edges_from([('County', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility'),
                      ('Model Year', 'Electric Vehicle Type'),
                      ('Electric Range', 'Electric Vehicle Type'),
                      ('Base MSRP', 'Electric Vehicle Type'),
                      ('Legislative District', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility'),
                      ('Electric Utility', 'Electric Vehicle Type')])

# Draw the Bayesian network
plt.figure(figsize=(12, 8))
try:
    pos = nx.shell_layout(model)
    nx.draw(model, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=12, arrows=False)
    plt.title("Bayesian Network")
    plt.show()
except Exception as e:
    print("An error occurred while drawing the network:", e)

Parallel coordinate plot.
# In[81]:


import pandas as pd
import matplotlib.pyplot as plt

# Select the columns for the parallel coordinates plot
columns = ['Postal Code', 'Model Year', 'Model', 'Electric Vehicle Type','Electric Range', 'Base MSRP', 'Legislative District', 'Electric Utility']

# Draw the parallel coordinates plot
plt.figure(figsize=(12, 8))
pd.plotting.parallel_coordinates(df2[columns], 'Electric Vehicle Type')  # Corrected the column name to 'Electric Range'
plt.title('Parallel Coordinates Plot')
plt.xlabel('Features')
plt.ylabel('Values')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()

Prediction of the total number of Electric Cars produced in the next year.Lets create a variable with 'aggfunc' function to get the total number of electric cars produced each year.

Since our first objective is to predict the number of electric cars produced in the future, we will use the 'Model Year' column of the dataset and by adding all the matching values we create a dataframe with the 'total_electric' column.
# In[82]:


total_electric = df.pivot_table(columns=['Model Year'], aggfunc='size')
print(total_electric)


# In[83]:


Ncars = total_electric.reset_index()
Ncars = Ncars.rename(columns={"Model Year": "year", 0: "total_cars"})


# In[84]:


# Drop 2024 cause there are not enough values
Ncars = Ncars.drop([21])
Ncars


# In[85]:


#Store columns in x and y variables

x = Ncars["year"]
y = Ncars["total_cars"]

OldCars = Ncars.iloc[:11, :]
NewCars = Ncars.iloc[11:, :]

x1 = OldCars["year"]
y1 = OldCars["total_cars"]

x2 = NewCars["year"]
y2 = NewCars["total_cars"]


# In[86]:


fig, ax = plt.subplots()

ax.plot(x1, y1, linewidth=2.0)

plt.show()


# In[87]:


fig, ax = plt.subplots()

ax.plot(x2, y2, linewidth=2.0)

plt.show()


# In[88]:


fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)

plt.show()

When we fit the linear regression model to our data, we see that it is not capable of predicting the exponential growth of this technology, so we will have to use another method for our predictions.

This is clearly shown at the point of 2023, which is very far from our linear regression.
# In[89]:


sns.lmplot(x="year", 
           y="total_cars",
          data = NewCars)

Transform to numpy and reshape data to feed out function later.
# In[90]:


X_train = x.to_numpy()
y_train = y.to_numpy()


# In[91]:


X_train= X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

Create our polynomial function with a 'degree' value to adjust our bias
# In[92]:


def polynomial_function(degree):
    
    x = X_train
    
    y = y_train

    df = pd.DataFrame({'x':[x], 'y':[y]}).T
    df.columns = [''] * len(df.columns)
    display(df)
    
    return x, y

Create the fit function for our regression.
# In[93]:


def fit(x, y, degree=2):
    
    x = np.array(x).reshape(-1, 1)

    poly = PolynomialFeatures(degree)
    poly_features = poly.fit_transform(x.reshape(-1, 1))

    lr = LR()
    lr.fit(poly_features, y)
    y_pred = lr.predict(poly_features)
    plt.plot(x, y_pred)
    plt.scatter(x, y)
    plt.title(f'Degree = {degree}')
    plt.show()

Overfitting and Underfitting

If the degree is equal to 1 it will behave as a linear regression would normally and it clearly shows underfitting.
# In[94]:


x, y = polynomial_function(degree=2)
fit(x, y, degree=1)

If the degree is equal to 3 it will try to memorize the parametres too much causing overfitting.
# In[95]:


x, y = polynomial_function(degree=2)
fit(x, y, degree=3)

If the degree is equal to 2 it will behave as we wish in this case and will be able to predict future values.
# In[96]:


x, y = polynomial_function(degree=2)
fit(x, y, degree=2)


# In[97]:


# "Determining the Optimal Degree for Polynomial Regression". 
# the main purpose of the code, which is to find the best degree for a 
# Polynomial Regression model that minimizes the Mean Squared Error (MSE) 
# on the validation set.

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

min_mse = float('inf')
best_degree = 0

# Try degrees from 1 to 10
for degree in range(1, 11):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_val_poly = poly.transform(X_val.reshape(-1, 1))
    
    lr = LR()
    lr.fit(X_train_poly, y_train)
    
    y_pred = lr.predict(X_val_poly)
    mse = mean_squared_error(y_val, y_pred)
    
    if mse < min_mse:
        min_mse = mse
        best_degree = degree

print(f'The best degree is {best_degree} with MSE: {min_mse}')


# In[98]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Create a PolynomialFeatures object with degree 2
poly = PolynomialFeatures(degree=2)

# Transform the features to higher degree features.
X_train_transform = poly.fit_transform(X_train)

# Fit the transformed features to Linear Regression
poly_model = LinearRegression()
poly_model.fit(X_train_transform, y_train)

# Predicting on training data-set
y_train_predicted = poly_model.predict(X_train_transform)

# Predict the number of cars for future years
future_years = np.array([2024, 2025, 2026]).reshape(-1, 1)
future_years_transform = poly.fit_transform(future_years)
predictions = poly_model.predict(future_years_transform)

# Print the predicted number of cars
for year, prediction in zip(future_years, predictions):
    print(f"The predicted number of cars in the year {year} is {prediction}")


# In[99]:


# Train the model as before
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_train_transform = poly.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_train_transform, y_train)

# Ask the user for the year they're interested in
year = input("Enter the year you want to predict for: ")

# Make sure the year is in the correct format
future_year = np.array(int(year)).reshape(-1, 1)
future_year_transform = poly.fit_transform(future_year)

# Predict the number of cars for the entered year
prediction = poly_model.predict(future_year_transform)

print(f"The predicted number of cars in the year {year} is {prediction[0]}")


# In[100]:


# Import necessary libraries
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'Ncars' is your existing DataFrame and 'total_cars' is your label
y = Ncars['total_cars']

# Fit ARIMA model
model = ARIMA(y, order=(5,1,0))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

# Density plot of residuals
residuals.plot(kind='kde')
plt.show()

# Print residuals
print(residuals.describe())

# Predict the future values
n_periods = 10  # for example, predict the next 3 periods
fc = model_fit.forecast(steps=n_periods)  # 95% conf

# Create a pandas series with the forecasted values
fc_series = pd.Series(fc)

# Plot the original data and the forecasted data
plt.figure(figsize=(10,8), dpi=1000)
plt.plot(Ncars.index, y, label='original')
plt.plot(range(Ncars.index[-1]+1, Ncars.index[-1] + n_periods + 1), fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.xlabel('Year')
plt.ylabel('Total Cars')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[ ]:





# In[ ]:




