# import neccessary libraries
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             precision_score, recall_score, 
                             ConfusionMatrixDisplay, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.stats import randint


import streamlit as st

data1 = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/jie43203/refs/heads/main/student_addiction_dataset_test.csv')
data1.head()

st.write(data1)

data2 = pd.read_csv('https://raw.githubusercontent.com/nicaaaaaaa/jie43203/refs/heads/main/student_addiction_dataset_test.csv')
data2.head()

# merge two data frames
print('After merging:')
drugsData = pd.concat([data1, data2], axis=0)
print(drugsData)

#check missing data
a = drugsData.isnull().any()
b = drugsData.isnull().sum()
print(a)
print()
print(b)

# remove nan value
newData = drugsData.dropna()
a = newData.isnull().any()
b = newData.isnull().sum()

#check again if nan value already removed
print(a)
print()
print(b)

newData.shape

# check duplicated
newData.duplicated().sum()  #count duplicated bcs no col that unique

# perform binary encode
df = pd.DataFrame(newData)

# Mapping 'yes' and 'no' to 1 and 0 respectively
binary_map = {'Yes': 1, 'No': 0}

# Apply binary encoding to each column
for col in df.columns:
    df[col] = df[col].map(binary_map) # Fill NaNs with a placeholder (-1) or handle them as needed


# add id columns as mapping for a person to avoid pandas read as duplicated
df.insert(0, 'id', range(1, len(df) + 1))  # Inserts 'id' column at position 0

print("Original DataFrame:")
df.head()

# column id has added as unique to avoid duplicated
df.duplicated().sum()

df.info()

df['Addiction_Class'].value_counts()

# perform statistical analyisis
df.describe()

sns.countplot(x='Addiction_Class', hue='Addiction_Class', data=df, palette='Set2')
plt.title("Target Variable vs Count")
plt.legend(labels=['No Addict', 'Addict'], title='Addiction Class')

fig = plt.gcf()
st.pyplot(fig)
