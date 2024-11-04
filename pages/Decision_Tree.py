# import neccessary libraries
# import neccessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from google.colab import drive
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


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

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#splitting data into train and test
X = df.drop(columns = ['Addiction_Class', 'id', 'Financial_Issues',
                       'Academic_Performance_Decline', 'Withdrawal_Symptoms', 'Experimentation'])

y = df['Addiction_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

dt = DecisionTreeClassifier(criterion="gini", max_depth=3)
dt = dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy:",accuracy_dt)

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

feature_cols = X.columns


dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols, class_names=['No addict','Addict'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())

# intepret the confusion matrix
#plt.figure(figsize=(8, 6))
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', xticklabels=df['Addiction_Class'].unique(), yticklabels=df['Addiction_Class'].unique())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Decision Tree')
plt.show()

st.pyplot(fig)
