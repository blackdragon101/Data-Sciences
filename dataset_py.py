#This is a program for linear regression.
import pandas as pd
dataset=pd.read_csv('ECAT_2024.csv')
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
print(dataset)
print(dataset.iloc[3:7,2:5])
print(dataset.keys())
print(dataset.info())
print(dataset.describe())
print(dataset.iloc[:,8:13].describe())
print(dataset.duplicated().sum())
print(dataset.duplicated())

# Give all unique values
print(dataset['Name'].unique())
print(dataset['Gender'].unique())
print(dataset['10th Marks'].unique())
print(dataset['9th Marks'].unique())
print(dataset['1st Year Marks'].unique())
print(dataset['2nd Year Marks'].unique())
print(dataset['ECAT Marks'].unique())

# Plot the graphs
fig, axes = plt.subplots(2, 3,figsize=(10,7))
sb.countplot(data=dataset, x='Gender',ax=axes[0,0]).set(title='Gender')
sb.countplot(data=dataset, x='9th Marks',ax=axes[0,1]).set(title='9th Marks')
sb.countplot(data=dataset, x='10th Marks',ax=axes[0,2]).set(title='10th Marks')

sb.countplot(data=dataset, x='1st Year Marks',ax=axes[1,0]).set(title='1st Year Marks')
sb.countplot(data=dataset, x='2nd Year Marks',ax=axes[1,1]).set(title='2nd Year Marks')

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Increase the distance between the subplots
plt.show()


# replace the all charactersb with only two unique
dataset['Gender'].replace('MALE','Male', inplace=True)
dataset['Gender'].replace('M ','Male', inplace=True)
dataset['Gender'].replace(' Male','Male', inplace=True)
dataset['Gender'].replace('Male ','Male', inplace=True)
dataset['Gender'].replace('male','Male', inplace=True)
dataset['Gender'].replace(np.nan,'Male', inplace=True)
dataset['Gender'].replace('Female ','Female', inplace=True)
dataset['Gender'].replace('female ','Female', inplace=True)
dataset['Gender'].replace('female','Female', inplace=True)
dataset['Gender'].replace('F','Female', inplace=True)
dataset['9th Marks'].replace(np.nan,'470', inplace=True)
dataset['9th Marks'].replace('Nil','470', inplace=True)

print(dataset['Gender'].unique())
print(dataset['9th Marks'].unique())


print(dataset.isnull().sum())
print(dataset.shape)

dataset.drop(['Timestamp'], axis=1,inplace=True)
dataset.drop(['Registration Number'], axis=1,inplace=True)
dataset.drop(['Name'], axis=1,inplace=True)
print(dataset.shape)
# Map string values to integer
dataset['Gender']=dataset['Gender'].map({'Male':1,'Female':2})

# Convert string values in LoanAmount column to float
dataset['9th Marks'] = dataset['9th Marks'].astype(float)
dataset['10th Marks'] = dataset['10th Marks'].astype(float)
dataset['1st Year Marks'] = dataset['1st Year Marks'].astype(float)
dataset['2nd Year Marks'] = dataset['2nd Year Marks'].astype(float)

print(dataset.corr())

sb.heatmap(data=dataset.corr(), cmap="YlGnBu", annot=True)
plt.show()

print(dataset.keys())

# Step 2: Split the Data
X = dataset[['Gender', '9th Marks','10th Marks','1st Year Marks','2nd Year Marks']]
y = dataset['ECAT Marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Selection
model = LinearRegression()

# Step 4: Train the Model
model.fit(X_train, y_train)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Step 6: Prediction
input_features = [[2,505,1093,486,1025]]
predicted = model.predict(input_features)
print("Predicted Marks:", predicted)

