# The data used is of travel insurance.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
# we have the following columns in our file
""" file["employment type,graduate,annual income,
          family memebers,chronic diseses,frequent flyer,ever travelled abbroad,
          travel insurance"]"""
# these are the features of pandas library
dataset=pd.read_csv('TravelInsurancePrediction.csv')
print(dataset)
print(dataset.iloc[3:7,2:5])
print(dataset.keys())
print(dataset.info())
print(dataset.describe())
print(dataset.iloc[:,8:13].describe())
print(dataset.duplicated().sum())
print(dataset.duplicated())

# This tells us about the unique values that exist in the data of the file
print(dataset['Employment Type'].unique())
print(dataset['GraduateOrNot'].unique())
print(dataset['AnnualIncome'].unique())
print(dataset['FamilyMembers'].unique())
print(dataset['ChronicDiseases'].unique())
print(dataset['FrequentFlyer'].unique())
print(dataset['EverTravelledAbroad'].unique())
print(dataset['TravelInsurance'].unique())

# # These are the methods to plot graphs using seaborn.
fig, axes = plt.subplots(3, 3)
sb.countplot(data=dataset, x='Employment Type',ax=axes[0,1]).set(title='Employment Type')
sb.countplot(data=dataset, x='GraduateOrNot',ax=axes[0,2]).set(title='GraduateOrNot')

sb.countplot(data=dataset, x='AnnualIncome',ax=axes[1,0]).set(title='AnnualIncome')
sb.countplot(data=dataset, x='FamilyMembers',ax=axes[1,1]).set(title='FamilyMembers')
sb.countplot(data=dataset, x='ChronicDiseases',ax=axes[1,2]).set(title='ChronicDiseases')

sb.countplot(data=dataset, x='FrequentFlyer',ax=axes[2,0]).set(title='FrequentFlyer')
sb.countplot(data=dataset, x='EverTravelledAbroad',ax=axes[2,1]).set(title='EverTravelledAbroad')
sb.countplot(data=dataset, x='TravelInsurance',ax=axes[2,2]).set(title='TravelInsurance')

plt.show()


# # # This part of code is using the numpy library.
dataset['Employment Type'].replace(np.nan,'Yes', inplace=True)
dataset['GraduateOrNot'].replace(np.nan,'0', inplace=True)
dataset['FamilyMembers'].replace(np.nan,'No', inplace=True)
dataset['AnnualIncome'].replace(np.nan,'14600', inplace=True)
dataset['ChronicDiseases'].replace(np.nan,'342', inplace=True)
dataset['FrequentFlyer'].replace(np.nan,'0', inplace=True)


print(dataset.isnull().sum())
print(dataset.shape)

dataset.drop(['AnnualIncome'], axis=1,inplace=True)
dataset.drop(['TravelInsurance'], axis=1,inplace=True)

dataset['Employment Type']=dataset['Employment Type'].map({'Government Sector':0,'Private Sector/Self Employed':1})
dataset['AnnualIncome']=dataset['GraduateOrNot'].map({'Yes':0,'No':1})
dataset['FamilyMembers']=dataset['EverTravelledAbroad'].map({'No':0,'Yes':1})
dataset['EverTravelledAbroad']=dataset['FrequentFlyer'].map({'Yes':1,'No':0})

# print(dataset.corr())

print(dataset['Age'].unique())
print(dataset['Employment Type'].unique())
print(dataset['GraduateOrNot'].unique())
print(dataset['AnnualIncome'].unique())
print(dataset['FamilyMembers'].unique())

# heatmap feature can be used only through seaborn library
# sb.heatmap(data=dataset.corr(), cmap="YlGnBu", annot=True)
# plt.show()

dataset.drop(['ChronicDiseases'], axis=1,inplace=True)
dataset.drop(['FrequentFlyer'], axis=1,inplace=True)
dataset.drop(['EverTravelledAbroad'], axis=1,inplace=True)
