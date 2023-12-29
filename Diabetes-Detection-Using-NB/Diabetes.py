# Commented out IPython magic to ensure Python compatibility.
import numpy as np #for working with arrays
import pandas as pd #data processing, csv read
import matplotlib.pyplot as plt #to plot charts
# %matplotlib inline
import seaborn as sns #for data visualization
import warnings #avoid warnings
warnings.filterwarnings('ignore')

#load the dataset
ds = pd.read_csv('diabetes.csv')

"""# Data Handling

# Understanding the dataset, it includes :
*   head
*   shape
*   data types
*   types of columns
*   information about the data
*   summary of the data
"""

ds.head() #displays first 5 data records

ds.shape #number of rows (768) and columns (9)

ds.columns #the corresponding columns

ds.dtypes #the types of the data

ds.info() # used to get a concise summary of a DataFrame.
#It comes in handy when doing exploratory analysis of the data.
#The information contains the number of columns, column labels, column data types, memory usage, range index, and the number of cells in each column (non-null values).

ds.describe() #helps us to understand how data has been spread across the table.
# count :- the number of NoN-empty rows in a feature.
# mean :- mean value of that feature.
# std :- Standard Deviation Value of that feature.
# min :- minimum value of that feature.
# max :- maximum value of that feature.
# 25%, 50%, and 75% are the percentile/quartile of each features.

"""# DATA CLEANING :
as some of the columns consist 0 values which is impossible in the case of diabetes recognition, removal of those values is needed.

*   Drop duplicate values
*   Check null values
*   Checking for 0 value and replacing it :- It isn't medically possible for some data record to have 0 value such as Blood Pressure or Glucose levels. Hence we replace them with the mean value of that particular column.

"""

ds = ds.drop_duplicates() #dropping duplicate values if any

#checking missing values if any
ds.isnull().sum()

#checking for 0 values in 5 columns , Age & DiabetesPedigreeFunction do not have have minimum 0 value so no need to replace
print(ds[ds['BloodPressure']==0].shape[0])
print(ds[ds['Glucose']==0].shape[0])
print(ds[ds['SkinThickness']==0].shape[0])
print(ds[ds['Insulin']==0].shape[0])
print(ds[ds['BMI']==0].shape[0])

#replacing 0 values with median of that column
ds['Glucose']=ds['Glucose'].replace(0,ds['Glucose'].mean())#normal distribution
ds['BloodPressure']=ds['BloodPressure'].replace(0,ds['BloodPressure'].mean())#normal distribution
ds['SkinThickness']=ds['SkinThickness'].replace(0,ds['SkinThickness'].median())#skewed distribution
ds['Insulin']=ds['Insulin'].replace(0,ds['Insulin'].median())#skewed distribution
ds['BMI']=ds['BMI'].replace(0,ds['BMI'].median())#skewed distribution

ds.describe()

"""# DATA VISUALIZATION
1.   Count plot - to see if the data is balanced or not
2.   Histograms - to see if the data is normally distributed or skewed
3.   Box Plot - to analyze the distribution and the outliers
4.   Scatter plot - to understand the relationship between two vatiables
5.   Pair plot - to create scatter plot with all the variables


"""

sns.countplot(x='Outcome', data=ds) #data is imbalanced as no. of non-diabetic is more than no. of diabetic

#histogram for each feature
ds.hist(bins=10, figsize=(7,7))
plt.show() #only glucose and bloodpressure are normally distributed

#box plot
plt.figure(figsize=(12,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x='Glucose',data=ds)
plt.subplot(3,3,2)
sns.boxplot(x='BloodPressure',data=ds)
plt.subplot(3,3,3)
sns.boxplot(x='Insulin',data=ds)
plt.subplot(3,3,4)
sns.boxplot(x='BMI',data=ds)
plt.subplot(3,3,5)
sns.boxplot(x='Age',data=ds)
plt.subplot(3,3,6)
sns.boxplot(x='SkinThickness',data=ds)
plt.subplot(3,3,7)
sns.boxplot(x='Pregnancies',data=ds)
plt.subplot(3,3,8)
sns.boxplot(x='DiabetesPedigreeFunction',data=ds)

from pandas.plotting import scatter_matrix
scatter_matrix(ds,figsize=(20, 20));

# FEATURE COLLECTION
#pearson's correlation coefficient

dscorr = ds.corr()
sns.heatmap(dscorr, annot=True)

#as per the heatmap glucose, BMI, age are mostly correlated with the outcome and BP, Insulin, DiabetesPedigreeFunction are least correlated. so we can drop them
ds_selected = ds.drop(['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction'], axis = 'columns')

"""# Handling Outliers"""

from sklearn.preprocessing import QuantileTransformer
x=ds_selected
quantile  = QuantileTransformer()
X = quantile.fit_transform(x)
ds_new=quantile.transform(X)
ds_new=pd.DataFrame(X)
ds_new.columns =['Pregnancies', 'Glucose','SkinThickness','BMI','Age','Outcome']
ds_new.head()

plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x=ds_new['Glucose'],data=ds_new)
plt.subplot(3,3,2)
sns.boxplot(x=ds_new['BMI'],data=ds_new)
plt.subplot(3,3,3)
sns.boxplot(x=ds_new['Pregnancies'],data=ds_new)
plt.subplot(3,3,4)
sns.boxplot(x=ds_new['Age'],data=ds_new)
plt.subplot(3,3,5)
sns.boxplot(x=ds_new['SkinThickness'],data=ds_new)

"""# SPLIT THE DATA"""

trgt = 'Outcome'
y = ds_new[trgt] #given predictions - training data
X = ds_new.drop(trgt, axis = 1) #dropping the outcome column and keeping rest of the cols

X.head() #carries all the independent features

y.head() #carries the dependent feature

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape,y_train.shape

X_test.shape,y_test.shape

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc

param_grid_nb = {
    'var_smoothing': np.logspace(0,-2, num=100) #It adds a small value (smoothing factor) to the variance of each feature, effectively preventing variance from being exactly zero.
}
nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)

best_model = nbModel_grid.fit(X_train, y_train)

nb_pred = best_model.predict(X_test)

print("Classification Report:\n", pd.DataFrame(classification_report(y_test,nb_pred, output_dict = True)).transpose())
print("\n F1:\n",f1_score(y_test,nb_pred))
print("\n Precision score is:\n",precision_score(y_test,nb_pred))
print("\n Recall score is:\n",recall_score(y_test,nb_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,nb_pred))

nbConfusion = confusion_matrix(y_test, nb_pred)
nbConfusion

ylabel = ["Actual [Non-Diab]","Actual [Diab]"]
xlabel = ["Pred [Non-Diab]","Pred [Diab]"]
sns.set(font_scale = 1.5)
plt.figure(figsize=(15,6))
sns.heatmap(nbConfusion, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)

"""The ROC score ranges from 0 to 1, with 0 indicating a poor classifier (classifies all instances incorrectly) and 1 indicating a perfect classifier (classifies all instances correctly). A higher ROC score indicates better performance."""

nb_Roc_Auc = roc_auc_score(y_test,nb_pred)
print ('Roc Auc Score: ', nb_Roc_Auc)

nb_pred_prob = best_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, nb_pred_prob)

"""The ROC curve is a graphical representation of the classifier's performance at different discrimination thresholds. It plots the true positive rate (TPR or sensitivity) on the y-axis against the false positive rate (FPR or 1-specificity) on the x-axis, where:
TPR (True Positive Rate) = True Positives / (True Positives + False Negatives)
FPR (False Positive Rate) = False Positives / (False Positives + True Negatives)

A perfect classifier would have a ROC curve that passes through the top-left corner of the plot (TPR=1, FPR=0), while a random classifier would have a diagonal line from the bottom-left to the top-right (indicating equal chances of true positive and false positive rates).
"""

plt.figure(figsize=(15,6))
sns.set(font_scale = 1.5)
plt.plot(fpr, tpr)
plt.title('ROC Curve for Naive Bayes Diabetes Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
