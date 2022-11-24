# Overview

According to the World Health Organization (WHO) stroke is the 2nd leading  cause of death globally, responsible for approximately 11% of total deaths. This  dataset is used to predict whether a patient is likely to get stroke based on the  input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient. 

In our project we want to predict stroke using machine learning classification algorithms, evaluate and compare their results. We did the following tasks: 
* Performance Comparison using Machine Learning Classification Algorithms on a Stroke Prediction dataset.
* using visualization libraries, ploted various plots like pie chart, count plot, curves, etc. 
* Used various Data Preprocessing techniques.
* Handle class imbalanced.
* Build various machine learning models
* Optimized SVM and Random Forest Classifiers using RandomizedSearchCV to reach the best model. 

**Domain**: Machine Learning, Data Science.

## Installing software and files
To do the project, we need to install some softwares and files. In this regard, we will be doing all the implementations in Python language on jupyter notebook. To install jupyter notebook and launch other application and files at first we have to download Anaconda which is free.

Link to Download Anaconda : https://www.anaconda.com/?modal=nucleus-commercial

Guideline for installing Anaconda : https://www.geeksforgeeks.org/how-to-install-anaconda-on-windows/

Once Anaconda is downloaded and installed successfully, we may proceed to download Jupyter notebook.

## Download and Install Jupyter Notebook
Link to download Jupyter using Anaconda : https://docs.anaconda.com/ae-notebooks/4.3.1/user-guide/basic-tasks/apps/jupyter/

More informations : https://mas-dse.github.io/startup/anaconda-windows-install/

Guideline to use Jupyter notebook : https://www.dataquest.io/blog/jupyter-notebook-tutorial/

## Using Google Colaboratory
For implementing the project with no configuration we can use Google Colaboratory as well.

## Installing Python libraries and packages
The required python libraries and packages are,
- pandas
- Numpy
- sklearn
- matplotlib
- seaborn

# Features of the Dataset
Dataset contains 5111 rows. Each row in the data provides relevant information about the patient. 
* gender: "Male", "Female" or "Other"
* age: age of the patient
* hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
* heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
* ever_married: "No" or "Yes"
* work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
* Residence_type: "Rural" or "Urban"
* avg_glucose_level: average glucose level in blood
* bmi: body mass index
* smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
* stroke: 1 if the patient had a stroke or 0 if not 

# Data Preprocessing
The data was cleaned to make it usable for the model. The following changes were made:

## Handling Missing Values - replaced the null values by median using Sklearn Simple Imputer.

```Python
from sklearn.impute import SimpleImputer

si_X_train = pd.DataFrame() # create a new dataframe to save the train dataset
si_X_test = pd.DataFrame() # create a new dataframe to save the test dataset

for column in X_train.columns:
  if (is_string_dtype(X_train[column].dtype)):
    si = SimpleImputer(strategy='most_frequent')
  else:
    si = SimpleImputer(strategy='median')
  si.fit(X_train[[column]])
  si_X_train[column] = si.transform(X_train[[column]]).flatten() # Flatten 2D matrix to 1D 
  si_X_test[column] = si.transform(X_test[[column]]).flatten()
```

## Handling Text Features - converted the text features into numeric value using LabelEncoder from Sklearn.

```python
categorical_features = []
for col in data.columns:
  if col=='Class':
    continue
  if is_string_dtype(data[col].dtype):
    categorical_features.append(col)
    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()   

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

l_X_train = pd.DataFrame() # Train dataset --> before scaling
l_X_test = pd.DataFrame() # Test dataset --> before scaling

# Convert the text features

for column in X_train.columns:
  if column in categorical_features:
    l_X_train[column] = le.fit_transform(si_X_train[column])
    l_X_test[column] = le.transform(si_X_test[column])
  else:
    l_X_train[column] = si_X_train[column].copy()
    l_X_test[column] = si_X_test[column].copy()
    
```

## Oversampling the dataset - increase the number of positive samples, by using RandomOverSampler from imblearn.

```python
from imblearn.over_sampling import RandomOverSampler

os=RandomOverSampler(0.75) # 75%
l_X_train_ns,y_train_ns = os.fit_resample(l_X_train,y_train)

print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))
```

## Feature Scaling - Standardization

```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

l_X_train_ns = ss.fit_transform(l_X_train_ns)
l_X_test = ss.transform(l_X_test)
```

# Exploratory Data Analysis (EDA)
At first, using visualization libraries, we did some data visualizations by plotting various plots like pie chart, count plot, curves, etc. in order to understand the dataset better, and to find out the correlation between the attributes.. Below are a few highlights. 

## Count Plot - Worktype

```python
sns.set(rc={'figure.figsize':(8,6)})
ax = sns.countplot(data=data, x="work_type")
plt.show()
```

<img src="https://user-images.githubusercontent.com/64092765/153050389-3dc5592f-3124-4abd-b615-19dff7f2160b.png" width="50%">

## Proportion of Different Smoking Categories among Stroke Population

<img src="https://user-images.githubusercontent.com/64092765/153050403-797bbcc7-fcf0-4830-933e-573beb29b2aa.png" width="50%">

## Finding correlation to class variable using Heatmap

```python
plt.figure(figsize=(16,8))
sns.heatmap(data.corr(),cmap="Greens");
```

<img src="https://user-images.githubusercontent.com/64092765/153050409-4de973ab-0d3a-4ef1-b4de-670248439bbb.png" width="50%">

## No Stroke vs Stroke by BMI

```python
plt.figure(figsize=(12,10))

sns.distplot(data[data['stroke'] == 0]["bmi"], color='green') # No Stroke - green
sns.distplot(data[data['stroke'] == 1]["bmi"], color='red') # Stroke - Red

plt.title('No Stroke vs Stroke by BMI', fontsize=15)
plt.xlim([10,100])
plt.show()
```

<img src="https://user-images.githubusercontent.com/64092765/153050416-aff6dcb1-a406-417d-b060-dbd5056c1177.png" width="50%">

## Catplot - Heart disease

```python
sns.catplot(x="heart_disease", y="stroke", hue='smoking_status', kind="bar", data=data);
```

<img src="https://user-images.githubusercontent.com/64092765/153050427-1b13cf20-49fd-437f-991c-548ad32a65e9.png" width="50%">

# Model Building- Machine Learning Models 

The categorical variables were transformed into dummy variables. Dataset was split into train and tests sets with a test size of 20%.   
After our dataset was finally ready, we have used some machine learning classification algorithms on this dataset and observed their performances. 

The different models used are:
* Logistic Regression 
* Naive Bayes
* k Nearest Neighbors
* 
* Random Forest Classifier

## Support Vector Machine – Gaussian SVM

```Python
from sklearn.svm import SVC
svc = SVC(kernel='rbf',random_state=0)
svc.fit(l_X_train_ns,y_train_ns)

y_pred = svc.predict(l_X_test)
model_metrics = evaluate_preds(y_test, y_pred)
```

## Naive Bayes 

```Python
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(l_X_train_ns,y_train_ns)

y_pred = naive.predict(l_X_test)
model_metrics = evaluate_preds(y_test, y_pred)
```

## Logistic Regression

```Python
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
logistic.fit(l_X_train_ns,y_train_ns)

y_pred = logistic.predict(l_X_test)
model_metrics = evaluate_preds(y_test, y_pred)
```

## k Nearest Neighbours

```Python
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=40)
neigh.fit(l_X_train_ns,y_train_ns)

y_pred = neigh.predict(l_X_test)
model_metrics = evaluate_preds(y_test, y_pred)
```

## RandomForestClassifier

```Python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='entropy')
rf.fit(l_X_train_ns,y_train_ns)

y_pred = rf.predict(l_X_test)
model_metrics = evaluate_preds(y_test, y_pred)
```

# Model performance

## Classification Evaluation Metrics

We then compared these results based on various classification metrics. 
The metrics are: accuracy, precision, recall, f1 score and mcc score.

```Python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def evaluate_preds(y_test,y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred) 
    f1 = f1_score(y_test,y_pred)
    mcc = matthews_corrcoef(y_test,y_pred)

    metric_dict = {
        "accuracy":round(accuracy,2),
        "precision":round(precision,2),
        "recall":round(recall,2),
        "f1":round(f1,2),
        "mcc": mcc 
    } # A dictionary that stores the results of the evaluation metrics
    
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    print(f'MCC Score: {mcc:.2f}')
    
    return metric_dict
```

## Results

<img src="https://user-images.githubusercontent.com/64092765/153050322-d2dca138-58d4-4bd1-9ac4-5927983b895a.png" width="75%">








