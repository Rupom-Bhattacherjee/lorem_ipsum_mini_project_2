# lorem_ipsum_mini_project_2
 # Term Deposit Opening Decision
 ### AGENDA
- DataSet 
    - DataSet Information
    - Attribute Information
- Descriptive Analysis
    - Visualizing
    - Correlations
- Data Preprocessing
    - Null-Missing Value Analysis
    - Encoding Categorical Variables
    - Oversampling data
    - Data Normalization
    - Principal Component Analysis
- Predictive Analysis
    - Logistic Regression (Base Model)
    - Logistic Regression with PCA and data oversampling
    - Random Forest
    - KNN
- Summary
### Dataset Information

- The data is related with direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls.

- DataSet Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

- The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

### Attribute Information

    1 - Age (numeric)
    2 - Job : type of job (categorical)
    3 - Marital : marital status (categorical)
    4 - Education (categorical)
    5 - Default: has credit in default? (categorical)
    6 - Housing: has housing loan? (categorical)
    7 - Loan: has personal loan? (categorical)

#### related with the last contact of the current campaign:
    8 - Contact: contact communication type (categorical)
    9 - Month: last contact month of year (categorical)
    10 - Day_of_week: last contact day of the week (categorical)
    11 - Duration: last contact duration, in seconds (numeric)

#### other attributes:
    12 - Campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    13 - Pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    14 - Previous: number of contacts performed before this campaign and for this client (numeric)
    15 - Poutcome: outcome of the previous marketing campaign (categorical)

#### social and economic context attributes
    16 - Emp.var.rate: employment variation rate - quarterly indicator (numeric)
    17 - Cons.price.idx: consumer price index - monthly indicator (numeric) 
    18 - Cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
    19 - Euribor3m: euribor 3 month rate - daily indicator (numeric)
    20 - Nr.employed: number of employees - quarterly indicator (numeric)
    
### EDA

- The regular Python language is used for EDA. 
- We have two types of variables in our data set. These are Continuous Variables and Categorical Variables. 

#### Continuous Variables

- 'age‘ , 'duration', 'campaign', ‘pdays ‘, 'previous', 'emp.var.rate‘
'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'

##### Summary stat of Continuous Variables
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/summary_stat.jpg)

##### Visualizing – Continuous Variables
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/vis_num.png)

###### Correlations for Continuous Variables
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/vis_corr_matrix.png)

#### Categorical Variables

- 'job', 'marital', 'education', 'default', 'housing', 'loan’
'contact', 'month', 'day_of_week', 'poutcome', 'y'

##### Visualizing – Categorical Variables
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/vis_cat.png)

#### Missing values
There isn’t Null/Missing values in the dataset, but we have unknown values for few categorical variables as seen in the visualization above. We kept the unknown values as they are. Because, these information is not known during a call is performed.
![]https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/missing_values.jpg

### Data Preprocessing and Feature Engineering

#### Principal Component Analysis and Oversampling

- From our  [descriptive study](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/code/descriptive_mini_project%20(2).ipynb) we found a high correlation between 4 columns based on the Heat Map Method. These columns are cons.price.idx, euribor3m, nr.employed,emp.var.rate. Therefore we applied [PCA](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/code/mini_project%20(2)%20smote%20copy.ipynb) on the numerical columns with 99% covariance to remove the multicolinearity between them.
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/pca_99.png)

- The dataset is also unbalanced regarding the target variable (36,548 y_no vs 4640 y_yes). So we oversampled the traing data using smote module from python imblearn library. In oversampling, for every observation in the majority class, we randomly select an observation from the minority class with replacement. The end result is the same number of observations from the minority and majority classes. The notebook for PCA and SMOTE analysis can be found [here](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/code/mini_project%20(2)%20smote%20copy.ipynb). The following figure shows the distribution of target variable in the training set before and after oversampling.
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/smote_result.png)
 
 
#### String Indexing and One-Hot Encoding

We applied One Hot Encoding method to categorical data with using string indexing. One hot encoding is a process by which categorical variables are converted into matrix form with 1 and 0 values.

#### Vector Assembler
We used vector assembler method of PySpark to put all the features in one vector.  

#### Data Normalization

- We normalized our numerical data using standard scaler function of PySpark. The goal of normalization is to change the values of numeric columns in the data set to use a common scale, without distorting differences in the ranges of values or losing information.

- String indexing, one hot encoding, vector assembling, and data normalization are done using a pipeline in PySpark. Before normalizing, the dataset is split in train and test set (80/20) and then normalization is done on training data and transformed in test data. 
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/pipeline.jpg)


#### Modeling
- Logistic Regression (base model) without PCA and Oversampling
- Logistic Regression with PCA and oversampled train data
- Decision Tree Classifier
- Random Forest Classifier



## Logistic Regression
- We applied Logistic Regression algorithm which is the most commonly used algorithm for solving all classification problems.

