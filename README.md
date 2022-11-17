# Lorem_Ipsum_Mini_Project_2
 # Term Deposit Opening Decision
 ### Outline
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
    - Logistic Regression 
    - Decision Tree Classifier
    - Gradient Boosted Decision Trees
    - Random Forest Classifier
    - Factorization ML classifier
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

Inferences: 

* The features are highly skewed. Hence, we need to perform scaling on all these data points.

* From the Pdays and Previous univariate distribution we can see that most of the clients in the dataset haven't been contacted in the current and previous campaigns

* Duration of the call is right skewed, with most calls ending early. 

* We did not find any null values


###### Correlations for Continuous Variables
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/vis_corr_matrix.png)

Inferences: 

High correlation between 4 columns based on the Heat Map Method

* cons.price.idx
* Euribor3m
* nr.employed
* emp.var.rate

We will conduct PCA to combat the multicollinearity problem


#### Categorical Variables

- 'job', 'marital', 'education', 'default', 'housing', 'loan’
'contact', 'month', 'day_of_week', 'poutcome', 'y'

#### Visualizing – Categorical Variables
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/vis_cat.png)

* Unknown values for few categorical variables.
* We kept the unknown values as they are.


#### Missing values
There isn’t Null/Missing values in the dataset, but we have unknown values for few categorical variables as seen in the visualization above. We kept the unknown values as they are. Because, these information is not known during a call is performed.
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/missing_values.jpg)

### Data Preprocessing and Feature Engineering

#### Principal Component Analysis

- From our  [descriptive study](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/code/descriptive_mini_project%20(2).ipynb) we found a high correlation between 4 columns based on the Heat Map Method. These columns are cons.price.idx, euribor3m, nr.employed,emp.var.rate. Therefore we applied [PCA](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/code/data_processing_pca_smote.ipynb) on the numerical columns with 99% covariance to remove the multicolinearity between them.
![](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/pca_99.png "Co-variance thresholding")

The following figure shows the loadings of each principal components:

![alt text](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/pca_loadings.jpeg "loadings of each principal components")

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
Following are the predictive models used to identify if the client will subscribe (yes/no) a term deposit (variable y). The confusion matrix in test set for each models are also shown below. Detailed code for the modeling can be found in this PySpark [Notebook](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/code/logreg.ipynb)

- Logistic Regression 
<p align="center">
<img width="800" alt="image" src="https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/cm_log_reg.png">  
</p> 

- Decision Tree Classifier
<p align="center">
<img width="800" alt="image" src="https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/cm_dt.png">  
</p> 

- Random Forest Classifier
<p align="center">
<img width="800" alt="image" src="https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/cm_RF.png">  
</p> 

- Gradient Boosted Decision Trees
<p align="center">
<img width="800" alt="image" src="https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/cm_gbdt.png">  
</p> 

- Factorization ML classifier
<p align="center">
<img width="800" alt="image" src="https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/visualization/cm_FMC.png">  
</p> 

##### Model Comparison
<p align="center">
 <img width="534" alt="image" src="https://user-images.githubusercontent.com/28525282/202321298-6946f69a-bc20-4e3b-bee4-6c102c904185.png">
</p> 


The best Model is Logistic Regression with the highest AUC score. The following is the test ROC curve:
<p align="center">
<img width="385" alt="image" src="https://user-images.githubusercontent.com/28525282/202321240-9c709acb-9900-47dc-8661-2c6f7740eadd.png">
</p>

## Kmeans Clustering
We initially decided on the number of clusters using Silhouette scores. The notebook for KNN can be found [here](https://github.com/Rupom-Bhattacherjee/lorem_ipsum_mini_project_2/blob/main/code/ModelcomparisonandKmeans.ipynb). We arrived at 3 clusters:

<img width="320" alt="image" src="https://user-images.githubusercontent.com/28525282/202333392-9ef30b91-7e2a-46c4-bd12-b589a695fd12.png">

Here is the cluster plot based on the Principal components with most explained variance.

<img width="419" alt="image" src="https://user-images.githubusercontent.com/28525282/202334019-9b86f00f-44dd-4c0d-97a9-a318f78d7ea6.png">

## Prescriptive Recomendations 


The efficiency of any marketing campaign is contingent on the efficacy of its Lead Generation mechanism. It directly correlates to workforce productivity, communication cost, strategic planning and even customer brand judgement. Keeping this in mind, this study looks to design an effective mechanism of lead generation that the marketing team can leverage for higher return on investment. 

We have identified the most important variables that affect the probability of a client signing up for a term deposit leveraging a Logistic Classifier. 

<p align="center">
<img width="522" alt="image" src="https://user-images.githubusercontent.com/28525282/202360339-6112738d-9e0a-4cbf-8d95-abe6a244f9ee.png">
</p>

A 8-point strategy has been recomended to the bank based on our study:

* The first reach-out to potential customers has the highest likelihood of lead conversion. In light of this, most of the marketing/sales workforce must be committed to first time leads. An additional intricacy to workforce management can be deploying highly rated efficient marketing/sales workforce to first time leads and the others to follow-up calls. 
* The duration of the call with the potential customer is positively related to lead conversion. Training programs aimed at the marketing/sales workforce being able to hold conversations with clients must be pushed by the bank. 
* The marketing team needs to closely monitor the monthly Consumer Confidence Index. Our study finds that in periods of rising interest rates, consumers are more likely to purchase term deposits since the increased cost of borrowing makes savings more attractive. An aggressive strategy must be adopted by the marketing/sales team during these monthly sprints. Additionally, younger people (Age 32-38) are more likely to purchase term deposits during these times of downturn and this age group must be targetted strategically in this period. 
* Monday, Tuesday and Thursday are the days of the week when lead conversion usually dips while weekends have a greater conversion rate. Workforce management wherein the marketing and sales team are deployed heavily during high productivity days and given time off during downturn days is suggested.
* Profiling the potential customers by their occupation also has a high impact on lead generation. Blue collar workers are less likely to purchase term deposits. 
* Credit reports of potential customers needs to be analyzed for lead generation. Individuals with a housing loan are less likely to purchase term deposits. 
* Education status of individuals has a direct relation to the likelihood of term deposit purchases. People who have a college degree are more likely to purchase these assets as compared to people with high school degrees. 

To summarize, an ideal scenario for a marketing/sales team would be to target an individual who has a college degree, has no housing loans and has a white collar job. In addition to this, an aggressive communication strategy on Sunday, Wednesday, Friday and Saturday is recomended. It is important to note however that first-time communication with the potential clients needs to be longer and needs to be done by highly efficient/highly rated sales/marketing individuals. In addition to this, during times of economic downturn, an expansive communication strategy needs be be adopted for individuals who are in the 32-38 age bracket. An additional study a quater after the implementation of these recomendations is advised. 


