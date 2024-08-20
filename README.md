Exno:1
Data Cleaning Process

AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

Coding and Output
Data Cleaning
import pandas as pd
df=pd.read_csv("/content/SAMPLEIDS.csv")
df

output

df.isnull().sum()
output

df.isnull().any()
output

df.dropna()
output

df.fillna(0)
output

df.fillna(method = "ffill")
output

df.fillna(method = 'bfill')
output

df_dropped = df.dropna()
df_dropped
output

df.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':305,'AVG':89.999999})
output

IQR(Inter Quartile Range)
ir=pd.read_csv('iris.csv')
ir
output

ir.describe()
output

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x='sepal_width',data=ir)
plt.show()

output

 c1=ir.sepal_width.quantile(0.25)
 c3=ir.sepal_width.quantile(0.75)
 iq=c3-c1
 print(c3)
output

rid=ir[((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
rid['sepal_width']
output

delid=ir[~((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
delid
output

sns.boxplot(x='sepal_width',data=delid)
output

Z - Score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
dataset=pd.read_csv("heights.csv")
dataset
output

df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
output

low = q1- 1.5*iqr
low
output

high = q3 + 1.5*iqr
high
output

df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
output

z = np.abs(stats.zscore(df['height']))
z
output

df1 = df[z<3]
df1
output

Result
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method.

About
data process

Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Report repository
Releases
No releases published
Packages
No packages published
Languages
Jupyter Notebook
81.3%
 
Python
18.7%
Footer
