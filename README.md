# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers


# Coding and Output
```
NAME   : AISHWARYA S
REF NO : 212222100003
DEP    : CSE(CS)

```
 ```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
df=pd.read_csv("/content/SAMPLEIDS (1).csv")
df
```
![image](https://github.com/user-attachments/assets/1a83f030-ff17-44a8-bc67-36a8f630bffe)
```
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/045a7626-aa1c-4439-a5cc-fcfe78a24888)
```
df.isnull().any()
```
![image](https://github.com/user-attachments/assets/ff397368-d05b-4991-8385-5f618738324e)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/be956ac4-69b8-4c19-a527-a95caccfb948)
```
df.fillna(0)
```
![image](https://github.com/user-attachments/assets/508d037c-ada7-44d1-9e0b-484fd92e805f)
```
df.fillna(method = 'ffill')
```
![image](https://github.com/user-attachments/assets/799a3e43-3ddc-4d72-8f4b-98eae0b0725b)

```
df.fillna(method = 'bfill')
```
![image](https://github.com/user-attachments/assets/867d49ce-2dc5-4a27-b42b-e3059d6594eb)
```
df_dropped = df.dropna()
df_dropped
```
![image](https://github.com/user-attachments/assets/25616d5c-f087-4b5d-a6ed-34e598516c57)
```
df.fillna({"GENDER":"MALE","NAME":"ROSE","ADDRESS":"VELACHERY","M1":98,"M2":87,"M3":76,"M4":92,"TOTAL":305,"AVG":89.999999
```
iris=pd.read_csv("/content/iris (1).csv")
iris

![image](https://github.com/user-attachments/assets/8b665a11-ffb0-45f3-8ebe-0aa7763247cd)
```
iris.describe()
```
![image](https://github.com/user-attachments/assets/d0500820-56dc-4354-9b71-a6ef1691e18d)
```

sns.boxplot(x='sepal_width', data=iris)
```
![image](https://github.com/user-attachments/assets/82255438-fa83-4b1a-9580-3fb29d93d22e)

```
c1=iris.sepal_width.quantile(0.25)
c3=iris.sepal_width.quantile(0.75)
iq=c3-c1
print(c3)
```
![image](https://github.com/user-attachments/assets/57d375a6-9605-4656-bc51-253f24c4d897)

```
rid=iris[((iris.sepal_width<(c1-1.5*iq))|(iris.sepal_width>(c3+1.5*iq)))]
rid['sepal_width']
```
![image](https://github.com/user-attachments/assets/efa4ef74-56c4-4b50-9fc8-c0947be601d0)
```
delid=iris[~((iris.sepal_width<(c1-1.5*iq))|(iris.sepal_width>(c3+1.5*iq)))]
delid

```
![image](https://github.com/user-attachments/assets/17fd99f7-bc52-4179-8966-b4d7df33b5b1)

```
sns.boxplot(x='sepal_width',data=delid)
```
![image](https://github.com/user-attachments/assets/17b4e2bc-32be-4736-ab15-099df77cdbad)

```
height=pd.read_csv("/content/heights.csv")
height
```
![image](https://github.com/user-attachments/assets/4e0e0728-9d5a-4b1a-b6a2-0e850de0e22a)
```
df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr

```
![image](https://github.com/user-attachments/assets/3c12a372-eb26-4e46-a419-43eabdde1c3f)

```
low = q1 - 1.5*iqr
low
```
![image](https://github.com/user-attachments/assets/60ace6b9-477c-429b-9f53-d3f5534c6140)
```
high = q3 + 1.5*iqr
high
```

![image](https://github.com/user-attachments/assets/7162df08-e506-4d9a-923b-7e21e7325075)

```

df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```
![image](https://github.com/user-attachments/assets/e2fc9060-a792-41af-9273-e2d9fba8a504)
```

z = np.abs(stats.zscore(df['height']))
z

```
![image](https://github.com/user-attachments/assets/5fe3038b-a539-4bb5-bd7e-03c5029957b5)

```
df1 = df[z<3]
df1
```
![image](https://github.com/user-attachments/assets/14914346-bd63-4075-8fbd-6435aa57fd22)


          
# Result
Thus to read the given data and perform data cleaning and save the cleaned data to a file done successfully.
