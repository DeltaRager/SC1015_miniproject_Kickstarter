# SC1015_miniproject_Kickstarter

# Importing Libraries
```py
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
sb.set()
```


# Importing the Dataset
```py
data = pd.read_csv('ks-projects-201801.csv')
```


# Cleaning the Dataset
```py
data.dropna(subset = ['name'], inplace=True)
clean_data = data.copy()
clean_data['state'] = clean_data['state'].replace(['canceled', 'suspended'], 'failed')
clean_data = clean_data.drop_duplicates('name', keep='last')
clean_data = clean_data.drop(['ID','pledged','currency','usd pledged'],axis=1)
clean_data = clean_data[clean_data['state'] != 'undefined']
clean_data = clean_data[clean_data['state'] != 'live']
clean_data.head()
```

Checking the number of failed campaigns vs successful campaigns
```py
sb.catplot(y = "state", data = clean_data, kind = "count")
```
![download](https://user-images.githubusercontent.com/26520694/233824255-805a8fab-0f09-4d1b-acf8-9337df76bc30.png)



# Creation of new variables for EDA
```py
clean_launched = pd.to_datetime(clean_data['launched'])
clean_deadline = pd.to_datetime(clean_data['deadline'])
duration = clean_deadline - clean_launched

clean_data['duration'] = duration.dt.days
clean_data['launch_month'] = clean_launched.dt.month
clean_data['launch_year'] = clean_launched.dt.year

clean_data = clean_data[clean_data['launch_year'] != 1970]

clean_data['name_length'] = clean_data['name'].str.len()

clean_data['pledge_per_backer'] = pd.DataFrame(clean_data['usd_pledged_real'] / clean_data['backers'])
clean_data['pledge_per_backer'] = clean_data['pledge_per_backer'].fillna(0)

failed = clean_data[clean_data.state == 'failed']
successful = clean_data[clean_data.state == 'successful']
```


# Basic EDA

### Checking the average goal amount and average pledged amount for failed and successful campaigns
```py
print("Average goal amount in USD\n")
print("Failed (USD) : ", failed['usd_goal_real'].mean())
print("Successful (USD) : ", successful['usd_goal_real'].mean())

print("\nAverage pledged amount in USD\n")
print("Failed (USD) : ", failed['usd_pledged_real'].mean())
print("Successful (USD) : ", successful['usd_pledged_real'].mean())
```
**Output:**
```
Average goal amount in USD

Failed (USD) :  66051.07049089693
Successful (USD) :  9552.797650802477

Average pledged amount in USD

Failed (USD) :  1549.8790561496423
Successful (USD) :  22744.237115292137
```

We can infer that failed campaigns have a larger goal amount but cannot raise as much where as successful campaigns have a much lower goal amount and are able to raise amounts much higher than their goals.

### Checking the average duration in days for failed and successful campaigns

```py
print("Average duration in Days\n")
print("Failed (Days) : ", failed['duration'].mean())
print("Successful (Days) : ", successful['duration'].mean())
```
**Output:**
```
Average duration in Days

Failed (Days) :  34.75573594355426
Successful (Days) :  31.16973731666266
```

Not much can be inferred from this. The average campaign seems to last around a month, with failed campaigns lasting a bit longer than the successful ones.

### Average goal per category by year

```py
ax = plt.subplot(111)
clean_data.groupby(['launch_year', 'main_category'])['usd_goal_real'].mean().unstack().plot(kind = 'bar',
                                                                                      figsize=(12,8), stacked=True, width=0.8, colormap='rainbow',ax=ax)

plt.title('Average Goal per Category by Year \n', fontsize=22)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Goal', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()
```
![download](https://user-images.githubusercontent.com/26520694/233824411-72097105-27ea-49fa-831a-860c81b9f777.png)

## The number of successful and failed campaigns per category

### Successful Campaigns
```py
a4_dims = (23.4, 16.5)
fig, ax = plt.subplots(figsize=a4_dims)

sb.barplot(x=successful.category.value_counts().index,
                  y=successful.category.value_counts().values, ax=ax)

plt.xticks(rotation=90)
plt.tight_layout()
plt.title('Number of successful campaigns per category', fontsize=60)
plt.show()
```
![download](https://user-images.githubusercontent.com/26520694/233824433-fbc9f5c1-4ae4-49ee-92ad-1bf56341b554.png)

### Failed Campaigns
```py
a4_dims = (23.4, 16.5)
fig, ax = plt.subplots(figsize=a4_dims)

sb.barplot(x=failed.category.value_counts().index,
                  y=failed.category.value_counts().values, ax=ax)

plt.xticks(rotation=90)
plt.tight_layout()
plt.title('Number of failed campaigns per category', fontsize=60)
plt.show()
```
![download](https://user-images.githubusercontent.com/26520694/233824447-dc0f4aaa-e003-41df-a5e9-e9da2d05d03b.png)



# Checking the Correlation
```py
cor = pd.DataFrame(clean_data, columns=['backers', 'usd_pledged_real', 'usd_goal_real', 'duration', 'launch_month', 'launch_year', 'name_length', 'pledge_per_backer'])
fig, ax = plt.subplots(figsize=(10,8))
sb.heatmap(cor.corr(), cmap='Greens',ax=ax)

plt.title('Correlation between Attributes \n', fontsize=20)
plt.show()
```
![download](https://user-images.githubusercontent.com/26520694/233824481-b4a1b2ef-1980-46c3-97ad-efb77c3c1ce2.png)


