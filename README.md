# SC1015_miniproject_Kickstarter

## Group Members
- Abhinav [U2223031L]
- Lucas Ng [U2220046K]

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
Done by Abhinav and Lucas
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
Done by Abhinav
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
Done by Abhinav

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
Done by Abhinav

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
Done by Abhinav

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

### Main Category vs Pledged Amount in USD
Done by Lucas

```py
sb.barplot(x=clean_data["main_category"], y=clean_data["usd_pledged_real"])
plt.xlabel('X-Axis Label' ,fontsize=2)
plt.xticks(rotation=45)
plt.title('Main Category vs usd_pledge_real \n', fontsize=30)
```
![download](https://user-images.githubusercontent.com/26520694/233837962-d480f50f-fa02-4ca1-b95a-eaf3a92f3b0f.png)



## The number of successful and failed campaigns per category
Done by Abhinav

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
Done by Abhinav

```py
cor = pd.DataFrame(clean_data, columns=['backers', 'usd_pledged_real', 'usd_goal_real', 'duration', 'launch_month', 'launch_year', 'name_length', 'pledge_per_backer'])
fig, ax = plt.subplots(figsize=(10,8))
sb.heatmap(cor.corr(), cmap='Greens',ax=ax)

plt.title('Correlation between Attributes \n', fontsize=20)
plt.show()
```
![download](https://user-images.githubusercontent.com/26520694/233824481-b4a1b2ef-1980-46c3-97ad-efb77c3c1ce2.png)



# Predictions


## Logistic Regression
Done by Abhinav

Using our root problem statement, we want to have a good way of measuring the chances of a campaign failing or succeeding with the attributes from our dataset. We can see from the previous correlation heatmap, **backers** and **usd_pledged_real** are heavily correlated. 

We would also like to **usd_goal_real** and **duration** to our list of predictors as it would be intuitive for a first time investor to look at those attributes of a campaign before investing.

```py
predictors = pd.DataFrame(clean_data, columns=['usd_goal_real', 'duration', 'backers', 'usd_pledged_real'])
X_train, X_test, y_train, y_test = train_test_split(predictors, clean_data['state'], test_size=0.2, random_state=3)

model = LogisticRegression()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print('Accuracy:', score)
```

**Output:**
```
Accuracy: 0.9912716348430924
```

The accuracy is 99.13%, which is very good. Thus, we can conclude that the following predictors are very useful in predicting a campaigns state.


We will also plot the confusion matrix to get better insights on its predictions.

```py
predictions = model.predict(X_test)

cm = metrics.confusion_matrix(y_test, predictions)
print(y_test.value_counts())

plt.figure(figsize=(9,9))
sb.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual state');
plt.xlabel('Predicted state');
all_sample_title = 'Accuracy Score: {0} | failed : 0 | successful : 1'.format(score)
plt.title(all_sample_title, size = 15);
```

**Output:**
```
failed        47125
successful    26772
Name: state, dtype: int64
```
![download](https://user-images.githubusercontent.com/26520694/233834140-251976e1-b2c6-40b9-afc7-03f578fda80e.png)




## Linear Regression
Done by Lucas

Given the strong correlation between the variables **backers** and **usd_pledged_real** and their predictive power, we will employ linear regression to explore the relationship between these variables and identify any patterns.

This could be done as a part of EDA without the use of Linear Regression, but we have found that the Regression model helps us visualize it better.

```py
fail = failed[failed.backers < 10000]
succ = successful[successful.backers < 10000]
```

The above is executed first to standardize the scales and limit the amount of outliers as there are a lot in this dataset,


### Failed Campaign Analysis

```py
from sklearn.linear_model import LinearRegression

y = pd.DataFrame(fail['backers'])
X = pd.DataFrame(fail['usd_pledged_real'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

lr.fit(X_train, y_train)

regline_x = X_train
regline_y = lr.intercept_ + lr.coef_ * X_train

print("Intercept:", lr.intercept_)
print("Coefficient:", lr.coef_[0])

f, axes = plt.subplots(1, 1, figsize=(16, 8))
plt.scatter(X_train, y_train)
plt.plot(regline_x, regline_y, 'r-', linewidth = 3)
plt.title('Failed Campaigns: backers vs usd_pledged_real', fontsize=20)
plt.ylim(top=10000)
plt.show()
```

**Output:**
```
Intercept: [6.12934067]
Coefficient: [0.00809498]
```
![download](https://user-images.githubusercontent.com/26520694/233835027-52dba815-021b-4849-aac3-1e58dbf2c9a7.png)



### Successful Campaign Analysis

```py
y = pd.DataFrame(succ['backers'])
X = pd.DataFrame(succ['usd_pledged_real'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

lr.fit(X_train, y_train)

regline_x = X_train
regline_y = lr.intercept_ + lr.coef_ * X_train

print("Intercept:", lr.intercept_)
print("Coefficient:", lr.coef_[0])

f, axes = plt.subplots(1, 1, figsize=(16, 8))
plt.scatter(X_train, y_train)
plt.plot(regline_x, regline_y, 'r-', linewidth = 3)
plt.ylim(top=10000)
plt.show()
```

**Output:**
```
Intercept: [114.29508982]
Coefficient: [0.00563004]
```
![download](https://user-images.githubusercontent.com/26520694/233835046-31997f3e-9e02-4e84-84bf-a4c75eb27cf6.png)


We noticed how the failed campaigns had a much lower slope than the successful campaigns, indicating that there were **less backers per pledged dollar** for failed campaigns as compared to a **higher backers per pledged dollar** for successful campaigns.

Hence, when combined with the Logistic Regression prediction model we can predict the state of a campaign using the following predictors:
- usd_goal_real
- duration
- backers
- usd_pledged_real

And for live campaigns, we can take the ratio between **backers** and the current **usd_pledged_real** for a visual on whether the project is heading on a trajectory of a successful campaign or a failed one.
