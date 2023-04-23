# SC1015_miniproject_Kickstarter

[Video presentation on YouTube](https://youtu.be/mgRbBjRoPTo)

## Team 3

### Group Members
- Abhinav [U2223031L]
- Lucas Ng [U2220046K]

## About

Kickstarter campaigns are investments and pose a risk to the investor, similar to stocks in the stock market. The project focuses on measuring the chances of a campaign failing or succeeding with the attributes from our dataset.

**Index:**
1. [Problem Definition](https://github.com/DeltaRager/SC1015_miniproject_Kickstarter/edit/main/README.md#problem-definition)
2. [Models Used](https://github.com/DeltaRager/SC1015_miniproject_Kickstarter/edit/main/README.md#models-used)
3. [Insight and Conclusion](https://github.com/DeltaRager/SC1015_miniproject_Kickstarter/edit/main/README.md#insight-and-conclusion)
4. [Consulted References](https://github.com/DeltaRager/SC1015_miniproject_Kickstarter/edit/main/README.md#consulted-references)

## Problem Definition

Decoding Kickstarter Campaing Success. Finding out whether a campaign will succeed or fail.
Classification task at the problems core.

## Models Used

1. Logistic Regression
2. Linear Regression


## Insight and Conclusion

Based on our analysis, the problem is a classification task. This involves using the selected predictors to predict which state a campaign is likely to head towards.

When combined with the Logistic Regression prediction model we can predict the state of a campaign using the following predictors:
- usd_goal_real
- duration
- backers
- usd_pledged_real

And for live campaigns, we can take the linear relation between **backers** and the current **usd_pledged_real** for an insight on whether the project is heading on a trajectory of a successful campaign or a failed one.

With successful campaigns having more backers that pledge less when compared to failed campaigns having less backers that pledge more money. This is also taken into account even with the total pledged amount for successful campaigns being ~3x that of the failed campaigns. 

We can thus conclude that, for a campaign to succeed, it is much preferrable for the campaign to have a large pool of support from backers, rather than a few backers shelling out large individual investments. A successful campaign is one that can reach the awareness of a large audience. 

  
## Consulted References
  
  1. https://www.kaggle.com/datasets/kemical/kickstarter-projects?select=ks-projects-201801.csv [dataset]
  2. https://seaborn.pydata.org/
  3. https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
  4. https://scikit-learn.org/stable/
