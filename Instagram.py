import pandas as pd 
import statsmodels 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import PassiveAggressiveRegressor 
# Reading the csv file 
data = pd.read_csv("/Users/loukyaraju/Downloads/Instagram data.csv", encoding = 'latin1') 
print(data.head()) 
data.isnull().sum() # checking for null values in the imported data 
data = data.dropna() # no null values to be dropped 
data.info() # checking for data type of the imported data 
# Analyzing Instagram Reach 
# Distribution of Impressions From Home
plt.figure(figsize=(10, 8)) 
plt.style.use('fivethirtyeight') 
plt.title("Distribution of Impressions From Home") 
sns.distplot(data['From Home']) 
plt.show() 
# Distribution of Impressions From Hashtags 
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags") 
sns.distplot(data['From Hashtags']) 
plt.show() 
# Impressions on Instagram Posts From Various Sources 
home = data["From Home"].sum() 
hashtags = data["From Hashtags"].sum() 
explore = data["From Explore"].sum() 
other = data["From Other"].sum() 
labels = ['From Home','From Hashtags','From Explore','Other'] 
values = [home, hashtags, explore, other] 
fig = px.pie(data, values=values, names=labels,  
title='Impressions on Instagram Posts From Various Sources', hole=0.5) 
fig.show()
# Analyzing Content 
text = " ".join(i for i in data.Caption) 
stopwords = set(STOPWORDS) 
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text) 
plt.style.use('classic') 
plt.figure( figsize=(12,10)) 
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.show() 
text = " ".join(i for i in data.Hashtags) 
stopwords = set(STOPWORDS) 
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text) 
plt.figure( figsize=(12,10)) 
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off") 
plt.show()
# Analyzing Relationships 
# Relationship Between Likes and Impressions 
figure = px.scatter(data_frame = data, x="Impressions", y="Likes", size="Likes", trendline="ols",  title = "Relationship Between Likes and Impressions")
figure.show() 
# Relationship Between Comments and Total Impressions 
figure = px.scatter(data_frame = data, x="Impressions", y="Comments", size="Comments", trendline="ols", title = "Relationship Between Comments and Total Impressions") 
figure.show() 
# Relationship Between Shares and Total Impressions 
figure = px.scatter(data_frame = data, x="Impressions", y="Shares", size="Shares", trendline="ols", title = "Relationship Between Shares and Total Impressions") 
figure.show() 
# Relationship Between Post Saves and Total Impressions 
figure = px.scatter(data_frame = data, x="Impressions", y="Saves", size="Saves", trendline="ols", title = "Relationship Between Post Saves and Total Impressions") 
figure.show() 
# Correlation of all the columns with the Impressions column 
correlation = data.corr() 
print(correlation["Impressions"].sort_values(ascending=False)) 
# Analyzing Conversion Rate 
# Conversion rate of an account = (Follows/Profile Visits) * 100 
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100 
print(conversion_rate) 
# Relationship between the total profile visits and the number of followers gained from all profile visits 
figure = px.scatter(data_frame = data, x="Profile Visits", y="Follows", size="Follows", trendline="ols", title = "Relationship Between Profile Visits and Followers Gained") 
figure.show() 
# Instagram Reach Prediction Model 
# Spliting the data into training and test sets 
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]) 
y = np.array(data["Impressions"]) 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)
# Building the model 
model = PassiveAggressiveRegressor() 
model.fit(xtrain, ytrain) 
model.score(xtest, ytest) 
# Predicting the reach of an Instagram post by giving inputs to the machine learning model 
# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']] 
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]]) 
model.predict(features) 
