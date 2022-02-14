#!/usr/bin/env python
# coding: utf-8

# Author: Harshit Tripathi

# Text Data Analysis
# 
We have used textblob library for text analysis. The features of the library are as follow:
1) Noun phrase extraction
2) Part-of-speech tagging
3) Sentiment analysis
4) Classification (Naive Bayes, Decision Tree)
5) Tokenization (splitting text into words and sentences)
6) Word and phrase frequencies
7) Parsing
8) n-grams
9) Word inflection (pluralization and singularization) and lemmatization
10) Spelling correction
11) Add new models or languages through extensions
12) WordNet integration

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


comments = pd.read_csv('C:/Users/Asus/Downloads/GBcomments.csv',error_bad_lines=False)


# In[3]:


comments.head(10)


# In[4]:


comments['likes'].max()


# In[5]:


comments.tail()


# In[6]:


comments['comment_text'].iloc[60630]


# In[ ]:





# In[7]:


from textblob import TextBlob


# In[8]:


TextBlob('Liked it before watching it cause I love what I eat in a day vids/vlogs 👌🏼').sentiment


# In[9]:


comments.isna().sum()


# In[10]:


comments.dropna(inplace=True)


# In[11]:


polarity=[]
for i in comments['comment_text']:
    polarity.append(TextBlob(i).sentiment.polarity)


# In[12]:


comments['polarity']=polarity


# In[13]:


comments.head(10)

Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance. Significant textual data points can be highlighted using a word cloud. Word clouds are widely used for analyzing data from social network websites.
# In[14]:


comments_positive=comments[comments['polarity']==1]


# In[15]:


get_ipython().system('pip install wordcloud')


# In[16]:


from wordcloud import WordCloud , STOPWORDS


# In[17]:


stopwords=set(STOPWORDS)


# In[18]:


total_string=' '.join(comments_positive['comment_text'])


# In[19]:


wordcloud = WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_string)


# In[20]:


plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
#These are the words which have high polarity among all.


# In[ ]:





# In[26]:


comments_negative=comments[comments['polarity']== -1]


# In[27]:


total_string=' '.join(comments_negative['comment_text'])


# In[28]:


wordcloud = WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_string)


# In[29]:


plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
#These are the words which have lowest polarity among all.


# In[ ]:





# In[ ]:





# In[ ]:




