#!/usr/bin/env python
# coding: utf-8

# In[42]:


from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

reviews_train = load_files("machine_learning/aclImdb/train/")
reviews_test = load_files("machine_learning/aclImdb/test/")

text_train, y_train = reviews_train.data, reviews_train.target
text_test, y_test = reviews_test.data, reviews_test.target

text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train, text_test)
X_train = vect.transform(text_train)
X_test = vect.transform(text_test)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("점수: {:.4f}".format(np.mean(score)))


# In[ ]:




