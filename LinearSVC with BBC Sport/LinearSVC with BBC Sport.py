#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.datasets import load_files
import numpy as np
import chardet

files = load_files("machine_learning/bbcsport/")
X,y = files.data, files.target

for i in range(len(X)):
    if(chardet.detect(X[i])!="utf-8"):
        X[i] = X[i].decode(chardet.detect(X[i])['encoding']).encode('utf8')

X = [doc.replace(b"<br />", b"") for doc in X]
X = [doc.replace(b"\n",b" ") for doc in X]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, norm="l2")
vect = vectorizer.fit_transform(X)

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

model = LinearSVC()
score = cross_val_score(model, vect, y, cv=5)
print("Cross-Validation 평균점수: {:.4f}".format(np.mean(score)))


# In[ ]:




