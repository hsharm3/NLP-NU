#Date: 2/6
#Version: 1.0

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math  
import nltk
import scipy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer   #see, sees, saw -> see
from collections import Counter
import re  # to get process information
import os
import gc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVC, LinearSVC

def get_tokens(text):
	lowers = text.lower()
	lowers = re.sub("[\:\.\!\/_|,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+","",lowers)    		#get rid of symbols
	tokens = nltk.word_tokenize(lowers)											#question here: lemma?big numbers?
	list_stopWords=list(set(stopwords.words('english')))
	tokens = [w for w in tokens if not w in list_stopWords]				#get rid of stopwords
	return tokens


#[['a','b','c'],['d','e','f']]   -------> ['a b c','d e f'] for tfidf format
def two2one(two_d):
	one_d = []
	for i in range(len(two_d)):
		for j in range(len(two_d[i])):
			if j ==0:
				temp = two_d[i][j]
			else:
				temp += ' '
				temp += two_d[i][j]
		one_d.append(temp)	
	return one_d




#read a file
file = open("ObesitySen1")
file_train = open("train_groundtruth.xml")
file_test = open("test_groundtruth.xml")
try:
	all_text = file.read()
	all_train = file_train.readlines()
	all_test = file_test.readlines()
finally:
	file.close()
	file_train.close()
	file_test.close()

#break into lists
text = all_text.split('[report_end]')		#break into patient's individual records
text.pop(0)									#eliminate the first element because it is a Null

EHR4patients = []							#Each EHR refers a list which contains all the words inside the EHR
for i in range(len(text)):
	temp = get_tokens(text[i])
	EHR4patients.append(temp)

EHR_in_one = two2one(EHR4patients)


#build train and test set
id_pattern = re.compile('\d+')
obesity_pattern = re.compile('[NYUQ]')
test_x_id = []
test_y = []
for i in range(5593,6040):					#5593-6040
	temp_test_x_id = id_pattern.findall(all_test[i])
	temp_test_y = obesity_pattern.findall(all_test[i])
	test_x_id.append(temp_test_x_id[0])
	test_y.append(temp_test_y[0])
for i in range(13495,13988):					#13495-13988
	temp_test_x_id = id_pattern.findall(all_test[i])
	temp_test_y = obesity_pattern.findall(all_test[i])
	test_x_id.append(temp_test_x_id[0])
	test_y.append(temp_test_y[0])

train_x_id = []
train_y = []
for i in range(8722,9387):							#8722-9387 in the xml
	temp_train_x_id = id_pattern.findall(all_train[i])
	temp_train_y = obesity_pattern.findall(all_train[i])
	train_x_id.append(temp_train_x_id[0])
	train_y.append(temp_train_y[0])
for i in range(20162,20892):							#20162-20892 in the xml
	temp_train_x_id = id_pattern.findall(all_train[i])
	temp_train_y = obesity_pattern.findall(all_train[i])
	train_x_id.append(temp_train_x_id[0])
	train_y.append(temp_train_y[0])



#now we have train_x_id --->train_y   and test_x_id  ---> test_y
train_x = []
test_x = []
for id in train_x_id:
	train_x.append(EHR_in_one[int(id)-1])
for id in test_x_id:
	test_x.append(EHR_in_one[int(id)-1])
print("train set & test set done")


# get tfidf score of words
vectorizer = TfidfVectorizer(min_df=1)
train_x_tfidf = vectorizer.fit_transform(train_x)
test_x_tfidf = vectorizer.fit_transform(test_x)
print(train_x_tfidf.shape, test_x_tfidf.shape)

temp_test_x = test_x_tfidf.toarray()
temp_test_x = temp_test_x.tolist()
print(len(temp_test_x),len(temp_test_x[0]))
for row in temp_test_x:
	for i in range(28916-22949):
		row.append(0)
print(len(temp_test_x),len(temp_test_x[0]))
test_x_tfidf = scipy.sparse.csr_matrix(temp_test_x)

print(train_x_tfidf.shape, test_x_tfidf.shape)
print("tfidf done")

gc.collect()
#training with SVM
svm_clf = LinearSVC()
done = svm_clf.fit(train_x_tfidf, train_y)
print("svm training done")
print(done)

print(train_y)
print(test_y)
svm_pred = svm_clf.predict(test_x_tfidf)
print(svm_pred)