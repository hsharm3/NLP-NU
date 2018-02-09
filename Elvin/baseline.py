#Date: 2/8
#Version: 1.1
# Add evaluation metric and also split textual and intuitive classfication

import numpy as np  # linear algebra
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

def testreshape(train,test):
	temp_test_x = test_x.toarray()
	temp_test_x = temp_test_x.tolist()
	print(len(temp_test_x),len(temp_test_x[0]))
	for row in temp_test_x:
		for i in range(len(train[0]) - len(test[0])):
			row.append(0)
	print(len(temp_test_x),len(temp_test_x[0]))
	test_x = scipy.sparse.csr_matrix(temp_test_x)
	return test_x

def get_accuracy(predict, groundtruth):
	count = 0
	for i in range(len(predict)):
		if predict[i] == groundtruth[i]:
			count += 1
	accuracy = count*1.0/len(predict)
	return accuracy

#read a file
file = open("ObesitySen1")
file_train = open("train_groundtruth.xml")
file_test = open("test_groundtruth.xml")
try:
	all_text = file.read()
	train_truth = file_train.readlines()
	test_truth = file_test.readlines()
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

test_textual_x_id = []
test_textual_y = []
for i in range(5593,6040):					#5593-6040
	temp_test_x_id = id_pattern.findall(test_truth[i])
	temp_test_y = obesity_pattern.findall(test_truth[i])
	test_textual_x_id.append(temp_test_x_id[0])
	test_textual_y.append(temp_test_y[0])

test_intui_x_id = []
test_intui_y = []
for i in range(13495,13988):					#13495-13988
	temp_test_x_id = id_pattern.findall(test_truth[i])
	temp_test_y = obesity_pattern.findall(test_truth[i])
	test_intui_x_id.append(temp_test_x_id[0])
	test_intui_y.append(temp_test_y[0])

train_textual_x_id = []
train_textual_y = []
for i in range(8722,9387):							#8722-9387 in the xml
	temp_train_x_id = id_pattern.findall(train_truth[i])
	temp_train_y = obesity_pattern.findall(train_truth[i])
	train_textual_x_id.append(temp_train_x_id[0])
	train_textual_y.append(temp_train_y[0])

train_intui_x_id = []
train_intui_y = []
for i in range(20162,20892):							#20162-20892 in the xml
	temp_train_x_id = id_pattern.findall(train_truth[i])
	temp_train_y = obesity_pattern.findall(train_truth[i])
	train_intui_x_id.append(temp_train_x_id[0])
	train_intui_y.append(temp_train_y[0])


# get tfidf score of words
vectorizer = TfidfVectorizer(max_df=0.95, min_df=3, max_features = 5000)
X = vectorizer.fit_transform(EHR_in_one)
print("tfidf done")
print(type(X))


X = X.toarray()
print(len(X[0]))

#find those train&test intui&textual in that tfidf matrix
train_textual_x = []
train_intui_x = []
for id in train_textual_x_id:
	train_textual_x.append(X[int(id)-1])
for id in train_intui_x_id:
	train_intui_x.append(X[int(id)-1])
test_textual_x = []
test_intui_x = []
for id in test_textual_x_id:
	test_textual_x.append(X[int(id)-1])
for id in test_intui_x_id:
	test_intui_x.append(X[int(id)-1])


print("textual training set size: ",len(train_textual_x),len(train_textual_x[0]))
print("textual testing set size: ",len(test_textual_x),len(test_textual_x[0]))
print("intuitive training set size: ",len(train_intui_x),len(train_intui_x[0]))
print("intuitive testing set size: ",len(test_intui_x),len(test_intui_x[0]))
print("train set & test set done")

train_textual_x = scipy.sparse.csr_matrix(train_textual_x)
test_textual_x = scipy.sparse.csr_matrix(test_textual_x)
train_intui_x = scipy.sparse.csr_matrix(train_intui_x)
test_intui_x = scipy.sparse.csr_matrix(test_intui_x)

gc.collect()

#training with SVM
svm_clf = LinearSVC()
textual_training = svm_clf.fit(train_textual_x, train_textual_y)
textual_pred = svm_clf.predict(test_textual_x)
print("svm training for textual task done")

intui_training = svm_clf.fit(train_intui_x, train_intui_y)
intui_pred = svm_clf.predict(test_intui_x)
print("svm training for intuitive task done")

#print(intui_pred)
print("textual accuracy is: ",get_accuracy(textual_pred,test_textual_y))
print("intuitive accuracy is: ",get_accuracy(intui_pred,test_intui_y))
