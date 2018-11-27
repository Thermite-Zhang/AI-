# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:50:21 2018

@author: zsr19
"""

import pandas as pd 
import numpy as np
data1 = pd.read_csv('C:/Users/zsr19/Desktop/AI_subject/AI_subject/camp_dataset/data/tiku_question_sx.csv')
label = pd.read_csv('C:/Users/zsr19/Desktop/AI_subject/AI_subject/camp_dataset/data/question_knowledge_hierarchy_sx.csv')
data1.set_index('que_id',inplace=True)
data_triangle_id = []
data_function_id = []
for i in range(len(label[:])):
    if label.iloc[i]['kh_id'] == 'hcwf4avcmp8l53s5iq010pelwlce000d':
        data_function_id.append(label.iloc[i]['question_id'])
    if label.iloc[i]['kh_id'] == 'hcwf4avcmp8l53s5iq010pelwlce0035':
        data_triangle_id.append(label.iloc[i]['question_id'])
data_triangle = pd.DataFrame((data1.loc[char]['content'] for char in data_triangle_id if char in data_triangle_id), index = data_triangle_id)
data_function = pd.DataFrame((data1.loc[char]['content'] for char in data_function_id if char in data_function_id), index = data_function_id)
data_triangle.rename(columns={ data_triangle.columns[0]: "content" }, inplace=True)
data_function.rename(columns={ data_function.columns[0]: "content" }, inplace=True)

import re
import jieba
math_words = ['Delta','angle','arccos','arcsin','arctan','bar','bot','cap','cdot','centerdot','circ','cos','cup','dfrac','exists','forall','frac','frown','grave','infty','int','lg','lim','ln','log','max','min','neg','odot','oplus','otimes','overline','partial','pi','prime','prod','sec','csc','sin','subset','subseteq','subsetneqq','supseteq','tan','triangle','tfrac','wedge','widehat','vee']
zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')#按这个pattern去匹配题目中的文字
img_pattern = re.compile(u'<img')#按这个pattern匹配图片
math_pattern = re.compile(u'\\\\([a-zA-Z]+)')#按这个pattern匹配数学用词，比如frac，sin之类的
stop_word = pd.read_csv('C:/Users/zsr19/Desktop/AI_subject/AI_subject/stopwords-master/baidu.txt',header=None)#读取听词表
stop_word = stop_word.values.reshape(-1)#reshape，变成nparray
stop_word = [stop for stop in stop_word]#变列表
stop_word.append(' ')
def extract_words(word):
    zh_word = re.findall(zh_pattern,word)#匹配题目中文
    img_word = re.findall(img_pattern,word)#匹配图片
    math_word = re.findall(math_pattern,word)#匹配数学用词
    math_word = [char for char in math_word if char  in math_words]
    lst=jieba.cut(' '.join(zh_word))
    final_word = [char for char in lst if char not in stop_word]+img_word+math_word#去掉中文听词表中的词，组合以上三种词的
    return  ' '.join(final_word)
tri_data = data_triangle.content.apply(lambda x:extract_words(x))
fun_data = data_function.content.apply(lambda x:extract_words(x))

tri_data = pd.DataFrame(tri_data)
tri_data['label'] = 1
tri_data.rename(columns={ tri_data.columns[0]: "content" }, inplace=True)
fun_data = pd.DataFrame(fun_data)
fun_data['label'] = 0
fun_data.rename(columns={ fun_data.columns[0]: "content" }, inplace=True)

df_all = pd.concat([tri_data,fun_data])
df_all.dropna(inplace = True)
df_all = df_all.drop_duplicates(subset = 'content', keep = False)
df_all = df_all[df_all['content']!='']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
counter = CountVectorizer(max_df=0.8,min_df=0.001)
counts = counter.fit_transform(df_all.content)
tfidfer = TfidfTransformer()
tfidf = tfidfer.fit_transform(counts)
data=tfidf.toarray()
label=df_all.label.values

#因子分解机

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state = 3)
x_train_dict = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in x_train]
x_test_dict = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in x_test]
v = DictVectorizer()
x_train_dv = v.fit_transform(x_train_dict)
x_test_dv = v.transform(x_test_dict)
fm = pylibfm.FM(num_factors=100, num_iter=300, verbose=True, task="classification", initial_learning_rate=0.001, learning_rate_schedule="optimal")
fm.fit(x_train_dv,y_train)

from sklearn.metrics import accuracy_score, roc_auc_score
print(accuracy_score(y_test,(fm.predict(x_test_dv)>0.5).astype(int)))
print(roc_auc_score(y_test,fm.predict(x_test_dv)))

#SVM model
#param_grid = {"gamma":[0.001,0.01,0.1,1,10,100],
            # "C":[0.001,0.01,0.1,1,10,100],
             #"kernel":['rbf']}
#grid_search_svm = GridSearchCV(svm.SVC(),param_grid,cv=5)
#fitted_model_svm = model_svm.fit(x_train,y_train)
#y_pred_svm = fitted_model_svm.predict(x_test)
#grid_search_svm.fit(x_train,y_train) 
#print("Test set score:{:.2f}".format(grid_search_svm.score(x_test,y_test)))
#print("Best parameters:{}".format(grid_search_svm.best_params_))
#print("Best score on train set:{:.2f}".format(grid_search_svm.best_score_))

model_svm = svm.SVC(kernel='rbf',C=2,gamma=0.1)
fitted_model_svm = model_svm.fit(x_train,y_train)
y_pred_svm = fitted_model_svm.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred_svm)*100, "%")
print("ROC_AUC:", roc_auc_score(y_test, y_pred_svm)*100, "%" )
#Random Forest model

model_for = RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=30)
fitted_model_for = model_for.fit(x_train, y_train)
y_pred_for = fitted_model_for.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred_for)*100, "%")
print("ROC_AUC:", roc_auc_score(y_test,y_pred_for)*100, "%" )

from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(model_svm, data, label, cv=5, scoring="accuracy")
scores_rf = cross_val_score(model_for, data, label, cv=5, scoring="accuracy")

from mlxtend.classifier import StackingClassifier
clf1 = model_svm
clf2 = model_for
lr = LogisticRegression()
sclf = StackingClassifier(classifiers = [clf1,clf2], meta_classifier = lr)
print("5-fold cross validation:\n")

for clf, label_clf in zip([clf1,clf2,sclf],['SVM', 'RF', 'SC']):
    scores = cross_val_score(clf, data, label, cv = 5,scoring = 'accuracy')
    print("Accuracy:%0.2f(+/-%0.2f), [%s]"% (scores.mean(), scores.std(), label_clf))


model_svm.probability = True
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold

cv = StratifiedKFold(y = y_train, n_folds=5,random_state = 1)
fig = plt.figure(figsize = (12,16))
for i, (train,test) in enumerate(cv): 
    prob1 = model_svm.fit(x_train[train], y_train[train]).predict_proba(x_train[test])
    fpr1,tpr1,threshold1 = roc_curve(y_train[test],prob1[:,1],pos_label = 1)  
    roc_auc1 = auc(fpr1,tpr1)
    ax = plt.subplot(2,1,1)
    plt.plot(fpr1, tpr1, lw=1, label='ROC_svm fold:{},auc:{}'.format(i, roc_auc1))
    

    prob2 = model_for.fit(x_train[train], y_train[train]).predict_proba(x_train[test])
    fpr2,tpr2,thresholds = roc_curve(y_train[test],prob2[:,1],pos_label = 1)  
    roc_auc2 = auc(fpr2,tpr2)
    ax = plt.subplot(2,1,2)
    plt.plot(fpr2, tpr2, lw=1, label='ROC_for fold:{},auc:{}'.format(i, roc_auc2))
'''
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
mean_tpr /= len(cv) 					#在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr[-1] = 1.0 						#坐标最后一个点为（1,1）
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2) 
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

'''





