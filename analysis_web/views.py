from django.shortcuts import render

# Create your views here.
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from pandas import plotting
from pandas import DataFrame
__all__ = [plotting]
iris_dataset=datasets.load_iris()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display
## Data 형태 분석
#X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
#
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#
# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60,alpha=.8, cmap=mglearn.cm3)
# plt.show()
#
# print("irist_dataset의 키:",iris_dataset.keys())
# #각 키들의 구체적인 내용
# print("타켓의 이름:",iris_dataset['target_names'])
# print("특성의 이름",iris_dataset['feature_names'])
# print("[DESCR]의 내용",iris_dataset['DESCR'][:200]+"\n...")
# print("data의 타입",type(iris_dataset['data']))
# print("data의 크기",iris_dataset['data'].shape)
# print("target의 타입",type(iris_dataset['target']))
# print("target의 크기",iris_dataset['target'].shape)
# print("타켓:",iris_dataset['target'])
##

## knn 분석
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)
#
# X_new = np.array([[3,2,1,5]])
# prediction = knn.predict(X_new)
# print("예측된 클래스:",prediction)
# print("예측한 타겟의 이름:",iris_dataset['target_names'][prediction])
# print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test,y_test)))
#
# y_pred = knn.predict(X_test)
# print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))

## plt 분석
# X, y = mglearn.datasets.make_forge()
#
# mglearn.discrete_scatter(X[:,0],X[:,1],y)
# plt.legend(["클래스 0","클래스 1"], loc=4)
# plt.xlabel("First Preperty")
# plt.ylabel("Second Property")
# plt.show()
# print("X.shape:",X.shape)
#
# X, y = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X, y, 'o')
# plt.ylim(-3,3)
# plt.xlabel("Property")
# plt.ylabel("Target")
# plt.show()
#
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# print("cancer.keys():\n",cancer.keys())
# print("유방암 데이터 형태:",cancer.data.shape)
# print("클래스별 샘플 개수:\n",{n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
# print("특성 이름:\n",cancer.feature_names)
#
# from sklearn.datasets import load_boston
# boston = load_boston()
# print("boston.keys():\n",boston.keys())
# print("boston 데이터 형태:\n",boston.data.shape)
# print("특성 이름:\n",boston.feature_names)
#
# X, y = mglearn.datasets.load_extended_boston()
# print(X.shape)
#
# from sklearn.model_selection import train_test_split
# X, y = mglearn.datasets.make_forge()
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#

## knn 분석2
# clf = KNeighborsClassifier(n_neighbors=3)
#
# clf.fit(X_train,y_train)
# print("예측된 테스트 세트:",clf.predict(X_test))
# print("정확도: {:.2f}".format(clf.score(X_test,y_test)))
##
# from sklearn.datasets import load_breast_cancer
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
#
# training_accuracy = []
# test_accuracy = []
#
# neighbors_settings = range(1,16)
# for n_neighbors in neighbors_settings:
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train,y_train)
#     training_accuracy.append(clf.score(X_train,y_train))
#     test_accuracy.append(clf.score(X_test,y_test))
#
# plt.plot(neighbors_settings,training_accuracy,label="Train Accuracy")
# plt.plot(neighbors_settings,test_accuracy,label="Test Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("n_neighbors")
# plt.legend()
# plt.show()


## Regression 분석
from sklearn.linear_model import LinearRegression
# X, y = mglearn.datasets.make_wave(n_samples=60)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#
# lr = LinearRegression().fit(X_train, y_train)
#
# print("lr.coef_:",lr.coef_)
# print("lr.intercept_:",lr.intercept_)
#
# print("훈련 세트 점수: {:.2f}".format(lr.score(X_train,y_train)))
# print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
#
# X, y = mglearn.datasets.load_extended_boston()
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# lr= LinearRegression().fit(X_train,y_train)
#
# print("훈련 세트 점수: {:.2f}".format(lr.score(X_train,y_train)))
# print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
# print("")


## Ridge, Lasso 분석
# from sklearn.linear_model import Ridge
#
# ridge = Ridge(alpha=10).fit(X_train, y_train)
# ridge2 = Ridge(alpha=0.1).fit(X_train, y_train)
#
# from sklearn.linear_model import Lasso
#
# lasso = Lasso().fit(X_train, y_train)
# print("훈련 세트 점수: {:.2f}".format(lasso.score(X_train,y_train)))
# print("테스트 세트 점수: {:.2f}".format(lasso.score(X_test,y_test)))
# print("사용한 특성의 개수:", np.sum(lasso.coef_ != 0))
# print("")
# lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
# print("훈련 세트 점수: {:.2f}".format(lasso001.score(X_train,y_train)))
# print("테스트 세트 점수: {:.2f}".format(lasso001.score(X_test,y_test)))
# print("사용한 특성의 개수:", np.sum(lasso001.coef_ != 0))
# print("")
# lasso002 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
# print("훈련 세트 점수: {:.2f}".format(lasso002.score(X_train,y_train)))
# print("테스트 세트 점수: {:.2f}".format(lasso002.score(X_test,y_test)))
# print("사용한 특성의 개수:", np.sum(lasso002.coef_ != 0))

import csv
f = open('loan.csv',encoding='utf-8-sig')
data = csv.reader(f)
print(data) #csv파일의 내용이 저장되는 위치 출력(csv reader객체인 data)
for row in data:#csv파일 각 열들을 출력
    print(row)
f.close()
print("")

#items : 실제 데이터 기반 텍스트용
#items = pd.read_csv('loan.csv',header=None,names=['stnd_dt','cusno','module','modulenm','acno','newdt','enddt','interest','target'],thousands=',')
items = pd.read_csv('loan.csv',header=None,names=['target','stnd_dt','cusno','module','modulenm','acno','newdt','enddt','interest'],thousands=',') #target 기반 데이터 네이밍
print(items.columns)
print(items['target'])

items2 = pd.read_csv('basic.csv',header=None,names=['target','sex','age','income','degree'],thousands=',') #target 기반 데이터 네이밍
print(items2.columns)
print(items2['target'])

items3 = pd.read_csv('supplement.csv',header=None,names=['target','sex','age','income','degree','postage','postage_delayed_cnt','health','safe_asset','invest_tendency','insurance_fee','tenant_info'],thousands=',') #target 기반 데이터 네이밍
print(items3.columns)
print(items3['target'])

## DataFrame을 통한 다차원 학습 실패(단 1:1 매칭 학습 및 예측은 가능) --> 일반 arrary 형태 사용
data_f = pd.DataFrame(data=dict(interest = items['interest'], module = items['module'])) #converting to DataFrame(딕셔너리 형태)
print(data_f)
#
#X_train, X_test, y_train, y_test=train_test_split(data_f,items['target'].to_frame(),random_state=0)
X_train, X_test, y_train, y_test=train_test_split(items['interest'].to_frame(),items['target'].to_frame(),random_state=0)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
X_new = np.array([[2.2]])
prediction = knn.predict(X_new)
print("예측:",prediction)
print("dd: {:.2f}".format(knn.score(X_test,y_test)))


## 일반 array형태를 통한 다차원적 학습
X = np.array(items.drop(['target'], 1))
print(X)
y = np.array(items['target'])
print(y)
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test) #test
print(accuracy) #this works fine

# 예측 예시를 통한 예측 진행
example = np.array([20190516,1,30,2,70700,20190501,20192003,7.3])
example = example.reshape(1, -1)

prediction = clf.predict(example)
print(prediction)

## 일반 array형태를 통한 다차원적 학습
X = np.array(items2.drop(['target'], 1))
print(X)
y = np.array(items2['target'])
print(y)
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test) #test
print(accuracy) #this works fine

# 예측 예시를 통한 예측 진행
example = np.array([1,30,9000,4])
example = example.reshape(1, -1)

prediction = clf.predict(example)
print(prediction)

plt.plot(X, y, 'o')
plt.xlabel("Property")
plt.ylabel("Target")
plt.show()

## 일반 array형태를 통한 다차원적 학습
# RandomForest 주요 매개변수 : n_estimators(클수록 안정적. 더 많은 트리 사용), max_features(트리의 무작위성. 적을 수록 안정적), max_depth(사전 가지치기 효과. 작을 수록 좋다. 통상 5)
# GradientBoosting 주요 매개변수 : n_estimators(너무 크면 과대 적합), learning_rate(낮출수록 좋음. 0.01), max_dept(3이 적절)
X = np.array(items3.drop(['target'], 1))
print(X)
y = np.array(items3['target'])
print(y)
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
#clf = RandomForestClassifier(n_estimators = 10, max_features = 10, max_depth = 5)
clf = DecisionTreeClassifier(max_features = 3, max_depth = 3)
#clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test) #test
print(accuracy) #this works fine

# 예측 예시를 통한 예측 진행
example = np.array([1,30,4000,3,38000,1,0,80,2,30,0])
example = example.reshape(1, -1)

prediction = clf.predict(example)
print(prediction)

#학습 과정 decision tree 이미지 생성
export_graphviz(clf,out_file="tree1.dot",class_names=["1","2","3","4","5","6","7","8","9"],feature_names=["sex","age","income","degree","postage","postage_delayed_cnt","health","safe_asset","invest_tendency","insurance_fee","tenant_info"],impurity=False,filled=True)

with open("tree1.dot") as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='tree2.png')
#items.describe()

## Django Rest Framework Post 예제
# class Train(views.APIView):
#     def post(self, request):
#         iris = datasets.load_iris()
#         mapping = dict(zip(np.unique(iris.target),iris.target_names))
#
#         X = pd.DataFrame(iris.data, columns=iris.feature_names)
#         y = pd.DataFrame(iris.target).replace(mapping)
#         model_name = request.data.pop('model_name')
#
#
#         try:
#             clf = RandomForestClassifier(**request.data)
#             clf.fit(X,y)
#
#         except Exception as err:
#             return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
#
#         path = os.path.join(settings.MODEL_ROOT, model_name)
#         with open(path,'wb') as file:
#             pickle.dump(clf,file)
#         return Response(status=status.HTTP_200_OK)

# 학습 매개변수 post하여(Jason) 학습 모델링(array 형태로 진행)

class Train(views.APIView):
     def post(self, request):

        #request.POST._mutable = True

        X = np.array(items3.drop(['target'], 1))
        y = np.array(items3['target'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        model_name = request.data.pop('model_name')


        if model_name == 'model_1':
         try:
             clf = KNeighborsClassifier(**request.data)
             #clf = RandomForestClassifier(**request.data)
             #clf = GradientBoostingClassifier(**request.data)
             clf.fit(X_train, y_train)
             #accuracy = clf.score(X_train, y_train) #학습 정확도 출력
             accuracy = clf.score(X_test, y_test) #테스트 정확도 출력
         except Exception as err:
             return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

         path = os.path.join(settings.MODEL_ROOT, model_name)
         with open(path,'wb') as file:
             pickle.dump(clf,file)
         return Response("모델의 정확도 : " + str(accuracy), status=status.HTTP_200_OK)
        elif model_name == 'model_2':
         try:
             #clf = KNeighborsClassifier(**request.data)
             clf = RandomForestClassifier(**request.data)
             # clf = GradientBoostingClassifier(**request.data)
             clf.fit(X_train, y_train)
             #accuracy = clf.score(X_train, y_train) #학습 정확도 출력
             accuracy = clf.score(X_test, y_test) #테스트 정확도 출력
         except Exception as err:
             return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

         path = os.path.join(settings.MODEL_ROOT, model_name)
         with open(path,'wb') as file:
             pickle.dump(clf,file)
         return Response("모델의 정확도 : " + str(accuracy), status=status.HTTP_200_OK)
        else:
         try:
             #clf = KNeighborsClassifier(**request.data)
             #clf = RandomForestClassifier(**request.data)
             clf = GradientBoostingClassifier(**request.data)
             clf.fit(X_train, y_train)
             #accuracy = clf.score(X_train, y_train) #학습 정확도 출력
             accuracy = clf.score(X_test, y_test) #테스트 정확도 출력
         except Exception as err:
             return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

         path = os.path.join(settings.MODEL_ROOT, model_name)
         with open(path,'wb') as file:
             pickle.dump(clf,file)
         return Response("모델의 정확도 : " + str(accuracy), status=status.HTTP_200_OK)


# 결과 예측 예시
# class Predict(views.APIView):
#     def post(self, request):$
#         predictions = []
#         for entry in request.data:
#             model_name = entry.pop('model_name')
#             path = os.path.join(settings.MODEL_ROOT, model_name)
#             with open(path,'rb') as file:
#                 model = pickle.load(file)
#             try:
#                 result = model.predict(pd.DataFrame([entry]))
#                 predictions.append(result[0])
#
#             except Exception as err:
#                 return Response(str(err),status=status.HTTP_400_BAD_REQUEST)
#         return Response(predictions, status=status.HTTP_200_OK)


# 앞서 만들어진 모델들을 기반으로 예측할 데이터(Jason --> DataFrame)를 post후 예측 결과 출력
# 예측에 쓰이는 데이터는 Jason 형태로 주고 받는 Django Rest Framework 특성상 DataFrame 형태로 예측에 사용되어야함(array x)
class Predict(views.APIView):
    def post(self, request):
        #request.POST._mutable = True

        predictions = []
        for entry in request.data:
            model_name = entry.pop('model_name')
            path = os.path.join(settings.MODEL_ROOT, model_name)
            with open(path,'rb') as file:
                model = pickle.load(file)

            try:
                result = pd.DataFrame([entry])
                #result = result.reshape(1, -1)
                result = model.predict(result)
                predictions.append(result[0])

            except Exception as err:
                return Response(str(err),status=status.HTTP_400_BAD_REQUEST)

        return Response("예측 결과(Target) : " + str(predictions), status=status.HTTP_200_OK)