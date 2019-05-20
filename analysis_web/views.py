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

import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from pandas import plotting
# __all__ = [plotting]
#
iris_dataset=datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
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

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[3,2,1,5]])
prediction = knn.predict(X_new)
print("예측된 클래스:",prediction)
print("예측한 타겟의 이름:",iris_dataset['target_names'][prediction])
print("테스트 세트의 정확도: {:.2f}".format(knn.score(X_test,y_test)))

y_pred = knn.predict(X_test)
print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
class Train(views.APIView):
    def post(self, request):
        iris = datasets.load_iris()
        mapping = dict(zip(np.unique(iris.target),iris.target_names))

        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.DataFrame(iris.target).replace(mapping)
        model_name = request.data.pop('model_name')


        try:
            clf = RandomForestClassifier(**request.data)
            clf.fit(X,y)

        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        path = os.path.join(settings.MODEL_ROOT, model_name)
        with open(path,'wb') as file:
            pickle.dump(clf,file)
        return Response(status=status.HTTP_200_OK)


class Predict(views.APIView):
    def post(self, request):
        predictions = []
        for entry in request.data:
            model_name = entry.pop('model_name')
            path = os.path.join(settings.MODEL_ROOT, model_name)
            with open(path,'rb') as file:
                model = pickle.load(file)
            try:
                result = model.predict(pd.DataFrame([entry]))
                predictions.append(result[0])

            except Exception as err:
                return Response(str(err),status=status.HTTP_400_BAD_REQUEST)
        return Response(predictions, status=status.HTTP_200_OK)

