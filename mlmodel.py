#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets   
# from sklearn.tree import  DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.svm import  SVC
from sklearn.decomposition  import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("**Exploring Differrent Machine  Learning Models**")
dataset_names = st.sidebar.selectbox('Select Dataset',('Iris','Wine','Breast Cancer'))

classifier_names = st.sidebar.selectbox('Select  Classifier',('Decision Tree Classifier','KNN Classifier','SVM Classifier','Random Forest Classifier'))

def get_dataset(dataset_names):
    if dataset_names == 'Iris':
        data = datasets.load_iris()
        X = data.data
        y = data.target
        return X,y
    elif dataset_names == 'Wine':
        data = datasets.load_wine()
        X = data.data
        y = data.target
        return X,y
    elif dataset_names == 'Breast Cancer':
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        return X,y
    
X,y = get_dataset(dataset_names)

st.write("Shape of dataset",  X.shape)
st.write('number of classes',  len(np.unique(y)))

def  add_parameter_ui(classifier_names):
    params = dict()
    if classifier_names == 'SVM':
        C =  st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif  classifier_names == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth =  st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators =   st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
        return  params
params = add_parameter_ui(classifier_names)

def   get_classifier(classifier_names,params):
    if classifier_names == 'SVM':
        clf =   SVC(C=params['C'])
    elif classifier_names == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf =   RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1234)
    return clf

clf =  get_classifier(classifier_names,params)

#using train test split
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc =  accuracy_score(y_test,y_pred)
st.write('Accuracy:', acc)
st.write(f'Classifier = {classifier_names}')

# Apply PCA to reduce dimensions to 2D
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:,0]
x2 =  X_projected[:,1]
fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel('Principal Component 1 ')
plt.ylabel('Principal Component 2 ')
plt.colorbar()
#plt.show()
st.pyplot(fig)


        
        









