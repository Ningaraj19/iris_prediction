import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import load_iris
import streamlit as st

def load_data():
    iris = load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    x['species'] = iris.target
    return x,iris.target_names
x, target_names = load_data()
model = RandomForestClassifier()
model.fit(x.iloc[:, :-1], x['species'])

st.sidebar.title("Input Features for Iris Prediction")
sepal_length = st.sidebar.slider("Sepal Length", float(x['sepal length (cm)'].min()), float(x['sepal length (cm)'].max()), float(x['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(x['sepal width (cm)'].min()), float(x['sepal width (cm)'].max()), float(x['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length", float(x['petal length (cm)'].min()), float(x['petal length (cm)'].max()), float(x['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width", float(x['petal width (cm)'].min()), float(x['petal width (cm)'].max()), float(x['petal width (cm)'].mean()))

input_data = [sepal_length, sepal_width, petal_length, petal_width]
prediction = model.predict([input_data])
prediction_species = target_names[prediction[0]]
st.write("## Iris Species Prediction")
st.write(f"The predicted species is: {prediction_species}")
if prediction_species == 'setosa':
    st.image("img/setosa.jpg", caption='Iris Setosa')
elif prediction_species == 'versicolor':
    st.image("img/versicolor.jpg", caption='Iris Versicolor')
else:

    st.image("img/virginica.jpg", caption='Iris Virginica')
