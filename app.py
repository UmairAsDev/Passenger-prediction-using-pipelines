import streamlit as st
import pickle
import numpy as np

st.title('Passenger survival prediction using pipelines')

pipe = pickle.load(open('pipe.pkl','rb'))

Pclass = st.selectbox('pclass',[1,2,3])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.number_input('Age', step=0.2)
SibSp = st.number_input('siblings/sapouse', step=0)
Parch = st.number_input('parents', step=0)
Fare = st.number_input('Fare', step=0.2)
Embarked = st.selectbox('Embarked', ['C', 'S', 'E'])


if st.button('predict'):
    
    test_input2 = np.array([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked],dtype=object).reshape(1,7)
    prediction = pipe.predict(test_input2)
    
    if prediction == 0:
        st.success('passenger survived.....')
    else:
        st.error(" passenger was not survived....")