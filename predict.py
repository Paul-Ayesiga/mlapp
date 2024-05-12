import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('model.pkl','rb') as file:
        data = pickle.load(file)
    return data

data  = load_model()

regressor= data['model']
le_Country = data['le_Country']
le_EdLevel = data['le_EdLevel']

def show_predict_page():
    st.title("Software Developer Salary Predictions")
    st.write("""### we need some to predict the salary """)

    countries = (
        "United States", 
        "India ", 
        "United Kingdom", 
        "Germany", 
        "Canada", 
        "Brazil", 
        "France", 
        "Spain", 
        "Australia", 
        "Netherlands", 
        "Poland", 
        "Italy", 
        "Russian Federation", 
        "Sweden", 
    )

    education = (
        'Bachelor`s Degree', 
        'Master`s Degree', 
        'Less than a Bachelors',
        'Post grad'
    )

    country = st.selectbox("Country",countries)
    education = st.selectbox("Education Level",education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")

    if ok:
        X = np.array([[country, education, experience]])
        X[:,0] = le_Country.transform(X[:,0])
        X[:,1] = le_EdLevel.transform(X[:,1])
        X = X.astype(float)
        
        Salary = regressor.predict(X)
        st.subheader(f"The Estimated Salary is ${Salary[0]:.2f}")
