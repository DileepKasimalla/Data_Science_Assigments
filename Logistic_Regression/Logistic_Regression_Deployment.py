import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from pickle import load

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    AGE = st.sidebar.number_input("Insert the Age", min_value=0)
    SIB_OR_SPO = st.sidebar.number_input("Insert Sib/Spo", min_value=0)
    PARCH = st.sidebar.number_input("Insert Parch", min_value=0)
    
    PCLASS_2 = st.sidebar.selectbox('Pclass_2', (1, 0))  # Converting to int
    Pclass_3 = st.sidebar.selectbox('Pclass_3', (1, 0))  # Converting to int
    SEX = st.sidebar.selectbox('Sex_male', (1, 0))       # Converting to int
    EMBARKED_Q = st.sidebar.selectbox('Embarked_Q', (1, 0))  # Converting to int
    EMBARKED_S = st.sidebar.selectbox('Embarked_S', (1, 0))  # Converting to int
    
    # Create a dictionary with the features
    data = {
        'AGE': AGE,
        'SIB_OR_SPO': SIB_OR_SPO,
        'PARCH': PARCH,
        'PCLASS_2': PCLASS_2,
        'Pclass_3': Pclass_3,
        'SEX': SEX,
        'EMBARKED_Q': EMBARKED_Q,
        'EMBARKED_S': EMBARKED_S
    }
    
    # Create a DataFrame from the user input
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df = user_input_features()

# Display user input parameters
st.subheader('User Input parameters')
st.write(df)

# Load the logistic regression model
loaded_model = load(open('Logistic_Model.sav', 'rb'))

# Make predictions
prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

# Display prediction results
st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)
