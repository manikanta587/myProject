import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import streamlit as st

# Load data
data = pd.read_csv("C:\\Users\\*******\\Desktop\\heart1.csv")

# Prepare data
X = data.drop('target', axis=1)
y = data['target']

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
nb_model = GaussianNB()

# Fit models
rf_model.fit(X, y)
gb_model.fit(X, y)
nb_model.fit(X, y)

# Soft Voting Ensemble
voting_ensemble = VotingClassifier(estimators=[
    ('RandomForest', rf_model),
    ('GradientBoosting', gb_model),
    ('NaiveBayes', nb_model)
], voting='soft')

voting_ensemble.fit(X, y)

# Streamlit app
st.title('Heart Disease Prediction')
st.sidebar.header('User Input')

# Sidebar inputs
age = st.sidebar.slider('Age', min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
cp = st.sidebar.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
chol = st.sidebar.slider('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'])
oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, value=2.0)
slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=1)
thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Map input values to numerical values
sex = 1 if sex == 'Male' else 0
cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
cp = cp_map[cp]
fbs = 1 if fbs == 'Yes' else 0
restecg_map = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
restecg = restecg_map[restecg]
exang = 1 if exang == 'Yes' else 0
slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
slope = slope_map[slope]
thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
thal = thal_map[thal]

# Create input data for prediction
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Function to make prediction
def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]
    return prediction, prediction_proba

# Predict button
if st.sidebar.button('Predict'):
    prediction, prediction_proba = make_prediction(voting_ensemble, input_data)

    # Display prediction
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('The patient is predicted to have Heart Disease.')
    else:
        st.write('The patient is predicted to not have Heart Disease.')
        # Display prediction probability
        st.subheader('Prediction Probability')
        st.write(f"Probability of Heart Disease: {prediction_proba[0]:.2f}")
    

    
