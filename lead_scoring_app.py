import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set page config with logo and title
st.set_page_config(page_title="Lead Scoring - IT Vedant", page_icon="logo.jpeg", layout="wide")

# Set theme colors and background logo
st.markdown(
    f"""
    <style>
        :root {{
            --primary-color: #2E7D32;
            --secondary-color: #757575;
        }}
        .stButton>button {{
            background-color: var(--primary-color);
            color: white;
            border-radius: 8px;
        }}
        .stSuccess {{ color: var(--primary-color); }}
        .stError {{ color: red; }}
        body {{
            background-color: #F1F8E9;
        }}
        body::before {{
            content: "";
            background: url('logo.jpeg') no-repeat center center fixed;
            background-size: cover;
            opacity: 0.1;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            z-index: -1;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Header with Logo and Title
st.image("logo.jpeg", width=100)
st.title('Lead Scoring - IT Vedant')
st.write('Decode your dreams')

# Load dataset and print columns for debugging
data = pd.read_csv('Lead Scoring.csv')
st.sidebar.write("### IT Vedant Training Institute")
st.sidebar.write("Dataset Loaded Successfully!")

# Select relevant features
selected_features = [
    'Lead Source', 'TotalVisits', 'Total Time Spent on Website',
    'Page Views Per Visit', 'Last Activity', 'Specialization', 'City'
]
data = data[selected_features + ['Converted']]

# Encode categorical variables
label_encoders = {}
categorical_columns = data.select_dtypes(include='object').columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Train and save the model
@st.cache_resource
def train_and_save_model():
    X = data.drop('Converted', axis=1)
    y = data['Converted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    model.save_model('lead_scoring_model.json')
    return model

# Load the trained model
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('lead_scoring_model.json')
    return model

# Check if model is already trained, otherwise train it
try:
    model = load_model()
    st.sidebar.success('Model loaded successfully!')
except Exception as e:
    st.sidebar.warning('No saved model found. Training a new model...')
    model = train_and_save_model()
    st.sidebar.success('Model trained and saved as lead_scoring_model.json!')

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Input Data", "Predict Lead Score"])

if page == "Input Data":
    st.header('📝 Lead Scoring Input')
    st.write("Provide the necessary inputs and save them for prediction.")

    inputs = {}
    col1, col2 = st.columns(2)
    with col1:
        for i, col in enumerate(selected_features[:len(selected_features)//2]):
            if col in categorical_columns:
                options = label_encoders[col].classes_.tolist()
                inputs[col] = st.selectbox(f'{col}', options, index=None, placeholder=f"Select {col}")
            else:
                inputs[col] = st.number_input(f'{col}', min_value=0.0, step=0.1, format="%.2f")

    with col2:
        for i, col in enumerate(selected_features[len(selected_features)//2:]):
            if col in categorical_columns:
                options = label_encoders[col].classes_.tolist()
                inputs[col] = st.selectbox(f'{col}', options, index=None, placeholder=f"Select {col}")
            else:
                inputs[col] = st.number_input(f'{col}', min_value=0.0, step=0.1, format="%.2f")

    if st.button('💾 Save Inputs', use_container_width=True):
        if None in inputs.values():
            st.error("⚠️ Please fill in all fields before saving.")
        else:
            input_df = pd.DataFrame([inputs])
            input_df.to_csv('user_inputs.csv', index=False)
            st.success("✅ Inputs saved successfully! Go to the 'Predict Lead Score' page.")

elif page == "Predict Lead Score":
    st.header('📈 Lead Scoring Prediction')
    st.write("Predict the lead score using the saved inputs.")

    try:
        features = pd.read_csv('user_inputs.csv')
        for col, le in label_encoders.items():
            if col in features.columns:
                features[col] = le.transform(features[col].astype(str))

        if st.button('⚡ Predict Lead Score', use_container_width=True):
            with st.spinner('Predicting...'):
                prediction = model.predict(features)[0]
                st.success(f'🎉 The predicted lead score is: {prediction:.2f}')
    except FileNotFoundError:
        st.error("❌ No inputs found. Please fill the inputs on the 'Input Data' page first.")

st.sidebar.write("### How to run the app:")
st.sidebar.code("streamlit run lead_scoring_app.py")
