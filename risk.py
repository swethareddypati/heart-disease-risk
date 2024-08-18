import pandas as pd
import pickle
import streamlit as st
def set_background_image(image_url):
    # Apply custom CSS to set the background image
    page_bg_img = '''
    <style>
    .stApp {
        background-position: top;
        background-image: url(%s);
        background-size: cover;
    }

    @media (max-width: 768px) {
        /* Adjust background size for mobile devices /
        .stApp {
            background-position: top;
            background-size: contain;
            background-repeat: no-repeat;
        }
    }
    </style>
    ''' % image_url
    st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    # Set the background image URL
    background_image_url = "https://media.istockphoto.com/id/1359314170/photo/heart-attack-and-heart-disease-3d-illustration.jpg?s=612x612&w=0&k=20&c=K5Y-yzsfs7a7CyuAw-B222EMkT04iRmiEWzhIqF0U9E="

    # Set the background image
    set_background_image(background_image_url)

    custom_css = """
       <style>
       body {
           background-color: #4699d4;
           color: #ffffff;
           font-family: Arial, sans-serif;
       }
       select {
           background-color: #000000 !important; / Black background for select box /
           color: #ffffff !important; / White text within select box /
       }
       label {
           color: #ffffff !important; / White color for select box label */
       }
       </style>
       """
    st.markdown(custom_css, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


# Streamlit App

st.title("Heart Disease Risk Prediction")

df = pd.read_csv('heart_chances2.csv')
# Load the trained model

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)


# User Input
st.markdown("""
    <style>
        /* Set font and size */
        body {
            font-family: 'Aharoni', sans-serif;
            font-size: 15px;
        }
    </style>
""", unsafe_allow_html=True)

age = st.slider("Age", min_value=1, max_value=100, value=25)
sex = st.radio('Sex', ['Male', 'Female'])
chest_pain = st.slider("Chest Pain Type", min_value=0, max_value=3, value=0)
blood_pressure = st.slider("Resting Blood Pressure", min_value=90, max_value=200, value=120)
chol = st.slider("Serum Cholesterol", min_value=100, max_value=600, value=200)
fast_blood_sugar = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
electrocardiographic = st.slider("Resting Electrocardiographic Results", min_value=0, max_value=2, value=0)
maximum_heart_rate = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise = st.slider("Exercise Induced Angina", min_value=0, max_value=1, value=0)
slp = st.slider("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=6.2, value=0.0)
caa = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
thall = st.slider("Thallium Heart Scan Results", min_value=0, max_value=3, value=1)
high_bp = st.radio("High Blood Pressure", ["Yes", "No"])


# Convert string inputs to numeric values
sex_mapping = {'Male': 0, 'Female': 1}
fast_blood_sugar_mapping = {'Yes': 1, 'No': 0}
high_bp_mapping = {'Yes': 1, 'No': 0}

sex = sex_mapping.get(sex, 0)
fast_blood_sugar = fast_blood_sugar_mapping.get(fast_blood_sugar, 0)
high_bp = high_bp_mapping.get(high_bp, 0)

# Create a DataFrame with the input features
X = {
    'age': [age],
    'sex': [sex],
    'chest_pain': [chest_pain],
    'blood_pressure': [blood_pressure],
    'chol': [chol],
    'fast_blood_sugar': [fast_blood_sugar],
    'electrocardiographic': [electrocardiographic],
    'maximum_heart_rate': [maximum_heart_rate],
    'exercise': [exercise],
    'slp': [slp],
    'caa': [caa],
    'thall': [thall],
    'high_bp': [high_bp]
}

# Create a DataFrame
input_df = pd.DataFrame(X)

# Ensure that the column names match the ones used during training
expected_columns = ['age', 'sex', 'chest_pain', 'blood_pressure', 'chol', 'fast_blood_sugar',
                    'electrocardiographic', 'maximum_heart_rate', 'exercise', 'slp', 'caa',
                    'thall', 'high_bp']

# Reorder columns to match the order during training
input_df = input_df.reindex(columns=expected_columns, fill_value=0)
st.write("Entire Dataset:")
st.write(df)

# Prediction button
if st.button('Predict'):
    # Make predictions
    prediction = random_forest_model.predict(input_df)[0]

    # Display the prediction
    st.write(f"Prediction: {prediction}")

    # Display interpretation based on the predicted value
    if prediction == 1:
        st.write("It is Risky!")
    else:
        st.write("It is not Risky.")


