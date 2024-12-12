import streamlit as st
import pandas as pd
import boto3
import joblib
import tempfile
import plotly.graph_objs as go

# Fetch model securely from S3
@st.cache_data
def fetch_model_from_s3():
    try:
        # Create a session using credentials from Streamlit secrets
        session = boto3.Session(
            aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["aws"]["AWS_DEFAULT_REGION"]
        )
        s3 = session.client("s3")

        # Fetch S3 bucket and model key from Streamlit secrets
        S3_BUCKET = st.secrets["s3"]["bucket_name"]
        MODEL_KEY = st.secrets["s3"]["model_key"]

        # Download the model file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3.download_file(S3_BUCKET, MODEL_KEY, temp_file.name)
            model_pipeline = joblib.load(temp_file.name)
        return model_pipeline

    except Exception as e:
        st.error("Failed to load the model. Please contact support.")
        raise

# Load the model
st.write("Fetching model from S3...")
model_pipeline = fetch_model_from_s3()
st.success("Model loaded successfully!")

# Streamlit UI
st.title("Moral Machines Prediction")
st.write("Input data to get predictions")

# Input fields
pedped = st.number_input("Pedestrian", min_value=0, max_value=1, value=0)
barrier = st.number_input("Barrier Present", min_value=0, max_value=1, value=0)
crossingsignal = st.number_input("Crossing Signal", min_value=0, max_value=2, value=0)
attribute_level = st.selectbox(
    "Attribute Level", 
    ["Hoomans", "Pets", "Female", "Male", "High", "Low", "Young", "Old", "Rand"]
)
user_country_3 = st.selectbox(
    "User Country", 
    ["USA", "CAN", "SGP", "CHN", "GBR", "ISR", "FRA", "DEU", "JPN", "KOR"]
)
review_political = st.number_input("Political Review", min_value=0, max_value=1, value=0)
review_religious = st.number_input("Religious Review", min_value=0, max_value=1, value=0)

# Prediction button
if st.button("Predict Saved Probability"):
    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            "pedped": pedped,
            "barrier": barrier,
            "crossingsignal": crossingsignal,
            "attribute_level": attribute_level,
            "user_country_3": user_country_3,
            "review_political": review_political,
            "review_religious": review_religious
        }])

        # Verify the model can predict probabilities
        if hasattr(model_pipeline, 'predict_proba'):
            # Get prediction probabilities
            prediction_proba = model_pipeline.predict_proba(input_data)[0]
            saved_probability = prediction_proba[1] * 100
            not_saved_probability = prediction_proba[0] * 100

            # Display textual results
            st.metric(label="Probability of Being Saved", value=f"{saved_probability:.2f}%")

            # Create a bar chart comparing probabilities
            bar_fig = go.Figure(data=[
                go.Bar(
                    x=['Not Saved', 'Saved'], 
                    y=[not_saved_probability, saved_probability],
                    text=[f'{not_saved_probability:.2f}%', f'{saved_probability:.2f}%'],
                    textposition='auto',
                )
            ])
            bar_fig.update_layout(
                title='Saved Probability Comparison',
                yaxis_title='Probability (%)'
            )
            st.plotly_chart(bar_fig)

        else:
            st.error("This model does not support probability prediction.")

    except Exception as e:
        st.error("An error occurred during prediction. Please try again later.")

# Add a second attribute selection for comparison
attribute_level_compare = st.selectbox(
    "Compare with Attribute Level", 
    ["Hoomans", "Pets", "Female", "Male", "High", "Low", "Young", "Old", "Rand"]
)

# Button for comparison
if st.button("Compare Attributes"):
    try:
        # Prepare input data for both attribute levels
        input_data_1 = pd.DataFrame([{
            "pedped": pedped,
            "barrier": barrier,
            "crossingsignal": crossingsignal,
            "attribute_level": attribute_level,
            "user_country_3": user_country_3,
            "review_political": review_political,
            "review_religious": review_religious
        }])
        input_data_2 = input_data_1.copy()
        input_data_2["attribute_level"] = attribute_level_compare

        # Get prediction probabilities for both attributes
        prob_1 = model_pipeline.predict_proba(input_data_1)[0][1] * 100
        prob_2 = model_pipeline.predict_proba(input_data_2)[0][1] * 100

        # Display results
        st.metric(f"Saved Probability ({attribute_level})", f"{prob_1:.2f}%")
        st.metric(f"Saved Probability ({attribute_level_compare})", f"{prob_2:.2f}%")

        # Visualization
        compare_fig = go.Figure(data=[
            go.Bar(
                x=[attribute_level, attribute_level_compare],
                y=[prob_1, prob_2],
                text=[f'{prob_1:.2f}%', f'{prob_2:.2f}%'],
                textposition='auto'
            )
        ])
        compare_fig.update_layout(
            title='Attribute Level Comparison',
            yaxis_title='Survival Probability (%)'
        )
        st.plotly_chart(compare_fig)

        # Display ethical implication analysis
        if prob_1 > prob_2:
            st.write(f"The AI system prioritizes `{attribute_level}` over `{attribute_level_compare}`, which could have ethical implications in scenarios where `{attribute_level_compare}` represents vulnerable groups.")
        else:
            st.write(f"The AI system prioritizes `{attribute_level_compare}` over `{attribute_level}`, highlighting a potential bias in favor of `{attribute_level_compare}`.")

    except Exception as e:
        st.error(f"Error comparing attributes: {e}")