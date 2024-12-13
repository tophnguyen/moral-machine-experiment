import streamlit as st
import pandas as pd
import boto3
import joblib
import tempfile
import plotly.graph_objs as go
import plotly.express as px

# fetch model from S3 using secrets 
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

        # fetch S3 bucket and model key from Streamlit secrets
        S3_BUCKET = st.secrets["s3"]["bucket_name"]
        MODEL_KEY = st.secrets["s3"]["model_key"]

        # download the model file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3.download_file(S3_BUCKET, MODEL_KEY, temp_file.name)
            model_pipeline = joblib.load(temp_file.name)
        return model_pipeline

    except Exception as e:
        st.error("Failed to load the model. Please contact support.")
        raise

# load the model
st.write("Fetching model from S3...")
model_pipeline = fetch_model_from_s3()
st.success("Model loaded successfully!")

# streamlit UI
st.title("Moral Machines Prediction")
st.write("Input data to get predictions")

# input fields
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

# prediction button
if st.button("Predict Saved Probability"):
    try:
        # prep input data
        input_data = pd.DataFrame([{
            "pedped": pedped,
            "barrier": barrier,
            "crossingsignal": crossingsignal,
            "attribute_level": attribute_level,
            "review_political": review_political,
            "review_religious": review_religious
        }])

        # verify the model can predict probabilities
        if hasattr(model_pipeline, 'predict_proba'):
            # probabilities
            prediction_proba = model_pipeline.predict_proba(input_data)[0]
            saved_probability = prediction_proba[1] * 100
            not_saved_probability = prediction_proba[0] * 100

            # display textual results
            st.metric(label="Probability of Being Saved", value=f"{saved_probability:.2f}%")

            # create a bar chart comparing probabilities
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

# contextual analysis for country only
st.subheader("Saved Probability by Country")
st.write("Analyze how saved probabilities vary across countries for the selected attribute")

# name country values
countries = ["USA", "CAN", "SGP", "CHN", "GBR", "ISR", "FRA", "DEU", "JPN", "KOR"]

# generate predictions for each country
country_data = []
for country in countries:
    input_data = pd.DataFrame([{
        "pedped": pedped,
        "barrier": barrier,
        "crossingsignal": crossingsignal,
        "attribute_level": attribute_level,
        "user_country_3": country,
        "review_political": review_political,
        "review_religious": review_religious
    }])
    saved_probability = model_pipeline.predict_proba(input_data)[0][1] * 100
    country_data.append({"Country": country, "Saved Probability": saved_probability})

# convert results to DataFrame
country_df = pd.DataFrame(country_data)

# create a bar chart for country comparison
fig = px.bar(
    country_df,
    x="Country",
    y="Saved Probability",
    text="Saved Probability",
    title="Saved Probability by Country",
    labels={"Country": "Country", "Saved Probability": "Probability (%)"},
    template="plotly_white"
)
fig.update_traces(texttemplate='%{text:.2f}%', textposition="outside")

# display the chart
st.plotly_chart(fig)