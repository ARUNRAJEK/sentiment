import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="LLM Sentiment Analysis", layout="centered")
st.title(" Sentiment Analysis ")

st.write("Please upload a CSV file. The file must contain a 'Review' column.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    st.subheader("Uploaded Dataset")
    st.dataframe(df)

    if "Review" not in df.columns:
        st.error("The CSV file must contain a column named 'Review' ❌")
    else:
        if st.button("Analyze Sentiment using LLM"):

            sentiments = []

            with st.spinner("Running LLM analysis..."):
                for text in df["Review"]:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant"
,
                        messages=[
                            {
                                "role": "user",
                                "content": f"""
                                Classify the sentiment of this review
                                as Positive, Negative, or Neutral.
                                Only give one word answer.

                                Review: {text}
                                """
                            }
                        ]
                    )

                    sentiment = response.choices[0].message.content.strip()
                    sentiments.append(sentiment)

            df["Sentiment"] = sentiments

            st.subheader("Sentiment Result")
            st.dataframe(df)

            # Count sentiment
            df["Sentiment"] = df["Sentiment"].str.strip()   # Remove spaces
            df["Sentiment"] = df["Sentiment"].str.replace(".", "", regex=False)  # Remove dot
            df["Sentiment"] = df["Sentiment"].str.capitalize()  # Standard format
            sentiment_count = df["Sentiment"].value_counts().reset_index()
            st.subheader("Sentiment Count")
            st.bar_chart(df["Sentiment"].value_counts())

