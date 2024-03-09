import streamlit as st
import csv
import os
import pandas as pd
# Home Page
def home():
    st.title("Streamlit Navigation Example")
    st.write("Choose an option from the sidebar.")

# Predict Page
def predict_page():
    st.title("Prediction Page")
    text = st.text_area("Enter text:")
    if st.button("Print Text"):
        st.write("Entered Text: ", text)
        cleaned_text = ",".join(text.split())  # Convert to comma-separated format
        st.text_area("Cleaned Text:", cleaned_text)
        preprocessed_text = text.replace('" ', '", "').replace(' "', '", "').replace(' ', ',')

        data_lines = preprocessed_text.splitlines()

# Create a list of lists to store the CSV data
        csv_data = []
        for line in data_lines:
         values = line.split(",")  # Adjust the delimiter if it's not ","
         csv_data.append(values)

# Create a Pandas DataFrame
        df = pd.DataFrame(csv_data)

# Write the DataFrame to a CSV file
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="data.csv",
            mime="text/csv",
        )

# View Data Page
def view_data_page():
    st.title("View Data Page")
    st.markdown("**Dummy Text in Bold**")

# Main App
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose an option", ["Home", "Predict Page", "View Data"])

    if app_mode == "Home":
        home()
    elif app_mode == "Predict Page":
        predict_page()
    elif app_mode == "View Data":
        view_data_page()

# Run the app
if __name__ == "__main__":
    main()
