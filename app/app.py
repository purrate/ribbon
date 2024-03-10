import streamlit as st
import csv
import os
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import requests
# Home Page

def get_deciding_factor(commenting, groping, facial_expression):
    # Define thresholds or rules for each factor
    commenting_threshold = 2.0  # Adjust as needed
    groping_threshold = 1.0  # Adjust as needed
    facial_expression_threshold = 1.0  # Adjust as needed

    # Compare values with thresholds
    if commenting >= commenting_threshold:
        return "Section 294 IPC [vi]\nThe section penalizes certain acts of obscenity in public places i.e., whoever sings, recites, utters any obscene song, ballad, words to the annoyance of others is made punishable."
    elif groping >= groping_threshold:
        return "Section 354 IPC [vii]\nThis section prescribes punishment for such acts of accused which not only causes insult or outrages the modesty of a woman but also causes or threatens to cause physical harm to her."
    elif facial_expression >= facial_expression_threshold:
        return "Section 354A [ix]\nAny man who shows pornography to a woman against her will, Makes sexually coloured remarks"
    else:
        return "No specific violation"
    
def home():
    st.title("Streamlit Navigation Example")
    st.write("Choose an option from the sidebar.")

# Predict Page
def predict_page():
    st.title("Prediction Page")
    text = st.text_area("Enter text:")
    if st.button("Print Text"):
        #
                            # Your input array
        #
        url = "http://127.0.0.1:5000/predict"  # Use the correct port number
        data = {'input_data': text}
        st.write(data)
        print(url,data)
        try:
            
            response = requests.post(url, json=data)
            st.write(response)

            print(response)
            if response.status_code == 200:
                result = response.json().get('output')
                if result is not None:
                    print(result)
                    data = np.array(result)

                    # Sum along each axis (0-indexed)
                    sums = np.sum(data, axis=0)

                    # Streamlit App
                    st.title("Bar Chart from Numpy Array")

                    # Display the original array
                    st.subheader("Original Numpy Array:")
                    st.write(data)

                    # Display the sums
                    st.subheader("Sums:")
                    st.write("Commenting:", sums[0])
                    st.write("Ogling:", sums[1])
                    st.write("Groping:", sums[2])

                    # Plotting the bar chart
                    fig, ax = plt.subplots()
                    fields = ['Commenting', 'Ogling', 'Groping']
                    ax.bar(fields, sums)
                    ax.set_ylabel('Sum')
                    ax.set_xlabel('Fields')
                    st.subheader("Bar Chart:")
                    st.pyplot(fig)
                    #
                    st.write("Prediction: ", result)
                    st.success(f"Prediction: {result}")
                    
                    st.write("Laws violated:", get_deciding_factor(sums[0], sums[1], sums[2]))
                    
                    url = "https://textanalysis-text-summarization.p.rapidapi.com/text-summarizer"

                    payload = {
                        "url": "http://en.wikipedia.org/wiki/Automatic_summarization",
                        "text": "",
                        "sentnum": 8
                    }
                    headers = {
                        "content-type": "application/json",
                        "X-RapidAPI-Key": "3a9d66c4a9mshfd39bb07ef66e54p18dc1fjsn96bb7aeaf350",
                        "X-RapidAPI-Host": "textanalysis-text-summarization.p.rapidapi.com"
                    }
    
                    response1 = requests.post(url, json=payload, headers=headers)

                    summary = response1.json()
                    
                    st.write("Summary:", summary)
                else:
                    st.error("The 'output' key is not present in the response.")
            else:
                st.error("Error in prediction.")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")


        #
        st.write(data)
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
    st.title("Social Media Page")
    st.markdown("Dummy Text in Bold")

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