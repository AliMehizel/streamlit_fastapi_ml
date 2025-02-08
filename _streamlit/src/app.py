import streamlit as st
import requests
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns



base_URL = 'http://127.0.0.1:8000'


def base_model(API_URL = base_URL + '/predict/'):

    """
    This function return Reccurence prediction from FastAPI.
    """

    st.title("Cancer Recurrence Prediction (Base model)")


    age = st.number_input("Enter Age:", min_value=0, max_value=120, step=1)
    tumor_size = st.number_input("Enter Tumor Size (in cm):", min_value=0.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status:", [0, 1, 2])


    if st.button("Predict Recurrence"):
        
        payload = {
        "age": age,
        "tumor_size": tumor_size,
        "smoking_status": smoking_status
        }

        response = requests.post(API_URL, json=payload)
    

        if response.status_code == 200:
            result = response.json()["prediction"]
            st.write(f"### Prediction: {result}")
        else:
            st.write("Error in prediction.")



def plot_data(data, plot_type, x, y):

    fig, ax = plt.subplots(figsize=(8, 6))  
    if plot_type == "Bar":
        sns.barplot(x=x, y=y, data=data, ax=ax)

    elif plot_type == "Box":
        sns.boxplot(x=x, y=y, data=data, ax=ax)

    elif plot_type == "Scatter":
        sns.scatterplot(x=x, y=y, data=data, ax=ax)

    elif plot_type == "Line":
        sns.lineplot(x=x, y=y, data=data, ax=ax)

    st.pyplot(fig)

def dashboad():

    """
    This function is responsible for vizualisation.
    """
    st.title("Dashboard")
    try: 
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
        data = pd.read_csv(uploaded_file)

        cat_columns = data.select_dtypes(include='object').columns
        data[cat_columns] = data[cat_columns].apply(lambda x: x.fillna(x.mode()[0]))

        num_columns = data.select_dtypes(exclude='object').columns
        data[num_columns] = data[num_columns].apply(lambda x: x.fillna(x.median()))

        plot_type = st.selectbox("Select the plot type", ["Bar", "Box", "Scatter", "Line"])
        x_axis_feature = st.selectbox("Select the feature for X-axis", data.columns)
        y_axis_feature = st.selectbox("Select the feature for Y-axis", data.columns)

        if uploaded_file is not None:
            plot_data(data,plot_type, x_axis_feature, y_axis_feature)
        else:
            st.write("Please upload a CSV file to start.")
    except Exception as e:
        st.error(f"Error reading the file: {e}")

   


    
        
        

 

def train_and_pred(base_URL=base_URL):

    """
    This function used to train and predict (basics model).
    """

    st.title("Train your own model")

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)


        st.write("Dataset preview:")
        st.dataframe(data.head())


        features = st.multiselect("Select the features (X)", options=data.columns.tolist())
        target = st.selectbox("Select the target variable (y)", options=data.columns.tolist())

        if len(features) > 0 and target:
    
            if st.button("Train Model"):
 
                X = data[features].fillna(0).values.tolist()
                y = data[target].fillna(0).values.tolist()
                response = requests.post(f"{base_URL}/train/", json={"X": X, "y": y})

                if response.status_code == 200:
                    st.success("Model trained successfully!")
                else:
                    st.error(f"Error: {response.json()['error']}")

            st.subheader("Make Predictions")

            prediction_data = []
            for feature in features:
                value = st.number_input(f"Enter {feature}", min_value=float(data[feature].min()), max_value=float(data[feature].max()))
                prediction_data.append(value)

            if st.button("Predict"):

                response = requests.post(f"{base_URL}/predict2/", json={"X": [prediction_data]})
                if response.status_code == 200:
                    predictions = response.json().get("predictions")
                    st.write(f"Predicted value: {predictions[0]}")
                else:
                    try:
                        error_message = response.json().get('error', 'Unknown error occurred')
                        st.error(f"Error: {error_message}")
                        
                    except Exception as e:
                        st.error(f"Error: Unable to parse response. {str(e)}")
                        st.write(response.text)  
    else:
        st.write("Please upload a CSV file to start.")





def base():

    """
    This function handle pages and others features.
    """
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Base Model Prediction", "Dashboard", "Train and Predict"])

    if page == 'Base Model Prediction':
        base_model()
    elif page == 'Dashboard':
        dashboad() 
    else:
        train_and_pred()


    pass



if __name__ == '__main__':

    base()




