import pandas as pd
from google.cloud import storage
#from google.colab import auth
from pycaret.clustering import *
from pycaret.anomaly import AnomalyExperiment
import streamlit as st

#model = load_model('deployment_28042020')
def load():
    #auth.authenticate_user()

    project_id = 'mcd-proyecto'
    bucket_name = "mcdproyectobucket"
    file_name = "dataset-v5-ofuscated.csv"
    kmeans_model_1 = "kmeans_model_downtime"
    kmeans_model_2 = "kmeans_model_downtime_grouped"
    
    storage_client = storage.Client(project=project_id)
    
    # Specify the bucket and file
    bucket = storage_client.get_bucket(bucket_name)
    
    blob = bucket.blob(file_name)
    downloaded_file_path = "/content/dataset.csv"
    blob.download_to_filename(downloaded_file_path)
    
    blob2 = bucket.blob(kmeans_model_1 + ".pkl")
    downloaded_file_path = "/content/" + kmeans_model_1 + ".pkl"
    blob2.download_to_filename(downloaded_file_path)
    
    blob3 = bucket.blob(kmeans_model_2 + ".pkl")
    downloaded_file_path = "/content/" + kmeans_model_2 + ".pkl"
    blob3.download_to_filename(downloaded_file_path)

def preprocess():
    data = pd.read_csv('/content/dataset.csv', sep=";", encoding="UTF-8")
    data['DATETIME'] = pd.to_datetime(data['DATETIME'])
    print(data.shape)
    
def predict(model, input_df):
    #predictions_df = predict_model(estimator=model, data=input_df)
    #predictions = predictions_df['Label'][0]
    return #predictions


def run():
    st.write(df)

if __name__ == '__main__':
    load()
    preprocess()
    run()
