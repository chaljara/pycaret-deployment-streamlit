import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from pycaret.clustering import *
from pycaret.anomaly import AnomalyExperiment
import streamlit as st

data = []

def load():
    project_id = 'mcd-proyecto'
    bucket_name = "mcdproyectobucket"
    file_name = "dataset-v5-ofuscated.csv"
    kmeans_model_1 = "kmeans_model_downtime"
    kmeans_model_2 = "kmeans_model_downtime_grouped"
    
    credentials = service_account.Credentials.from_service_account_file("google-credentials.json")
    storage_client = storage.Client(project=project_id, credentials=credentials)
    
    # Specify the bucket and file
    bucket = storage_client.get_bucket(bucket_name)
    st.write(bucket_name)
    blob = bucket.blob(file_name)
    #downloaded_file_path = "/content/dataset.csv"
    downloaded_file_path = "dataset.csv"
    st.write(downloaded_file_path)
    
    blob.download_to_filename(downloaded_file_path)
    
    blob2 = bucket.blob(kmeans_model_1 + ".pkl")
    downloaded_file_path = kmeans_model_1 + ".pkl"
    blob2.download_to_filename(downloaded_file_path)
    
    blob3 = bucket.blob(kmeans_model_2 + ".pkl")
    downloaded_file_path = kmeans_model_2 + ".pkl"
    blob3.download_to_filename(downloaded_file_path)   
    
def getData():
    data = pd.read_csv("dataset.csv", sep=";", encoding="UTF-8")
    data['DATETIME'] = pd.to_datetime(data['DATETIME'])
    return data
       

if __name__ == '__main__':
    load()
    
    data = getData()
    
    st.write(data) 
