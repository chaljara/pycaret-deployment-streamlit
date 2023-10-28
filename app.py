from google.cloud import storage
from google.oauth2 import service_account
from pycaret.clustering import *
from pycaret.anomaly import AnomalyExperiment
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

project_id = 'mcd-proyecto'
bucket_name = "mcdproyectobucket"
file_name = "dataset-v5-ofuscated.csv"
kmeans_model_1 = "kmeans_model_downtime"
kmeans_model_2 = "kmeans_model_downtime_grouped"
iforest_model_1 = "iforest_model_downtime"
iforest_model_2 = "iforest_model_downtime_grouped"

#dataframes
data = []
data_g = []
categories = []
data_pivot = []
data_pivot_no_geo = []
cluster_anomaly = []
anomalies = []
merged = []

@st.cache_data
def load():
    
    credentials = service_account.Credentials.from_service_account_file("google-credentials.json")
    storage_client = storage.Client(project=project_id, credentials=credentials)
    
    # Specify the bucket and file
    bucket = storage_client.get_bucket(bucket_name)
    
    blob = bucket.blob(file_name)
    dataset_filename = "dataset.csv"
    
    blob.download_to_filename(dataset_filename)
    
    blob2 = bucket.blob(kmeans_model_1 + ".pkl")
    kmeans1_filename = kmeans_model_1 + ".pkl"
    blob2.download_to_filename(kmeans1_filename)
    
    blob3 = bucket.blob(kmeans_model_2 + ".pkl")
    kmeans2_filename = kmeans_model_2 + ".pkl"
    blob3.download_to_filename(kmeans2_filename)


    
def evaluate():
    global data
    global data_g
    global categories
    global data_pivot
    global data_pivot_no_geo
    global cluster_anomaly
    global anomalies
    global merged
    
    data = pd.read_csv("dataset.csv", sep=";", encoding="UTF-8")
    data['DATETIME'] = pd.to_datetime(data['DATETIME'])

    categories = data[["CUSTOMER", "ID", "MODEL", "FUNCTION", "FAMILY", "SITE", "STATE", "CITY", "COUNTRY"]]
    categories = categories.drop_duplicates(keep='first').reset_index()
    categories.set_index("ID", inplace=True)
    categories.drop(columns=["index"], inplace=True)

    downtime = data[["ID", "DATETIME", "WEEK", "CARD_DOWNTIME", "CASH_DOWNTIME", "ACCEPTOR_DOWNTIME",
                     "DEPOSITOR_DOWNTIME", "EPP_DOWNTIME", "PRINTER_DOWNTIME"]]

    data_g = downtime.groupby(by=["ID", "WEEK"]).agg(
                CARD_DOWNTIME = ("CARD_DOWNTIME", "mean"),
                CASH_DOWNTIME = ("CASH_DOWNTIME", "mean"),
                ACCEPTOR_DOWNTIME = ("ACCEPTOR_DOWNTIME", "mean"),
                DEPOSITOR_DOWNTIME = ("DEPOSITOR_DOWNTIME", "mean"),
                EPP_DOWNTIME = ("EPP_DOWNTIME", "mean"),
                PRINTER_DOWNTIME = ("PRINTER_DOWNTIME", "mean")
             ).reset_index()

    data_pivot = data_g.pivot_table(index='ID', columns='WEEK', values=["CARD_DOWNTIME", "CASH_DOWNTIME", "ACCEPTOR_DOWNTIME", "DEPOSITOR_DOWNTIME", "EPP_DOWNTIME", "PRINTER_DOWNTIME"], aggfunc='mean') #
    data_pivot.columns = [f'W{i}' for i in range(data_pivot.columns.size)]
    data_pivot_week_columns = data_pivot.columns
    
    data_pivot_no_geo = pd.DataFrame(data_pivot, copy=True)
    data_pivot_no_geo["CUSTOMER"] = categories["CUSTOMER"]
    data_pivot_no_geo["MODEL"] = categories["MODEL"]
    data_pivot_no_geo["FUNCTION"] = categories["FUNCTION"]
    data_pivot_no_geo["FAMILY"] = categories["FAMILY"]
    data_pivot_no_geo["SITE"] = categories["SITE"]
    data_pivot_no_geo["COUNTRY"] = categories["COUNTRY"]


    step = 12
    columnsByDevice = {}
    columnsByDeviceEvents = {}
    
    for i in range(0, len(data_pivot_week_columns.values), step):
        chunk = data_pivot_week_columns[i:i+step].values
        key = f'group_{i//step + 1}'
        columnsByDevice[key] = chunk
    
    auth = {"project": project_id, "bucket": bucket_name}
    kmeans_no_geo_downtime = load_model(model_name=kmeans_model_1, platform="gcp", authentication=auth)
    kmeans_no_geo_downtime_grouped = load_model(model_name=kmeans_model_2, platform="gcp", authentication=auth)
    iforest_downtime = load_model(model_name=iforest_model_1, platform="gcp", authentication=auth)
    iforest_downtime_grouped = load_model(iforest_model_2, platform="gcp", authentication=auth)
    
    kmeans_model = ClusteringExperiment()
    
    iforest_model = AnomalyExperiment()
    iforest_model_grouped = AnomalyExperiment()

    kmeans_model.setup(data_pivot_no_geo,
                            normalize = True,
                            ignore_features=["ID"],
                            ordinal_features=None,
                            session_id = 3,
                            categorical_features=["CUSTOMER", "MODEL", "FUNCTION", "FAMILY", "SITE", "COUNTRY"])
    
    iforest_model.setup(data_pivot_no_geo, session_id = 123, ignore_features=["ID"])
    
    kmeans_model.evaluate_model(kmeans_no_geo_downtime)
    iforest_model.evaluate_model(iforest_downtime)
    
    result_kmeans = kmeans_model.assign_model(kmeans_no_geo_downtime)
    result_iforest = iforest_model.assign_model(iforest_downtime)

    #Anomaly label
    cluster_count = result_kmeans.groupby("Cluster").agg(Count = ("Cluster", "count")).reset_index().sort_values(by="Count", ascending=True).reset_index(drop=True)
    anomaly_cluster_label = cluster_count.iloc[0,0]
    
    cluster_anomalies = result_kmeans.loc[result_kmeans["Cluster"] == anomaly_cluster_label]

    st.write("Cluster label: ", anomaly_cluster_label)
    st.write("Cluster elements: ", cluster_anomalies.shape)
    
    #kmeans_labels = result_kmeans["Cluster"].reset_index()
    iforest_labels = result_iforest.loc[:,["Anomaly", "Anomaly_Score"]].reset_index()
    iforest_anom_count = iforest_labels.loc[iforest_labels["Anomaly"] == 1].shape
    st.write("iforest elements: ", iforest_anom_count)
    merged = pd.merge(cluster_anomalies, iforest_labels, on='ID')
    
    merged = merged.loc[merged["Anomaly"] == 1].reset_index(drop=True)

if __name__ == '__main__':
    load()

    evaluate()
    
    #Customers
    customers = data_pivot_no_geo.sort_values(by="CUSTOMER", ascending=True)["CUSTOMER"].unique()
    st.subheader('Módulo de detección de anomalías', divider='orange')
    
    #col1, col2 = st.columns(2)
    
    #with col1:
    customerSelected = st.selectbox(
        "Seleccione un cliente: ",
        customers,
        key="selectbox_customers"#,
        #on_change=lambda new_option: st.write(f"Seleccionaste: {customerSelected}")
    )
    
    data_filtered = pd.DataFrame(merged, copy=True)
    data_filtered = data_filtered.loc[data_filtered["CUSTOMER"] == customerSelected]#[["ID", "MODEL", "FUNCTION", "FAMILY", "SITE"]]

    #data_g.loc[data_g["ID"] == "0000MTA"]["CARD_DOWNTIME"]
    
    data_filtered =  pd.DataFrame({
        "ID": data_filtered["ID"],
        "FAMILY": data_filtered["FAMILY"],
        "FUNCTION": data_filtered["FUNCTION"],
        "MODEL": data_filtered["MODEL"],
        "SITE": data_filtered["SITE"], 
        "CARD_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[1:13]["value"]) for id in data_filtered["ID"]],
        "CASH_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[13:25]["value"]) for id in data_filtered["ID"]],
        "ACCEPTOR_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[25:37]["value"]) for id in data_filtered["ID"]],
        "DEPOSITOR_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[37:49]["value"]) for id in data_filtered["ID"]],
        "EPP_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[49:61]["value"]) for id in data_filtered["ID"]],
        "PRINTER_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[61:73]["value"]) for id in data_filtered["ID"]],
    })

    st.dataframe(data_filtered, hide_index=True, column_config={
                    "ID": "ATM",
                    "FAMILY": "FAMILIA",
                    "FUNCTION": "FUNCION",
                    "MODEL": "MODELO",
                    "SITE": "TIPO",
                    "CARD_DOWTIME": st.column_config.LineChartColumn("TARJETA", y_min=0, y_max=86400),
                    "CASH_DOWTIME": st.column_config.LineChartColumn("DISPENSADOR", y_min=0, y_max=86400),
                    "ACCEPTOR_DOWTIME": st.column_config.LineChartColumn("ACEPTADOR", y_min=0, y_max=86400),
                    "DEPOSITOR_DOWTIME": st.column_config.LineChartColumn("CHEQUE", y_min=0, y_max=86400),
                    "EPP_DOWTIME": st.column_config.LineChartColumn("TECLADO", y_min=0, y_max=86400),
                    "PRINTER_DOWTIME": st.column_config.LineChartColumn("IMPRESORA", y_min=0, y_max=86400),
                })
        
    if st.session_state.selectbox_customers != customerSelected:
        st.session_state.selectbox_customers = customerSelected
