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
file_name = "dataset-v6-testweek35-ofuscated.csv"
iforest_model_1 = "iforest_model_downtime"

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

    week=35
    data_g_c = pd.DataFrame(data_g, copy=True)
    
    for i in range(week+1, week+12):
      data_g_b = pd.DataFrame(data_g, copy=True)
      data_g_b["WEEK"] = i
      data_g_c = data_g_c.append(data_g_b).reset_index(drop=True)
      print(data_g_b.shape, data_g_c.shape)
    
    data_g = pd.DataFrame(data_g_c, copy=True)

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
    iforest_model = load_model(model_name=iforest_model_1, platform="gcp", authentication=auth)
    
    iforest_setup = AnomalyExperiment()
    result_iforest = iforest_setup.predict_model(iforest_model, data=data_pivot_no_geo)

    no_anomalies = result_iforest[result_iforest["Anomaly"] == 0]
    anomalies = result_iforest[result_iforest["Anomaly"] == 1]
    
    st.write("no_anomalies: ", no_anomalies.shape)
    st.write("anomalies: ", anomalies.shape)
    st.dataframe(anomalies)
    
    merged = anomalies.reset_index(drop=True)

if __name__ == '__main__':
    load()

    evaluate()

    st.set_page_config(
        #page_title="Ex-stream-ly Cool App",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        #menu_items={
        #    'Get Help': 'https://www.extremelycoolapp.com/help',
        #    'Report a bug': "https://www.extremelycoolapp.com/bug",
        #    'About': "# This is a header. This is an *extremely* cool app!"
        #}
    )
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

    st.dataframe(data_filtered, hide_index=True, 
                 column_config={
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
