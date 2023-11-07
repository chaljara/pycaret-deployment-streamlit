from google.cloud import storage
from google.oauth2 import service_account
from pycaret.clustering import *
from pycaret.anomaly import AnomalyExperiment
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.colors as pc
import holoviews as hv

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
    
    data_g = pd.DataFrame(data_g_c, copy=True)

    data_pivot = data_g.pivot_table(index='ID', columns='WEEK', values=["CARD_DOWNTIME", "CASH_DOWNTIME", "ACCEPTOR_DOWNTIME", "DEPOSITOR_DOWNTIME", "EPP_DOWNTIME", "PRINTER_DOWNTIME"], aggfunc='mean', sort=False)
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
    
    merged = anomalies.reset_index()

if __name__ == '__main__':
    st.set_page_config(layout="wide")#, theme="light")
    
    load()

    evaluate()
    
    #Customers
    #customers = data_pivot_no_geo.sort_values(by="CUSTOMER", ascending=True)["CUSTOMER"].unique()
    st.subheader('Módulo de detección de anomalías', divider='red')
    
    customer_count = anomalies.groupby("CUSTOMER").agg(Cantidad = ("CUSTOMER","count")).reset_index()
    customer_count["CUSTOMER_B"] = customer_count["CUSTOMER"].astype(str) + "   (" + customer_count["Cantidad"].astype(str) + ")"
    n_anomalies = 0
    
    def custom_format(option):
        n_anomalies = customer_count.loc[customer_count["CUSTOMER"] == option]["Cantidad"]
        return customer_count.loc[customer_count["CUSTOMER"] == option].iat[0,2]
    
    customerSelected = st.selectbox("Seleccione un cliente: ", customer_count["CUSTOMER"], key="selectbox_customers", format_func=custom_format)
    st.subheader('')
    
    data_filtered = pd.DataFrame(merged, copy=True)
    data_filtered = data_filtered.loc[data_filtered["CUSTOMER"] == customerSelected]
    
    data_filtered =  pd.DataFrame({"ID": data_filtered["ID"],
                                    "FAMILY": data_filtered["FAMILY"],
                                    "FUNCTION": data_filtered["FUNCTION"],
                                    "SITE": data_filtered["SITE"], 
                                    "MODEL": data_filtered["MODEL"],
                                    "CARD_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[1:13]["value"]) for id in data_filtered["ID"]],
                                    "CASH_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[13:25]["value"]) for id in data_filtered["ID"]],
                                    "ACCEPTOR_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[25:37]["value"]) for id in data_filtered["ID"]],
                                    "DEPOSITOR_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[37:49]["value"]) for id in data_filtered["ID"]],
                                    "EPP_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[49:61]["value"]) for id in data_filtered["ID"]],
                                    "PRINTER_DOWTIME": [np.array(data_filtered.loc[data_filtered["ID"] == id].melt()[61:73]["value"]) for id in data_filtered["ID"]]
                                    })

    anomalies_by_customer = anomalies.loc[anomalies["CUSTOMER"] == customerSelected]
    
    df1 = anomalies_by_customer.groupby(['FAMILY', 'FUNCTION'])['W0'].count().reset_index()
    df1.columns = ['source', 'target', 'value']
    
    df2 = anomalies_by_customer.groupby(['FUNCTION', 'SITE'])['W0'].count().reset_index()
    df2.columns = ['source', 'target', 'value']
    
    df3 = anomalies_by_customer.groupby(['SITE', 'MODEL'])['W0'].count().reset_index()
    df3.columns = ['source', 'target', 'value']
    
    links = pd.concat([df1, df2, df3], axis=0)
    #unique_source_target = list(pd.unique(links[['source', 'target']].values.ravel('K')))
    #mapping_dict = {k: v for v, k in enumerate(unique_source_target)}
    #links['source'] = links['source'].map(mapping_dict)
    #links['target'] = links['target'].map(mapping_dict)
    #links_dict = links.to_dict(orient='list')
    
    #fig = go.Figure(data=[go.Sankey(
    #    node = dict(
    #      pad = 12,
    #      thickness = 10,
    #      #line = dict(color = "orange", width = 0.5),
    #      label = unique_source_target,
    #      color = "red"
    #    ),
    #link = dict(
    #      source = links_dict["source"],
    #      target = links_dict["target"],
    #      value = links_dict["value"],
    #     color = "lightgray"
    #  ))])
    #fig.update_layout(title_text="", 
                  #font=dict(size=16, color="black", family="Arial"),
                  #font_size=10, 
                  #width=250, 
                  #height=700,
                  #hovermode='y unified' )
    
    hv.extension('bokeh')
    links_filtered = links.loc[links["value"] > 0]
    
    def hide_hook(plot, element):
        plot.handles["xaxis"].visible = False
        plot.handles["yaxis"].visible = False 
        # plot.handles["xgrid"].visible = False
        # plot.handles["ygrid"].visible = False
        plot.handles["plot"].border_fill_color = None
        #plot.handles["plot"].background_fill_color = None
        plot.handles["plot"].outline_line_color = None
    
    #if len(links_filtered) > 0:
    sankey = hv.Sankey(links_filtered, label='')
    sankey.opts(hooks=[hide_hook], toolbar=None, default_tools = [], label_position='outer', edge_color='slategray', node_color='index', cmap='tab10', node_padding=15)

    st.dataframe(links_filtered.shape)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Cajeros anómalos detectados")
        st.dataframe(data_filtered.reset_index(drop=True), 
                     hide_index=False, 
                     use_container_width=True, 
                     column_config={
                        "ID": st.column_config.TextColumn(label="ATM ID", width="small"),
                        "FAMILY": st.column_config.TextColumn(label="FAMILIA", width="small"),
                        "FUNCTION": st.column_config.TextColumn(label="FUNCION", width="small"),
                        "SITE": st.column_config.TextColumn(label="TIPO", width="small"),
                        "MODEL": st.column_config.TextColumn(label="MODELO", width="small"),
                        "CARD_DOWTIME": st.column_config.LineChartColumn("TARJETA (s)", y_min=0, y_max=86400, width="small", help="Promedio semanal del tiempo de inactividad de la lectora de tarjetas"),
                        "CASH_DOWTIME": st.column_config.LineChartColumn("DISPENSADOR (s)", y_min=0, y_max=86400, width="small", help="Promedio semanal del tiempo de inactividad del dispensador de efectivo"),
                        "ACCEPTOR_DOWTIME": st.column_config.LineChartColumn("ACEPTADOR (s)", y_min=0, y_max=86400, width="small", help="Promedio semanal del tiempo de inactividad del aceptador de efectivo"),
                        "DEPOSITOR_DOWTIME": st.column_config.LineChartColumn("CHEQUE (s)", y_min=0, y_max=86400, width="small", help="Promedio semanal del tiempo de inactividad del depósito de cheques"),
                        "EPP_DOWTIME": st.column_config.LineChartColumn("TECLADO (s)", y_min=0, y_max=86400, width="small", help="Promedio semanal del tiempo de inactividad del teclado electrónico"),
                        "PRINTER_DOWTIME": st.column_config.LineChartColumn("IMPRESORA (s)", y_min=0, y_max=86400, width="small", help="Promedio semanal del tiempo de inactividad de la impresora de recibos"),
                    })
    with col2:
        #Diagrama Sanky
        st.subheader("Distribución jerárquica")
        #st.plotly_chart(fig, use_container_width=True)
        st.bokeh_chart(hv.render(sankey, backend='bokeh'))
        #st.bokeh_chart(sankey, use_container_width=True)

    if st.session_state.selectbox_customers != customerSelected:
        st.session_state.selectbox_customers = customerSelected
