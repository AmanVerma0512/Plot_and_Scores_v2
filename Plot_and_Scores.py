from Plot_helper import FormScore,Scoring
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

class Interface():
    def __init__(self,h_params, y,log_dir=r"TestDir/"):
        plot_dir = log_dir + "plots/"
        self.fs=FormScore(y,plot_dir=plot_dir)
        self.fs.h_params=h_params
        response=Scoring(self.fs, "rep_and_threshold").scores()

# In[21]:

scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

cred = {
  "type": st.secrets["type_s"],
  "project_id": st.secrets["project_id_s"],
  "private_key_id": st.secrets["private_key_id_s"],
  "private_key": st.secrets["private_key_s"],
  "client_email": st.secrets["client_email_s"],
  "client_id": st.secrets["client_id_s"],
  "auth_uri": st.secrets["auth_uri_s"],
  "token_uri": st.secrets["token_uri_s"],
  "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url_s"],
  "client_x509_cert_url": st.secrets["client_x509_cert_url_s"]}
credentials = ServiceAccountCredentials.from_json_keyfile_dict(cred, scope)

client = gspread.authorize(credentials)
# Open the spreadhseet
sheet = client.open("DFIA").worksheet("Samples")

# In[6]:


df = pd.DataFrame(sheet.get_all_records())

# In[22]:


st.set_page_config(layout="wide")

st.title("Plots and Scores Dashboard")
signal_id = st.selectbox('Select a Signal ID', list(df["user_id"]),key="signal_id_p")

if st.button("Generate Plots and Model Response",key="plot_button"):
    signal_input = json.loads(df[df["user_id"] == signal_id]["signal"].values[0])
    score_dict = eval(df[df["user_id"] == signal_id]["response"].values[0])
    h_p = eval(df[df["user_id"] == signal_id]["h_params"].values[0])
    st.header("Plots for Signal ID: " + str(signal_id))
    Interface(h_p, signal_input)



    st.header("Model Scores for Signal ID: " + str(signal_id))
    st.write("Power: " + str(score_dict["power"]))  
    st.write("Sudden Metric: " + str(score_dict["form"]["sudden_metric"]))  
    st.write("Jitter Metric: " + str(score_dict["form"]["jitter_metric"]))  
    st.write("Inconsistent Tempo: " + str(score_dict["form"]["it_metric"]))  
    st.write("Blip Jitter Metric: " + str(score_dict["form"]["blip_jitter_metric"]))  
    st.write("Stamina: " + str(score_dict["ring stamina"]))  
    st.write("Rep Score: " + str(score_dict["rep_score"]))  
    st.write("User Form Score: " + str(score_dict["global_score"]))  
    st.write("Qualitative coaching tip provided by the model:")
    st.write(score_dict['coaching_tip'])
