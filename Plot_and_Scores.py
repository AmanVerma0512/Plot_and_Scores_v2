#!/usr/bin/env python
# coding: utf-8

# In[20]:
import streamlit as st
import pandas as pd
import matplotlib as plt
from PIL import Image
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import ast


# In[21]:



scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

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
    signal_input = ast.literal_eval(df[df["user_id"] == signal_id]["signal"].values[0])
#     plot_path = df[df["user_id"] == signal_id]["plots"][0]
    score_dict = eval(df[df["user_id"] == signal_id]["response"].values[0])

#    st.header("Plots for Signal ID: " + str(signal_id))
#     plot_peaks = Image.open(plot_path+"peaks.png")
#     st.image(plot_peaks, caption='Plot indicating peaks signal for signal ID ' + str(signal_id))

#     plot_peaks_tempo = Image.open(plot_path+"peaks&tempo.png")
#     st.image(plot_peaks_tempo, caption='Plot indicating peaks & Tempo signal for signal ID ' + str(signal_id))

#     plot_jitter = Image.open(plot_path+"jitter.png")
#     st.image(plot_jitter, caption='Plot indicating jitter signal for signal ID ' + str(signal_id))

#     plot_smooth_blips = Image.open(plot_path+"smooth-blips.png")
#     st.image(plot_smooth_blips, caption='Plot indicating smooth blips signal for signal ID ' + str(signal_id))

#     plot_sudden_release = Image.open(plot_path+"sudden-release.png")
#     st.image(plot_sudden_release, caption='Plot indicating sudden release signal for signal ID ' + str(signal_id))


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