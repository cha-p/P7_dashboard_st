#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:52:03 2021
Dashboard projet 7
To run : streamlit run P7_dashboard.py
@author: charlottepostel
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
import plotly.express as px
import plotly.graph_objects as go
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import data
data = pd.read_csv("app_test.csv")
model = pickle.load(open("scoring_model_f2.sav", 'rb'))

data_X = data.drop(columns="SK_ID_CURR")
# print(model.predict(data_X))


# Layout & Navigation panel
st.set_page_config(page_title="Dashboard",
                   page_icon="☮",
                   initial_sidebar_state="expanded")
sb = st.sidebar # add a side bar 
sb.write('# Sommaire')
sb.write('###')
rad1 = sb.radio('Pages',('🏠 Accueil', 
                         '👨‍ Données relatives aux clients'))

# ecrire fonction pour faire page ex et P pour un client donné
#def boxplot(y_app, valeur):
    #fig = go.Figure()
    #fig.add_trace(go.Box(
        #y=y_app,
        #boxpoints=False,
        #boxmean=True,
        #name='essai'))
    #fig.add_hline(y=valeur, line_width=3, line_color="red")
    #fig.update_layout(autosize=False,
                      #width=500,
                      #height=500)
    #fig.show()

# Déroulement menu en fonction choix
if rad1 == '🏠 Accueil': # with this we choose which container to display on the screen
    st.title("Dashboard\n ----")
    st.header("**OpenClassrooms Data Scientist, Projet 7, Sept. 2021**")
    st.markdown("This project was composed of two main objectives:")
    st.markdown("- **Develop a scoring machine learning model** to predict the solvency of clients of a bank-like company (i.e. probability of credit payment failure). It is therefore a **binary classification issue**. Class 0 is solvent client whereas class 1 represents clients with payment difficulties.")
    st.markdown("- **Build an interactive dashboard** allowing interpretations of these probabilities and improve the company's knowledge on its clients.")
    st.markdown("")
    st.markdown("Vous pouvez sélectionner l'identifiant du client souhaité sur la barre de gauche.")
elif rad1 == '👨‍ Données relatives aux clients':
    #np.random.seed(13) # one major change is that client is directly asked as input since sidebar
    sb.write('### ')
    label_test = data['SK_ID_CURR'].sort_values()
    input_client = sb.selectbox("Veuillez sélectionner l'identifiant du client", label_test)
    sb.write('### ')
    #sb.markdown('## Données ou prédiction ?')
    rad2 = sb.radio('Données ou prédiction ?',['🔎 Exploration des données',
                                               '📉 Prédiction'])
    

    if rad2 == '🔎 Exploration des données':
        st.subheader("**Exploration des données**")
        st.write('**ID client : **', str(input_client))
        st.markdown("Données brutes:")
        st.write(data[data['SK_ID_CURR']==input_client])
        st.write('###')
        col1, col2 = st.columns(2)
        with col1:
            input_d1 = st.selectbox("Veuillez sélectionner la variable souhaitée", data_X.columns.values)
            valeur = data.loc[data['SK_ID_CURR']==input_client, input_d1]
            st.write("Valeur : ", str(valeur.to_numpy()[0]))
            if data[str(input_d1)].dtypes == 'O':
                x = data[input_d1].value_counts()/len(data)*100
                fig = px.pie(values=x,
                             names=data[input_d1].value_counts().index)
                fig.update_traces(textposition='outside', textinfo='percent+label')
                fig.update(layout_showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            elif  set(data[str(input_d1)].unique()) == set([0,1]):
                x = data[input_d1].value_counts()/len(data)*100
                fig = px.pie(values=x,
                             names=data[input_d1].value_counts().index)
                fig.update_traces(textposition='outside', textinfo='percent+label')
                fig.update(layout_showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure()
                fig.add_trace(go.Box(y=data[str(input_d1)],
                                     boxpoints=False,
                                     boxmean=True,
                                     name=str(input_d1)))
                if str(valeur.values[0]) != 'nan':
                    fig.add_hline(y=valeur.values[0], line_width=3, line_color="red")
                #fig.update_layout(autosize=False,
                              #width=500,
                              #height=500)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            input_d2 = st.selectbox("", data_X.columns.values)
            valeur2 = data.loc[data['SK_ID_CURR']==input_client, input_d2]
            st.write("Valeur : ", str(valeur2.to_numpy()[0]))
            fig2 = go.Figure()
            fig2.add_trace(go.Box(y=data[str(input_d2)],
                                 boxpoints=False,
                                 boxmean=True,
                                 name=str(input_d2)))
            fig2.add_hline(y=valeur2.values[0], line_width=3, line_color="red")
            #fig2.update_layout(autosize=False,
                              #width=500,
                             # height=500)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.header("**Prédiction**")
        st.subheader(input_client)
        st.markdown("This project was composed of two main objectives:")

