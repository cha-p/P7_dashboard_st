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

print(model.predict(data_X))