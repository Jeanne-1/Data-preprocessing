# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:52:10 2024

Functions used in my app.py that do not need streamlit.

@author: jeann
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from basic_display_functions import filtered_data_basic_display, error

def load_data(uploaded_file):
    """
    Returns
    -------
    a dataframe corresponding to the uploaded file or a default dataframe if there are no uploaded file (healthcare dataset stroke data)
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("healthcare-dataset-stroke-data.csv")

def merge_features(selected_feature, merging_cat1, merging_cat2, df):
    """
    Merge 2 features together, with or without a condition: modifying merging_cat1 with condition in merging_cat2
    """
    try:
        for cat in merging_cat1:
            df.loc[df[selected_feature] == cat, selected_feature] = merging_cat2
    except Exception as e:
        error(e)
    return df

def smart_rounding(data):
    """
    Round values according to their best order of magnitude.
    """
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    range_val = max_val - min_val

    if range_val == 0:  # If all data are identical
        #st.error("All the data are identical. Have you thought about deleting this feature ?")
        return np.round(data)
    
    order_of_magnitude = 10 ** int(np.floor(np.log10(range_val)))  

    # Find the best rounding factor
    rounding_factor = order_of_magnitude // 10
    #rounding_factor = math.pow(10, pwr_rounding_factor)
    #st.write(rounding_factor)

    # Appliquer l'arrondi
    rounded_data = np.round(data / rounding_factor) * rounding_factor
    return rounded_data

def summary_features(df):
    summary_features = pd.DataFrame({
        "Feature Type": df.dtypes,
        "Unique Values": df.nunique(),
        "Missing %": df.isna().mean() * 100  # Calcul du pourcentage de valeurs manquantes
    })
    return summary_features

def apply_a_cond(condition, df, disp=False):
    """
    Verifies the truthness of the condition. Display the number of records corresponding.
    Inputs:
    ---------
    condition: the condition to test (str)
    df: the dataset where to apply this condition.
    disp: boolean, True displays more informations about the condition. Default is False
    Returns:
    ---------
    filtered_data:
    """
    try:
        filtered_data = df.query(condition)
        filtered_data_basic_display(filtered_data, disp)
        return filtered_data
    except Exception as e:
        error(f"Error : {e}")
        return None
    