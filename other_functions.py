# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:52:10 2024

Functions used in my app.py that do not need streamlit.

@author: jeann
"""

import pandas as pd
import numpy as np
import math
from basic_display_functions import filtered_data_display, error, success

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
        "Feature Type": df.dtypes.astype(str),
        "Unique Values": df.nunique(),
        "Missing %": df.isna().mean() * 100  # Calcul du pourcentage de valeurs manquantes
    })
    return summary_features

def apply_a_cond(condition, df, disp=False, msg="Filtered Data:"):
    """
    Verifies the truthness of the condition. Display the number of records corresponding.
    Parameters:
    ---------
    condition: the condition to test (str)
    df: the dataset where to apply this condition.
    disp: boolean, True displays more informations about the condition. Default is False
    Returns:
    ---------
    filtered_data:
    """
    #DEAL IF YOU WANT TO WORK WITH THE SAME COL IN DATA CLEANING & TRANSFO
    try:
        filtered_data = df.query(condition)
        if len(filtered_data)!=0:
            filtered_data_display(filtered_data, disp, msg)
        return filtered_data
    except Exception as e:
        error(f"Error : {e}")
        return None

def equal_width_reduction(df, nb_slices):
    return pd.cut(df, bins=nb_slices, labels=False, include_lowest=True)

def equal_freq_reduction(df, nb_slices):
    try:
        # Compute quantile-based bins (adaptive to the distribution)
        quantiles = np.linspace(0, 1, nb_slices + 1)  # Create nb_slices bins
        bins = df.quantile(quantiles).values  # Get actual data values

        # Assign each value to a bin
        return pd.cut(df, bins=bins, labels=False, include_lowest=True)
    except Exception as e:
        error(e)

def nan_replace(df, col, cat, threshold_cat=40):
    try:
        if df[col].nunique() < threshold_cat:
            changed_col = df[col].replace(cat, np.NaN)
        else: 
            changed_col = df[col].mask(~df[col].between(*cat))
        return changed_col
    except Exception as e:
        error(e)
        return df

def erase_records(df, col, cond, threshold_cat=40):
    try:
        if df[col].nunique() < threshold_cat:
            if isinstance(cond, list):  # cond is a list of categories to delete
                df = df[~df[col].isin(cond)]
            else:  # cond is a unique value to delete
                df = df[df[col] != cond]

        elif pd.api.types.is_numeric_dtype(df[col]):  # Continuous data
            if isinstance(cond, tuple) and len(cond) == 2:
                df = df[(df[col] >= cond[0]) & (df[col] <= cond[1])]  # We only keep values in between cond[0] and cond[1]
            else:
                error("Incorrect condition for continuous data. Please use a tuple (min, max).")

        else:
            error("Unrecognize type of condition.")

        return df
    
    except Exception as e:
        error(e)
        return df

def delete_feature(df, feature):
    return df.drop(feature, axis=1)