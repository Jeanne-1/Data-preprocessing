# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:52:10 2024

Functions used in my app.py that do not need streamlit.

@author: jeann
"""

import pandas as pd
import numpy as np
import math
from pandas.api.types import is_numeric_dtype
from basic_display_functions import filtered_data_display, error, success, plot_decision_tree
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def load_data(uploaded_file):
    """
    Returns:
        A dataframe corresponding to the uploaded file or a default dataframe if there are no uploaded file (healthcare dataset stroke data)
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("healthcare-dataset-stroke-data.csv")

def merge_features(selected_feature, merging_cat1, merging_cat2, df, input_condition=None):
    """
    Merge categories in a feature based on an optional condition.
    
    Parameters:
    - selected_feature (str): The column where merging happens
    - merging_cat1 (list): List of categories to replace
    - merging_cat2 (str): New category name
    - df (pd.DataFrame): The dataframe
    - input_condition (str): A condition to filter rows before merging (e.g., 'age < 14')
    
    Returns:
    - pd.DataFrame: The modified dataframe
    """
    try:
        condition_mask = df[selected_feature].isin(merging_cat1)

        # Apply user-defined condition if provided
        if input_condition:
            condition_mask &= df.query(input_condition).index.to_series().isin(df.index)

        # Apply the merge
        df.loc[condition_mask, selected_feature] = merging_cat2
        
    except Exception as e:
        error(f"Error merging categories: {e}")

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

def feature_normalization(serie):
    # CHOICE: min/max, p1/p99, 3sigma ?
    max_val = np.nanmax(serie)
    min_val = np.nanmin(serie)
    serie = (serie - min_val)/(max_val - min_val)
    return serie

def feature_standardization(serie):
    # CHOICE: z-score, mean_centering ?
    return (serie - serie.mean()) / serie.std()

def summary_features(df):
    summary_features = pd.DataFrame({
        "Feature Type": df.dtypes.astype(str),
        "Unique Values": df.nunique(),
        "Missing %": df.isna().mean() * 100  # Calcul du pourcentage de valeurs manquantes
    })
    return summary_features

def apply_a_cond(condition, df, disp=False, msg="Filtered Data:"):
    """
    Verifies the truthness of the condition. Displays the number of records corresponding.
    
    Parameters:
    - condition (str): the condition to test
    - df (pd.DataFrame): the dataset where to apply this condition.
    - disp (bool): True displays more informations about the condition. Default is False
    
    Returns:
    filtered_data (pd.DataFrame)
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

        elif is_numeric_dtype(df[col]):  # Continuous data
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

def encode(serie):
    le = LabelEncoder()
    encoded_serie = le.fit_transform(serie)
    return encoded_serie, le

def multiple_encode(df, selected_features = None):
    """
    Encode columns of a dataset that need to be encode (type object or category).

    Parameters:
        df (pd.DataFrame): The dataset
        selected_features (list): The columns to encode if needed. When None, encode all the columns. Default is None

    Returns:
        df_encoded (pd.DataFrame): The encoded dataframe
        label_encoders (list): The LabelEncoder of each encoded column
    """
    if selected_features is None:
        df_encoded = df.copy()
    else:
        df_encoded = df[selected_features].copy()
    label_encoders = {}

    for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    return df_encoded, label_encoders

def knn_imputation(df, imputation_feature, selected_features, k=5):
    """
    Impute missing values in a column using K-Nearest Neighbors (KNN).

    Parameters:
        df (pd.DataFrame): Full dataset
        imputation_feature (str): The column to impute
        selected_features (list): Features used for imputation
        k (int): Number of neighbors (default: 5)

    Returns:
        pd.Series: The imputed values
    """
    df_known = df[df[imputation_feature].notna()]  

    if df_known.shape[0] == 0:
        print("No known values available for training the model.")
        return None

    X, label_encoders = multiple_encode(df, selected_features)

    # The column to fill is it categorical ?
    is_categorical = (
        df[imputation_feature].dtype == "object" 
        or df[imputation_feature].dtype.name == "category"
        )

    X[imputation_feature] = df[imputation_feature]

    # Encode if yes
    if is_categorical:
        target_encoder = LabelEncoder()
        X.loc[df[imputation_feature].notna(), imputation_feature] = target_encoder.fit_transform(df[imputation_feature].dropna().astype(str))

    imputer = KNNImputer(n_neighbors=k)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Get only imputed values
    imputed_values = X_imputed.loc[df[imputation_feature].isna(), imputation_feature]

    # Reconvert to string
    if is_categorical:
        imputed_values = target_encoder.inverse_transform(imputed_values.astype(int))

    return imputed_values 


def decision_tree_imputation(df, imputation_feature, selected_features, max_depth=5, is_continuous=None):
    """
    Impute missing values in a column using a Decision Tree. Plot the 4 first nodes.

    Parameters:
        df (pd.DataFrame): The dataset
        imputation_feature (str): The column to impute
        selected_features (list): Features used for imputation
        max_depth (int): Number of nodes (default: 5)
        is_continuous (bool): If the target variable is continuous. Default calculate it based on a threshold of 40.

    Returns:
        pd.Series: The imputed values
    """
    df_known = df[df[imputation_feature].notna()] #lines without NaN

    if df_known.shape[0] == 0:
        error("No known values available for training the model.")
        return None

    if is_continuous==None:
        is_continuous = df[imputation_feature].nunique(dropna=True)>40

    df_encoded, label_encoders = multiple_encode(df, selected_features)

    X_train = df_encoded.loc[df[imputation_feature].notna()]
    y_train = df_known[imputation_feature]
    X_missing = df_encoded.loc[df[imputation_feature].isna()]

    # Encode if the target is categorical
    if df[imputation_feature].dtype == "object" or df[imputation_feature].dtype.name == "category":
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(y_train)
        target_is_categorical = True
    else:
        target_is_categorical = False

    # The model is different if the target is continuous or not
    if is_continuous:
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    else:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    model.fit(X_train, y_train)
    imputed_values = model.predict(X_missing)
    
    # Reconvert to string
    if target_is_categorical:
        imputed_values = target_encoder.inverse_transform(imputed_values.astype(int))

    if max_depth > 4:
        # Modify the max_depth to show only the 4 first nodes
        max_depth = 4 
    plot_decision_tree(model, feature_names=selected_features, max_depth=max_depth)
    
    #ROUND IF CONTINUOUS VALUES
    return imputed_values