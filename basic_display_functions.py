# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:52:10 2024

Displaying functions used in my app.py.

@author: jeann
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import streamlit as st

def error(e):
    st.error(e)

def display_shape(df):
    col1, col2 = st.columns(2) #display 2 columns
    with col1:
        st.write("Number of records:", df.shape[0])
    with col2:
        st.write("Number of features:", df.shape[1])

def cor_mat(df_cor):
    st.write("Select the features to include in the correlation matrix:")

    selected_features = item_selection(df_cor.columns, min_non_selected=0, default_value=True, multi_column=True)
    df_cor = df_cor[selected_features] if selected_features else df_cor
    
    for col in df_cor.select_dtypes(exclude=['int64', 'float64']).columns:
        df_cor[col] = df_cor[col].astype('category').cat.codes
        
    mat_cor = df_cor.corr()
    plt.figure()
    sns.heatmap(mat_cor, annot=True, fmt=".2f")
    st.pyplot(plt)

    return df_cor

def repartition_display(show_method, serie, bin_choice = None):
    if show_method == "Categories with more than 1 occurance":
        nb_of_values_per_cat = serie.value_counts()
        if any(nb_of_values_per_cat[nb_of_values_per_cat > 1]):
            st.dataframe(nb_of_values_per_cat[nb_of_values_per_cat > 1].reset_index().rename(columns={'index': serie.name, serie.name: 'Count'}), use_container_width=True)
        else:
            st.write("None")
    else:
        if bin_choice==None:
            st.error("Select a number of bins")
        plt.figure()
        sns.histplot(serie, kde=True, bins=bin_choice)
        plt.title(f"Histogram of {serie.name}")
        st.pyplot(plt)

def display_boxplot(df, chosen_col, error_msg="Too many features. Please explore features one at a time"):
    """
    Displays a boxplot of the chosen_col (NEEDS TO BE NUMERICAL) in the dataframe df
    """
    #CHECK IF NUMERIC DATA
    num_size = len(chosen_col)
    if num_size<9:
        fig, axes = plt.subplots(1, num_size, figsize=(num_size * 3, 5))
        if num_size == 1: #if there is only 1 numerical column
            axes = [axes]
        for ax, column in zip(axes, chosen_col):
            sns.boxplot(data=df, y=column, ax=ax)
            ax.set_title(column)
        plt.tight_layout()
        st.pyplot(fig)
    else: st.write(error_msg)

def display_data(df, chosen_col, chart_type="Histogram"):
    """
    Displays an histogram or a pie chart of categorical values chosen_col of df
    """
    #CHECK IF NUMERIC DATA
    size = len(chosen_col)
    cols_per_row = 4
    rows = math.ceil(size / cols_per_row)

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row*3, rows*2))
    axes = axes.flatten()
    for idx, column in enumerate(chosen_col):
        ax = axes[idx]  # take the right subplot
        if chart_type == "Histogram":
            sns.histplot(data=df, y=column, ax=ax)
        elif chart_type == "Pie chart":
            df[column].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
        ax.set_title(column)
        ax.set_ylabel('')
    if size<4:
        for j in range(idx + 1, len(axes)): #delete unuse col
            fig.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig)

def new_title(title_name, referral=None, is_hr=True):
    """
    Displays a title title_name, with refers to referral (default is title_name).
    is_hr: presence of a separation bar, by default is True
    """
    if referral is None:
        referral = title_name
    st.markdown(f'<a name="{referral}"></a>', unsafe_allow_html=True)
    if is_hr: st.markdown("<hr>", unsafe_allow_html=True)
    st.header(title_name)

def filtered_data_basic_display(df, details=False):
    if details:
        st.write("Filtered Data:")
        st.dataframe(df)
        display_shape(df)
        with st.expander("Basic statistics"):
            st.dataframe(round(df.describe(), 2), use_container_width=True)
    else:
        st.write("Number of Records:", df.shape[0])


def item_selection(items, min_non_selected=1, default_value=False, multi_column=False):
    """
    Displays items you can select.
    
    Parameters:
    - items: List of items to display as checkboxes.
    - min_non_selected: Minimum number of items that must remain unselected. Default is 1.
    - default_value: Default state of checkboxes. Default is False.
    - multi_column: If True, checkboxes are displayed in two columns. Default is False.
    
    Returns:
    - List of selected items.
    """
    checked_items = []
    checked_count = sum(st.session_state.get(c, False) for c in items)
    max_selected = len(items) - min_non_selected  

    # Gestion des colonnes si multi_column est activé
    columns = st.columns(2) if multi_column else [st.container()]  # Liste de colonnes (soit 2, soit 1)

    for idx, c in enumerate(items):
        col = columns[idx % len(columns)]  # Alterne entre col1 et col2 si multi_column=True

        disabled = checked_count >= max_selected and not st.session_state.get(c, False)
        
        with col:
            if st.checkbox(c, value=default_value, key=c, disabled=disabled):
                checked_items.append(c)

    return checked_items

def display_rounded_df(df):
    st.dataframe(round(df,2), use_container_width=True)