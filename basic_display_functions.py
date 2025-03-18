# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:52:10 2024

Displaying functions used in my app.py.

@author: jeann
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.tree import plot_tree
import streamlit as st
import random as rd
from pandas.api.types import is_numeric_dtype

def error(e):
    st.error(e)

def success(s):
    st.success(s)

def new_title(title_name, referral=None, is_hr=True):
    """
    Displays a title, with refers to a referral.

    Parameters
    ----------
    title_name : str
        Title of the section.
    referral : str, optional
        Referral of the title. When None, title_name is taken as referral.
        The default is None.
    is_hr : bool, optional
        Presence of a separation bar. The default is True.

    Returns
    -------
    None.

    """
    if referral is None:
        referral = title_name
    st.markdown(f'<a name="{referral}"></a>', unsafe_allow_html=True)
    if is_hr: st.markdown("<hr>", unsafe_allow_html=True)
    st.header(title_name)

def item_selection(items, min_non_selected=1, default_value=False, multi_column=False, is_popover=False, popover_msg="Filter items"):
    """
    Displays items you can select.
    
    Parameters:
        items (list): List of items to display as checkboxes.
        min_non_selected (int): Minimum number of items that must remain unselected. Default is 1.
        default_value (bool): Default state of checkboxes. Default is False.
        multi_column (bool): If True, checkboxes are displayed in two columns. Default is False.
        is_popover (bool): If True, checkboxes are displayed in a popover. Default is False.
        popover_msg (str): The message to put in the popover, if is_popover is True.
    
    Returns:
        List of selected items.
    """
    is_nb = is_numeric_dtype(items)
    items = [x for x in items if pd.notna(x) and x != ""] #get rid of NaN if there are in items
    
    checked_items = []
    checked_count = sum(st.session_state.get(c, False) for c in items)
    max_selected = len(items) - min_non_selected  

    container = st.popover(popover_msg) if is_popover else st
    columns = container.columns(2) if multi_column else [container.container()] #2 columns if multi_column==True else 1


    for idx, c in enumerate(items):
        col = columns[idx % len(columns)]

        disabled = checked_count >= max_selected and not st.session_state.get(c, False)
        
        with col:
            if is_nb==False:
                key = f"checkbox_{st.session_state.get('tab_key', 'default')}_{idx}_{c}"
            elif is_nb==True:
                c = str(c)
                #the key is reinforce with a random number if c is an int
                key = f"checkbox_{st.session_state.get('tab_key', 'default')}_{idx}_{c}_{rd.randint(0,10**5)}"
            
            if st.checkbox(c, value=default_value, key=key, disabled=disabled):
                checked_items.append(c)

    return checked_items

def display_shape(df, percent = False):
    """
    Display the shape of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to analyze.
    percent : bool, optional
        If True, compare the size of the given dataframe to the size of the dataframe
        in session_state. The default is False.

    Returns
    -------
    None.

    """
    col1, col2 = st.columns(2) #display 2 columns
    with col1:
        nb_records = df.shape[0]
        if percent and "data" in st.session_state:
            percent_records = round(nb_records / st.session_state.data.shape[0] * 100, 2)
        st.write(f"Number of records: {nb_records} {f'({percent_records}%)' if percent else ''}")
    with col2:
        st.write("Number of features:", df.shape[1])


#%% Figures
def cor_mat(df):
    """
    Displays a correlation matrix

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe in which we want to see the correlations.

    Returns
    -------
    mat_cor : pd.DatFframe
        The correlation matrix.

    """
    st.write("Select the features to include in the correlation matrix:")

    selected_features = item_selection(df.columns, min_non_selected=0, default_value=True, multi_column=True, is_popover=True)
    df = df[selected_features] if selected_features else df
    
    for col in df.select_dtypes(exclude=['int64', 'float64']).columns:
        df[col] = df[col].astype('category').cat.codes
        
    mat_cor = df.corr()
    plt.figure()
    sns.heatmap(mat_cor, annot=True, fmt=".2f")
    st.pyplot(plt)

    return mat_cor

def repartition_display(show_method, serie, bin_choice = None):
    """
    Displays the repartition of occurance, either with a dataframe of categories
    with more than 1 occurance or with a histogram.

    Parameters
    ----------
    show_method : int (0 or 1)
        Choice of the method to use to display
        0: "Categories with more than 1 occurance" with a dataframe
        1: "Repartition of occurance" with an histogram
    serie : pd.Serie
        The serie to display.
    bin_choice : int, optional
        Number of bins to display the histogram. The default is None.

    Returns
    -------
    None.

    """
    try:
        if show_method == 0:
            nb_of_values_per_cat = serie.value_counts()
            if any(nb_of_values_per_cat[nb_of_values_per_cat > 1]):
                st.dataframe(nb_of_values_per_cat[nb_of_values_per_cat > 1].reset_index().rename(columns={'index': serie.name, serie.name: 'Count'}), use_container_width=True)
            else:
                st.write("Each category occurs one time.")
        elif show_method == 1:
            if bin_choice==None:
                st.error("Select a number of bins")
            plt.figure()
            sns.histplot(serie, kde=True, bins=bin_choice)
            plt.title(f"Histogram of {serie.name}")
            st.pyplot(plt)
        else:
            error("This displaying method has not been created yet.")
    except Exception as e:
        error(e)

def display_cat_repartition(df, col, hue_on = False, threshold_cat=40):
    """
    Displays the repartition of categories using an histplot

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe we want to display the repartition of a column.
    col : str
        The column to display.
    hue_on : bool, optional
        If True, we display the histplot with hue=y. It will automatically be
        transformed to False if y is not in session_state or if y is continuous.
        The default is False.
    threshold_cat : int, optional
        The threshold separating categorical variable from non categorical. 
        The default is 40.

    Returns
    -------
    None.

    """
    is_cat = df[col].nunique()<threshold_cat #categorical variable
    kde = not is_cat #display kde only if non categorical
    
    if "y" in st.session_state:
        y = st.session_state.y
        if df[y].nunique()>threshold_cat: 
            hue_on = False #always false if y is not categorical
            st.write(f"You cannot displays according to {y} because it's a continuous feature.")
    else:
        hue_on = False
        st.write("You didn't initialize y. Please do it in the sidebar.")
    
    nb_plots = 1 if is_cat else 2
    fig, axes = plt.subplots(1, nb_plots, figsize=(6, 3))
    if nb_plots==1: axes = [axes]
    if hue_on:
        sns.histplot(data=df, x=col, kde=kde, hue=y, stat="percent", common_norm=False, multiple="dodge", ax=axes[0])
    else:
        sns.histplot(data=df, x=col, kde=kde, ax=axes[0])
    axes[0].set_title(f"Histogram of {col}")
    
    if nb_plots==2:
        if hue_on:
            sns.boxplot(data=df, y=col, x=y, flierprops={"marker": "x"},ax=axes[1])
        else:
            sns.boxplot(data=df, y=col, ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")
    plt.tight_layout()
    st.pyplot(fig)

def display_boxplots(df, chosen_col, error_msg="Too many features. Please explore features one at a time"):
    """
    Displays a boxplot of each column in chosen_col in the dataframe df

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe.
    chosen_col : list (str)
        List of columns to display.
    error_msg : str, optional
        Message to write if there are more than 8 boxplots to display. 
        The default is "Too many features. Please explore features one at a time".

    Returns
    -------
    None.

    """
    num_size = len(chosen_col)
    if num_size<9:
        fig, axes = plt.subplots(1, num_size, figsize=(num_size * 3, 5))
        if num_size == 1: # if there is only 1 numerical column
            axes = [axes]
        for ax, column in zip(axes, chosen_col):
            sns.boxplot(data=df, y=column, ax=ax)
            ax.set_title(column)
        plt.tight_layout()
        st.pyplot(fig)
    else: st.write(error_msg)

def display_charts(df, chosen_col, chart_type="Histogram"):
    """
    Displays histograms or a pie charts of categorical values chosen_col of df

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which you display columns.
    chosen_col : list (str)
        The columns to display.
    chart_type : str, optional
        Can be either "Histogram" and "Pie Chart", corresponding to the type of
        chart to display. The default is "Histogram".

    Returns
    -------
    None.

    """
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
    for j in range(idx + 1, len(axes)): # delete unuse charts
        fig.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig)

def filtered_data_display(df, details=False, stat=True, msg="Filtered Data:"):
    """
    Display data, originally for filtered data. Either displays just the size, 
    the dataframe or the statistics.

    Parameters
    ----------
    df : pd.DataFrame 
        The dataframe to display.
    details : bool, optional
        If True, displays the dataframe. The default is False.
    stat : bool, optional
        If True and details is too, displays the statistics in an expander.
        The default is True.
    msg : str, optional
        Message to write at the beggining if details is True. 
        The default is "Filtered Data:".

    Returns
    -------
    None.

    """
    if len(df)!=0 and details:
        st.write(msg)
        st.dataframe(df)
        display_shape(df, percent=True)
        if stat:
            with st.expander("Basic statistics"):
                display_rounded_df(df.describe())
    else:
        st.write("Number of Records:", df.shape[0])

def display_rounded_df(df):
    st.dataframe(round(df,2), use_container_width=True)

def plot_decision_tree(model, feature_names, max_depth=4):
    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True, fontsize=8, max_depth=max_depth)
    with st.expander("See the decision tree"):
        st.pyplot(plt)
