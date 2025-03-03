# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:52:10 2024

@author: jeann
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import basic_display_functions as bdf
from other_functions import *

def merge_and_update(selected_feature, merging_cat1, merging_cat2):
    st.session_state["data"] = merge_features(
        selected_feature, merging_cat1, merging_cat2, st.session_state["data"]
    )

def update_selected_feature():
    st.session_state["data"][selected_feature] = df  # Met à jour les données

#%%Page config
st.set_page_config(page_title="Data Preprocessing App", page_icon=":clean:")

st.title("Data preprocessing")
st.write("This application will help you analyze, clean and reduce a dataset of your choice. Remember you can navigate between the different pages at any time, and process each preprocessing step at the desired time. You can go to page Learn to learn how to do it by yourself. ")
st.write("Keep in mind this app is designed for global cleaning and dataset can need extra preprocessing steps.")
st.text("These pages have been created by Jeanne Auger from ESILV.")

tab1, tab2, tab3 = st.tabs(["Data Visualization", "Data cleaning", "Data transformation and reduction"])

st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if 'data' not in st.session_state or st.sidebar.button('Reload Data'):
    st.session_state['data'] = load_data(uploaded_file)

register = False #if the df changes, we need to register it in clean_dataset.csv
st.sidebar.selectbox("y : ",st.session_state['data'].columns, key="y")
binary_sol = {"Yes": True, "No": False}

with tab1:
    df = st.session_state['data'] #as we don't modify the dataset in this step, we can stock it in df
    if df is not None:
        st.markdown("""
        ## Exploratory Data Analysis
        - [Dataset Preview](#introduction)
        - [Global Features Info](#features)
        - [Global Statistics](#stat)
        - [Feature Exploration](#feat)
        - [Filtered Data](#filtered)
        """, unsafe_allow_html=True)

        bdf.new_title("Dataset Preview", "introduction", False)
        st.write(df.head(10)) #display the beggining of the dataset
        bdf.display_shape(df)
        
        bdf.new_title("Global Features Information", "features")
        st.dataframe(summary_features(df), use_container_width=True)
        
        bdf.new_title("Basic Statistics", "stat")
        st.dataframe(round(df.describe(), 2), use_container_width=True)

        st.subheader("Boxplot of numerical features")

        cat_col = [col for col in df.columns if df[col].nunique() < 10] #columns of df corresponding in categorical values
        num_col = df.select_dtypes(include=['number']).columns #columns in df corresponding in numerical values
        num_col = [col for col in num_col if col not in cat_col] #we keep only non categorical columns
        
        bdf.display_boxplot(df, num_col)

        st.subheader("Categorical features")
        chart_type = st.radio("Prefered display type : ", ["Histogram", "Pie chart"])
        
        bdf.display_data(df, cat_col, chart_type)
        
        bdf.new_title("Explore a feature more in detail", "feat")
        feature = st.selectbox("Choose a feature", df.columns)
        col1, col2 = st.columns(2) #display on 2 columns
        with col1:
            st.write('Statistical Info:')
            st.dataframe(df[feature].describe(), use_container_width=True)
        with col2:
            st.write("Unique Data:", df[feature].unique().shape[0])
            st.dataframe(df[feature].value_counts(), use_container_width=True)

        bdf.new_title('Filtered Data', "filtered")
        user_input = st.text_input("Enter your condition:", placeholder="feature == a")
        if st.button('Apply', key=1):
            if user_input: apply_a_cond(user_input, df=st.session_state['data'], disp=True)
            else: st.error("Please enter a condition")

with tab2:
    if st.session_state['data'] is not None:
        st.write(f"This column has been chosen for y: {st.session_state.y}. You can change it in the sidebar.")
        st.header("12")

with tab3:
    if st.session_state['data'] is not None:
        st.write(f"This column has been chosen for y: {st.session_state.y}. You can change it in the sidebar.")
        
        st.dataframe(summary_features(df), use_container_width=True)

        bdf.cor_mat(st.session_state['data'])
        
        col = st.session_state['data'].columns
        #col = [c for c in col if col != y] # A REGLER
        st.selectbox("Feature you want to modify", col, key="selected_feature")
        selected_feature = st.session_state.selected_feature

        col1, col2 = st.columns(2) #display 2 columns
        with col1:
            st.header("Type correction")
            #new_type = st.selectbox(f"New type for {selected_feature} : ", {"Numerical", "String"})
            #if new_type == "Numerical":
                #transform the data to numerical data : 
                ##check if they are already numerical or not
                ##if not, function to numeric them

        with col2:
            st.header("Numerosity reduction")
            #curseur de quand c'est considéré comme categorical value
            if st.session_state['data'][selected_feature].nunique() < 40:
                st.markdown(f"Categorical feature with **{st.session_state['data'][selected_feature].unique().shape[0]}** unique data.")
                st.dataframe(st.session_state['data'][selected_feature].value_counts(), use_container_width=True)
                choice = st.radio("What would you like to do ?", {"Erase lines", "Merge categories"}, horizontal=True)
                
                #possibilité de remplacer une valeur par NaN

                if choice == "Erase lines":
                    erase_cat = st.selectbox("Which category ?", st.session_state['data'][selected_feature].unique())
                    if st.button("Confirm the erasing"):
                        st.session_state['data'] = st.session_state['data'][st.session_state['data'][selected_feature] != erase_cat]
                        st.write(f"Lines successfully deleted. New shape : {st.session_state['data'].shape}")
                        st.experimental_rerun() #To update the display
                
                elif choice == "Merge categories":
                    n_unique = st.session_state['data'][selected_feature].unique()
                    merging_cat1 = []
                    st.write("Which category will be replaced ?")
                    checked_count = sum(st.session_state.get(c, False) for c in n_unique)
                    for c in n_unique:
                        disabled = checked_count >= len(n_unique) - 1 and not st.session_state.get(c, False)
                        if st.checkbox(c, key=c, disabled=disabled): #enable merging multiple categories in 1
                            merging_cat1.append(c)
                    merging_cat2 = st.selectbox("With which category ?", [cat for cat in n_unique if cat not in merging_cat1])
                    #is_condition = st.radio("Would you like to add a condition on the category for the merging ?", binary_sol, horizontal=True)
                    is_condition = "No"
                    if binary_sol[is_condition]:
                        condition = st.text_input("Type the condition", placeholder="cat == a")
                        if condition: apply_a_cond(f"{condition} and {selected_feature}=='{merging_cat1}'", df=st.session_state['data'])
                        else: st.error("You didn't enter a condition")
                    else:
                        filtered_data = st.session_state['data']
                    
                    if merging_cat1:
                        st.button('Merge', on_click=merge_and_update, args=(selected_feature, merging_cat1, merging_cat2))
                    
            else:
                st.markdown(f"Continuous feature with **{st.session_state['data'][selected_feature].nunique()}** unique data.")
                show_method = st.radio("What do you want to show ?", {"Categories with more than 1 occurance", "Repartition of occurance"})
                #Optimal nb of bins
                bin_edges = np.histogram_bin_edges(st.session_state['data'][selected_feature].dropna(), bins="auto")  # Supprime les NaN si besoin
                optimal_bins = len(bin_edges) - 1
                bin_choice = 2

                if show_method == "Repartition of occurance":
                    bin_choice = st.slider("Number of bins:", min_value=2, max_value=round(optimal_bins*1.5), value=optimal_bins)
                bdf.repartition_display(show_method, st.session_state['data'][selected_feature], optimal_bins, bin_choice)

                set_reduce_methods = {"Containing slices of same number of records":1, "Containing slice evenly separates":2, "Round the data":3}
                reduce_method = st.radio("How do you want to reduce your data ?", set_reduce_methods)
                nb_slice = st.slider("Number of slices:", min_value=2, max_value=round(optimal_bins*1.5), value=optimal_bins)
                #PREVIEW of stats (boxplot ?) and hist ?
                df = st.session_state['data'][selected_feature]
                df = df.sort_values() #to obtain the values from the littlest to the bigger
                min_val = np.nanmin(df)
                nb_values = len(df)
                max_val = np.nanmax(df)

                if set_reduce_methods[reduce_method]==1:
                    nb_per_slice = nb_values//nb_slice
                    for i in range(nb_slice):
                        new_min = df[i]
                        new_max = df[i+nb_per_slice]
                        avg_value = df[i:i+nb_per_slice]/nb_slice

                elif set_reduce_methods[reduce_method]==3:
                    st.dataframe(df.describe(), use_container_width=True)
                    is_float = df.dtype == "float64"
                    rounding_factor = st.slider("Rounding factor (in power of 10)", min_value=-2 if is_float else 1, max_value=int(np.log10(max_val)))
                    df = np.round(st.session_state['data'][selected_feature] / 10**rounding_factor) * 10**rounding_factor

                    bdf.repartition_display(show_method, st.session_state['data'][selected_feature], optimal_bins, bin_choice)

                    st.button("It's perfect like that", on_click=update_selected_feature)


                #take average value between all of them / median value
            #erase useless categorical choices (appearing less time than the other), or merge them with other categories
            #for continuous: identical number of person in each category or identical range
        st.header("Dimensionality reduction")
