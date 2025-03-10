# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:52:10 2024

@author: jeann
"""

import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import basic_display_functions as bdf
from other_functions import *

if "threshold_cat" not in st.session_state:
    st.session_state.threshold_cat = 40


def merge_and_update(selected_feature, merging_cat1, merging_cat2):
    st.session_state["data"] = merge_features(
        selected_feature, merging_cat1, merging_cat2, st.session_state["data"]
    )

def update_selected_feature(df, selected_feature = None, desactivate=False):
    if selected_feature is not None: 
        st.session_state["data"][selected_feature] = df  # Update data
        if desactivate: 
            st.session_state.reducing_desactivation[selected_feature] = True
    else: st.session_state["data"] = df

def update_fill_na(df_col, feature):
    new_value = st.session_state.imputation  # Récupère la valeur actuelle
    imputed_df = df_col.fillna(f"{new_value}")  # Remplit les NaN avec la nouvelle valeur
    st.session_state.data[feature] = imputed_df

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

if "reducing_desactivation" not in st.session_state:
    st.session_state.reducing_desactivation = {} #dictionnary used to deactivate the reducing possibilities when it already has been reduced
if "edited_types" not in st.session_state:
    st.session_state["edited_types"] = st.session_state["data"].dtypes.astype(str).to_dict()

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
        bdf.display_rounded_df(df.describe())

        st.subheader("Boxplot of numerical features")

        cat_col = [col for col in df.columns if df[col].nunique() < 10] #columns of df corresponding in categorical values
        num_col = df.select_dtypes(include=['number']).columns #columns in df corresponding in numerical values
        num_col = [col for col in num_col if col not in cat_col] #we keep only non categorical columns
        
        bdf.display_boxplots(df, num_col)

        st.subheader("Categorical features")
        chart_type = st.radio("Prefered display type : ", ["Histogram", "Pie chart"])
        
        bdf.display_charts(df, cat_col, chart_type)
        
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
        threshold_cat = st.session_state.threshold_cat
        st.write(f"This column has been chosen for y: {st.session_state.y}. You can change it in the sidebar.")
        col = st.session_state['data'].columns
        #erase doublons (choice of cat to consider in doublons)
        #deal with noise and outliers
        #put outliers apart and get rid of noise
        #what to do with noise ? erase record, erase the value, make an hypothesis and replace the value, use DT to replace the value
        #erase records with more than n NaN values ?

        #consider as noise any data above or under first and last percentiles ?
        bdf.new_title("Handle anormal values", is_hr=False)
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            selected_feature = st.selectbox("Which feature do you want to work with ?", col)
        with col2:
            hue_on = st.toggle(f"Comparing to {st.session_state.y}")
        bdf.display_cat_repartition(st.session_state.data,selected_feature, hue_on)

        df_selected_feature = st.session_state.data[selected_feature]
        nb_cat_selected_feature = df_selected_feature.nunique()
        few_categories = False #has the feature less than 3 cat ?

        if nb_cat_selected_feature<3:
            col1, col2 = st.columns([3,2], vertical_alignment="center")
            with col1: st.write(f"There are already only {nb_cat_selected_feature} categories.")
            with col2:
                if st.button("Delete the whole feature"):
                    df = delete_feature(st.session_state.data, selected_feature)
                    update_selected_feature(df)
                    st.rerun()
            few_categories = True

        #erase features with anormal values: when is it an anormal value ? do a check
        #add a condition: anormal value given a certain cat (outlier detection)
        elif nb_cat_selected_feature<threshold_cat:
            st.write("Select anormal values :")
            selected_items = bdf.item_selection(df_selected_feature.unique(), multi_column=True)

        elif is_numeric_dtype(df_selected_feature):
            min_val = df_selected_feature.min()
            max_val = df_selected_feature.max()
            selected_items = st.slider(f"{selected_feature} is anormal when its value is not in between:", min_value = min_val, max_value = max_val, value=(min_val, max_val)) #get rid of the NaN
            apply_a_cond(f"{selected_feature}<{selected_items[0]} or {selected_feature}>{selected_items[1]}", st.session_state.data, disp=True, msg="Data concerned by the change:")
        
        col1, col2 = st.columns([1,3])
        with col1:
            if not few_categories and st.button("Erase the record(s)"):
                #erase lines values outside [selected_items[0], selected_items[1]] or selected_items
                #UPDATE THE DATA OBTAINED
                df = erase_records(st.session_state.data, selected_feature, selected_items, threshold_cat)
                update_selected_feature(df)
                st.rerun() #To update the display
        with col2:
            if not few_categories and st.button("Replace with NaN"):
                #replace values outside [selected_items[0], selected_items[1]] or selected_items with NaN
                df = nan_replace(st.session_state.data, selected_feature, selected_items)
                update_selected_feature(df, selected_feature)
                st.rerun()

        
        #WHAT DO YOU WANT TO DO WITH MISSING VALUES ? REMOVAL/IMPUTATION /IGNORE (if use of ML algorithm that deals with MV)
        bdf.new_title("Handle missing values")
        df = st.session_state.data

        #DISPLAY FEATURES WITH NAN VALUES AND CHOSE WETHER OR NOT DELETE THEM
        nan_features = df.isnull().sum()>0
        nb_nan_features = nan_features.sum()
        if nb_nan_features>0:
            col1, col2 = st.columns([2,1], vertical_alignment="bottom")
            with col1:
                df_nan = pd.DataFrame({
                    "Missing %": df.isna().mean()[nan_features] * 100,
                    "Remove ?": False  # Initial values is false
                })
                edited_df_nan = st.data_editor(
                    df_nan,
                    disabled=["", "Missing %"],
                    use_container_width=True
                )
            with col2:
                if st.button("Delete checked features"):
                    #For each checked feature in the tab, we remove the column from the df
                    cols_to_remove = edited_df_nan[edited_df_nan["Remove ?"]].index.tolist()
                    st.session_state.data.drop(columns=cols_to_remove, inplace=True)
                    st.success(f"Features {cols_to_remove} successfully removed!")
                    st.rerun()
        
            #CHANGE THE MAX TO BE THE MAX A RECORD CAN HAVE OF NAN VALUES
            if nb_nan_features>1:
                max_nan_values = st.slider(
                    "Records with these NaN values or more:", 
                    min_value=1, max_value=nb_nan_features, 
                    value=1
                    )
            else: max_nan_values=1
            bdf.filtered_data_display(
                df[df.isna().sum(axis=1)>=max_nan_values], 
                details=True, 
                msg=f"Records with {max_nan_values} null value(s) or more:"
                )
            
            col1, col2 = st.columns([1,2])
            with col1:
                st.subheader("Removal")
                #check which records are concerned
                if st.button("Erase these records"):
                    #erase values with more than max_nan_values NaN
                    df = df.dropna(thresh=df.shape[1] - max_nan_values) 

            with col2:
                st.subheader("Imputation")
                imputation_feature = (
                    df.columns[nan_features][0] if nb_nan_features==1
                    else st.selectbox("Which feature do you want to impute null values ?", df.loc[:,nan_features].columns)
                    )
                is_continuous = df[imputation_feature].nunique(dropna=True)>threshold_cat
                imputation_method_choice = {f"by a constant":0, f"with machine learning":1}
                imputation_method = st.selectbox(f"How do you want to replace null values in {imputation_feature} ?", imputation_method_choice)
                if imputation_method_choice[imputation_method]==0:
                    if is_numeric_dtype(df[imputation_feature]):
                        imputation = df[imputation_feature].median() # Median
                        name = "median"
                        ex_value = -1.0
                    else:
                        imputation = df[imputation_feature].mode().iloc[0]  # Mode
                        name = "mode"
                        ex_value = "Other"
                    colA, colB = st.columns(2, vertical_alignment="center")
                    with colA:
                        st.write("Impute with...")
                        if st.button(f"the {name}: {imputation}"):
                            imputed_df = df[imputation_feature].fillna(imputation)
                            update_selected_feature(imputed_df, imputation_feature)
                            st.rerun()
                        if is_continuous:
                            imputation_mean = df[imputation_feature].mean()
                            if st.button(f"the mean: {round(imputation_mean,2)}"):
                                # TO ROUND CORRECTLY
                                imputed_df = df[imputation_feature].fillna(imputation_mean)
                                update_selected_feature(imputed_df, imputation_feature)
                                st.rerun()
                    with colB:
                        if is_numeric_dtype(df[imputation_feature]):
                            imputation_nb = st.number_input(
                                "A new value:", 
                                value=ex_value
                                )
                            st.button(
                                "Impute", 
                                on_click = update_selected_feature, 
                                args=(df[imputation_feature].fillna(imputation_nb), imputation_feature)
                                )
                        else:
                            st.text_input(
                                "A new value:", 
                                placeholder=ex_value, 
                                key="imputation", 
                                on_change=update_fill_na, 
                                args=(df[imputation_feature], imputation_feature)
                            )

        else:
            st.write("There are no missing value in the dataframe.")
        

with tab3:
    if st.session_state['data'] is not None:
        threshold_cat = st.session_state.threshold_cat
        st.write(f"This column has been chosen for y: {st.session_state.y}. You can change it in the sidebar.")
        
        bdf.cor_mat(st.session_state['data'])
        
        col = st.session_state['data'].columns
        #col = [c for c in col if col != y] # A REGLER
        
         # Définition des types possibles
        possible_types = ["int64", "float64", "object", "category", "bool"]

        # Utilisation de st.data_editor avec édition restreinte
        edited_summary = st.data_editor(
            summary_features(st.session_state.data),
            column_config={
                "Feature Type": st.column_config.SelectboxColumn(
                    "Feature Type", options=possible_types
                )
            },
            disabled=["", "Unique Values", "Missing %"], #You can only modify feature_type
            use_container_width=True,
            key="edited_rows"
        )
        st.button("Apply_change", on_click=apply_type_change) ##NOT WORKING
        
        st.selectbox("Feature you want to modify", col, key="selected_feature")

        selected_feature = st.session_state.selected_feature

        col1, col2 = st.columns(2) #display 2 columns
        with col1:
            st.header("Normalization and standardization")
            

        with col2:
            st.header("Numerosity reduction")
            #curseur de quand c'est considéré comme categorical value
            if st.session_state['data'][selected_feature].nunique() < threshold_cat: #categorical feature
                st.markdown(f"Categorical feature with **{st.session_state['data'][selected_feature].unique().shape[0]}** unique data.")
                st.dataframe(st.session_state['data'][selected_feature].value_counts(), use_container_width=True)
                choice = st.radio("What would you like to do ?", {"Erase lines", "Merge categories"}, horizontal=True)

                ## CHANGE THE WAY OF DOING IT: st.data_editor WITH STH TO DEL AND STH TO MODIFY THE CAT
                ## THEN BUTTON TO VALIDATE THE MODIFICATIONS

                if choice == "Erase lines":
                    erase_cat = st.selectbox("Which category ?", st.session_state['data'][selected_feature].unique())
                    if st.button("Confirm the erasing"):
                        st.session_state['data'] = st.session_state['data'][st.session_state['data'][selected_feature] != erase_cat]
                        st.write(f"Lines successfully deleted. New shape : {st.session_state['data'].shape}")
                        st.rerun() #To update the display
                
                elif choice == "Merge categories":
                    n_unique = st.session_state['data'][selected_feature].unique()
                    st.write("Which category will be replaced ?")
                    merging_cat1 = bdf.item_selection(n_unique)
                    merging_cat2 = st.selectbox("With which category ?", [cat for cat in n_unique if cat not in merging_cat1])
                    is_condition = st.toggle("Add a condition")
                    if is_condition: #MAKE IT WORK
                        condition = st.text_input("Type the condition", placeholder="cat == a")
                        if condition: apply_a_cond(f"{condition} and {selected_feature}=='{merging_cat1}'", df=st.session_state['data'])
                        #else: st.error("You didn't enter a condition.")
                    else:
                        filtered_data = st.session_state['data']
                    
                    if merging_cat1: #merge button is available only if some features are selected by the user to prevent errors
                        st.button('Merge', on_click=merge_and_update, args=(selected_feature, merging_cat1, merging_cat2))

            else: #continuous features
                st.markdown(f"Continuous feature with **{st.session_state['data'][selected_feature].nunique()}** unique data.")
                show_method = st.radio("What do you want to show ?", {"Categories with more than 1 occurance", "Repartition of occurance"})
                #Optimal nb of bins
                bin_edges = np.histogram_bin_edges(st.session_state['data'][selected_feature].dropna(), bins="auto")  # Supprime les NaN si besoin
                optimal_bins = len(bin_edges) - 1
                bin_choice = 2

                if show_method == "Repartition of occurance":
                    bin_choice = st.slider("Number of bins:", min_value=2, max_value=round(optimal_bins*1.5), value=optimal_bins)
                bdf.repartition_display(show_method, st.session_state['data'][selected_feature], bin_choice)

                set_reduce_methods = {"Equal-width intervals":1, "Equal-frequency intervals":2} #, "Round the data":3}
                
                if selected_feature not in st.session_state.reducing_desactivation:
                    st.session_state.reducing_desactivation[selected_feature] = False

                if not st.session_state.reducing_desactivation[selected_feature]:
                    reduce_method = st.radio("How do you want to reduce your data ?", set_reduce_methods)
                    #PREVIEW of stats (boxplot ?) and hist ?

                    df = st.session_state['data'][selected_feature]
                    default_value = st.session_state['data'][selected_feature].nunique() - 1 #nb of unique value for the feature -1 to prevent any error
                    max_value = round(optimal_bins*1.5)
                    if max_value > default_value:
                        max_value = default_value
                    elif optimal_bins < default_value:
                        default_value = optimal_bins

                    nb_slices = st.slider("Number of slices:", min_value=2, max_value=max_value, value=default_value)
                    
                    if set_reduce_methods[reduce_method]==1:
                        binned_series = equal_width_reduction(df, nb_slices)

                    elif set_reduce_methods[reduce_method]==2:
                        binned_series = equal_freq_reduction(df, nb_slices)

                    # Summarize each bin
                    bin_summary = df.groupby(binned_series).agg(["min", "max", "mean", "median"])

                    st.write("Bin Ranges & Summary Statistics:")
                    bdf.display_rounded_df(bin_summary)

                    aggregate_type = st.selectbox("Which value would you like to keep foreach bin ?", {"Min", "Max", "Mean", "Median"})
                    bin_values = df.groupby(binned_series).agg(aggregate_type.lower())
                    df = binned_series.map(bin_values)

                    #elif set_reduce_methods[reduce_method]==3: ECQ ON GARDE CETTE METHOD ?
                    #    is_float = df.dtype == "float64"
                    #   rounding_factor = st.slider("Rounding factor (in power of 10)", min_value=-2 if is_float else 1, max_value=int(np.log10(max_val)))
                    #  df = np.round(st.session_state['data'][selected_feature] / 10**rounding_factor) * 10**rounding_factor

                    dif_stat = pd.concat(
                        [st.session_state['data'][selected_feature].describe(), df.describe()],
                        axis=1,
                        keys=["Before", "After"]
                    )

                    bdf.display_rounded_df(dif_stat)

                    bdf.repartition_display(show_method, df, bin_choice)

                    st.button("It's perfect like that", on_click=update_selected_feature, args=(df, selected_feature, True))
                else: st.write("Discretization already done.")

        bdf.new_title("Dimensionality reduction")
        type_dim_red = st.selectbox("Type", {"Feature selection", "Feature construction"})

        if type_dim_red == "Feature selection":
            #whole_search = st.toggle("Exhaustive search")
            #filter or wrapper ? Or only mRMR and QRA and EBR

            st.write("We will only deal with filter methods. You can try wrapper or embedded methods if you're doing a ML algorithm.")
            fs_method = st.selectbox("Which method do you want to adopt ?", {"mRMR", "QRA", "EBR"})
            #when do you stop ? when you don't lost any information VS when you have a certain nb of features
            #if fs_method == "mRMR":
                
            #filter methods: mRMR, QRA, EBR
            #wrapper: GA
        #FEATURE SELECTION: get rid of features that do not seam to be correlate with your y
        #FEATURE CONSTRUCTION: do a PCA
        #do a DT and keep only features that appear inside the DT
        #FEATURE SELECTION