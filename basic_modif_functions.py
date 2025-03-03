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
    if condition: # if the user entered sth
        try:
            filtered_data = df.query(condition)
            if disp:
                st.write("Filtered Data:")
                st.dataframe(filtered_data)
                col1,col2 = st.columns(2)
                with col1:
                    st.write("Number of Records:", filtered_data.shape[0])
                with col2:
                    st.write(f"Number of Features:", filtered_data.shape[1])
                with st.expander("Basic statistics"):
                    st.dataframe(round(filtered_data.describe(), 2), use_container_width=True)
            else:
                st.write("Number of Records:", filtered_data.shape[0])
            st.session_state.filtered_data = filtered_data
        except Exception as e:
            st.error(f"Error : {e}")
    else:
        st.error("Please enter a condition.") #DONT DISPLAY THAT EVERY TIME YOU LAUNCH

def merge_features(selected_feature, merging_cat1, merging_cat2, df):
    """
    Merge 2 features together, with or without a condition: modifying merging_cat1 with condition in merging_cat2
    """
    df.loc[df[selected_feature] == merging_cat1, selected_feature] = merging_cat2

    # Mettre à jour le dataset dans session_state
    st.session_state["data"] = df


def smart_rounding(data):
    """
    Round values according to their best order of magnitude.
    """
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    range_val = max_val - min_val

    if range_val == 0:  # If all data are identical
        st.error("All the data are identical. Have you thought about deleting this feature ?")
        return np.round(data)
    
    order_of_magnitude = 10 ** int(np.floor(np.log10(range_val)))  

    # Find the best rounding factor
    rounding_factor = order_of_magnitude // 10
    #rounding_factor = math.pow(10, pwr_rounding_factor)
    st.write(rounding_factor)

    # Appliquer l'arrondi
    rounded_data = np.round(data / rounding_factor) * rounding_factor
    return rounded_data

def update_selected_feature():
    st.session_state["data"][selected_feature] = df  # Met à jour les données

#%%Page config
st.set_page_config(page_title="Data Preprocessing App", page_icon=":clean:")

st.title("Data preprocessing")
st.write("This application will help you analyze, clean and reduce a dataset of your choice. Remember you can navigate between the different pages at any time, and process each preprocessing step at the desired time. You can go to page Learn to learn how to do it by yourself. ")
st.write("Keep in mind this app is designed for global cleaning and dataset can need extra preprocessing steps.")
st.text("These pages have been created by Jeanne Auger from ESILV.")

tab1, tab2, tab3 = st.tabs(["Data Visualization", "Data cleaning", "Data transformation and reduction"])

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

st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if 'data' not in st.session_state or st.sidebar.button('Reload Data'):
    st.session_state['data'] = load_data(uploaded_file)

register = False #if the df changes, we need to register it in clean_dataset.csv
y = st.sidebar.selectbox("y : ",st.session_state['data'].columns)
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

        st.markdown('<a name="introduction"></a>', unsafe_allow_html=True)
        st.header("Dataset Preview")
        st.write(df.head(10)) #display the beggining of the dataset
        col1, col2 = st.columns(2) #display 2 columns
        with col1:
            st.write("Total number of records:", df.shape[0])
        with col2:
            st.write("Total number of features:", df.shape[1])
        
        st.markdown('<a name="features"></a>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Global Features Information")
        summary_features = pd.DataFrame({
            "Feature Type": df.dtypes,
            "Unique Values": df.nunique(),
            "Missing %": df.isna().mean() * 100  # Calcul du pourcentage de valeurs manquantes
        })
        st.dataframe(summary_features, use_container_width=True)
        
        st.markdown('<a name="stat"></a>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Basic Statistics")
        st.dataframe(round(df.describe(), 2), use_container_width=True)

        st.subheader("Violonplot of numerical features")
        cat_col = [col for col in df.columns if df[col].nunique() < 10]
        num_col = df.select_dtypes(include=['number']).columns
        num_col = [col for col in num_col if col not in cat_col] #we keep only non categorical columns
        num_size = len(num_col)
        if num_size<9:
            fig, axes = plt.subplots(1, num_size, figsize=(num_size * 3, 5))
            if num_size == 1: #if there is only 1 numerical column
                axes = [axes]
            for ax, column in zip(axes, num_col):
                sns.boxplot(data=df, y=column, ax=ax)
                ax.set_title(column)
            plt.tight_layout()
            st.pyplot(fig)
        else: st.write("Too many numerical features. Please explore features one at a time")

        st.subheader("Categorical features")
        chart_type = st.radio("Prefered display type : ", ["Histogram", "Pie chart"])
        cat_size = len(cat_col)
        cols_per_row = 4
        rows = math.ceil(cat_size / cols_per_row)
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row*3, rows*2))
        axes = axes.flatten()
        for idx, column in enumerate(cat_col):
            ax = axes[idx]  # take the right subplot
            if chart_type == "Histogram":
                sns.histplot(data=df, y=column, ax=ax)
            else:
                df[column].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
            ax.set_title(column)
            ax.set_ylabel('')
        if cat_size<4:
            for j in range(idx + 1, len(axes)): #delete unuse col
                fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown('<a name="feat"></a>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Explore a feature more in detail")
        feature = st.selectbox("Choose a feature", df.columns)
        col1, col2 = st.columns(2) #display on 2 columns
        with col1:
            st.write('Statistical Info:')
            st.dataframe(df[feature].describe(), use_container_width=True)
        with col2:
            st.write("Unique Data:", df[feature].unique().shape[0])
            st.dataframe(df[feature].value_counts(), use_container_width=True)

        st.markdown('<a name="filtered"></a>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header('Filtered Data')
        user_input = st.text_input("Enter your condition:", placeholder="feature == a")
        st.button('Apply', on_click=lambda: apply_a_cond(user_input, df=st.session_state['data'], disp=True), key=1)
            

with tab2:
    if st.session_state['data'] is not None:
        st.write(f"This column has been chosen for y: {y}. You can change it in the sidebar.")
        st.header("12")
with tab3:
    if st.session_state['data'] is not None:
        st.write(f"This column has been chosen for y: {y}. You can change it in the sidebar.")
        
        st.dataframe(summary_features, use_container_width=True)

        df_cor = st.session_state['data']
        selected_features = []
        st.write("Select the features to include in the correlation matrix:")

        col1,col2 = st.columns(2)
        for idx, c in enumerate(df_cor.columns):
            with col2 if idx%2 else col1:
                if st.checkbox(c, value = True, key=c):
                    selected_features.append(c)

        df_cor = df_cor[selected_features] if selected_features else df_cor
        for col in df_cor.select_dtypes(exclude=['int64', 'float64']).columns:
            df_cor[col] = df_cor[col].astype('category').cat.codes
        
        mat_cor = df_cor.corr()
        plt.figure()
        sns.heatmap(mat_cor, annot=True, fmt=".2f")
        st.pyplot(plt)

        col = st.session_state['data'].columns
        #col = [c for c in col if col != y] # A REGLER
        selected_feature = st.selectbox("Feature you want to modify", col)

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
                
                if choice == "Erase lines":
                    erase_cat = st.selectbox("Which category ?", st.session_state['data'][selected_feature].unique())
                    if st.button("Confirm the erasing"):
                        st.session_state['data'] = st.session_state['data'][st.session_state['data'][selected_feature] != erase_cat]
                        st.write(f"Lines successfully deleted. New shape : {st.session_state['data'].shape}") # DEAL WITH THE MAJ OF DATA
                
                elif choice == "Merge categories":
                    n_unique = st.session_state['data'][selected_feature].unique()
                    merging_cat1 = st.selectbox("Which category will be merged ?", n_unique) #check of each category instead to enable merging multiple categories in 1
                    merging_cat2 = st.selectbox("With which category ?", [cat for cat in n_unique if cat != merging_cat1])
                    #is_condition = st.radio("Would you like to add a condition on the category for the merging ?", binary_sol, horizontal=True)
                    is_condition = "No"
                    if binary_sol[is_condition]:
                        st.text_input("Type the condition", placeholder="cat == a", key = "condition")
                        apply_a_cond(f"{st.session_state.condition} and {selected_feature}=='{merging_cat1}'", df=st.session_state['data'])
                    else:
                        filtered_data = st.session_state['data']
                    
                    st.button('Merge', on_click=lambda:merge_features(selected_feature, merging_cat1, merging_cat2, st.session_state["data"]))
            else:
                st.markdown(f"Continuous feature with **{st.session_state['data'][selected_feature].nunique()}** unique data.")
                show_method = st.radio("What do you want to show ?", {"Categories with more than 1 occurance", "Repartition of occurance"})
                #Optimal nb of bins
                bin_edges = np.histogram_bin_edges(df[selected_feature].dropna(), bins="auto")  # Supprime les NaN si besoin
                optimal_bins = len(bin_edges) - 1
                if show_method == "Categories with more than 1 occurance":
                    nb_of_values_per_cat = st.session_state['data'][selected_feature].value_counts()
                    if any(nb_of_values_per_cat[nb_of_values_per_cat > 1]):
                        st.dataframe(nb_of_values_per_cat[nb_of_values_per_cat > 1].reset_index().rename(columns={'index': selected_feature, selected_feature: 'Count'}), use_container_width=True)
                    else:
                        st.write("None")
                else:
                    bin_choice = st.slider("Number of bins:", min_value=2, max_value=round(optimal_bins*1.5), value=optimal_bins)
                    #ADD STH TO COMPARE (HUE PARAMETER)
                    plt.figure()
                    sns.histplot(st.session_state['data'][selected_feature], kde=True, bins=bin_choice)
                    plt.title(f"Histogram of {selected_feature}")
                    st.pyplot(plt)
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
                    if show_method == "Categories with more than 1 occurance":
                        nb_of_values_per_cat = df.value_counts()
                        if any(nb_of_values_per_cat[nb_of_values_per_cat > 1]):
                            st.dataframe(nb_of_values_per_cat[nb_of_values_per_cat > 1].reset_index().rename(columns={'index': selected_feature, selected_feature: 'Count'}), use_container_width=True)
                        else:
                            st.write("None")
                    else:
                        #ADD STH TO COMPARE (HUE PARAMETER)
                        plt.figure()
                        sns.histplot(df, kde=True, bins=bin_choice)
                        plt.title(f"Histogram of {selected_feature}")
                        st.pyplot(plt)
                    st.button("It's perfect like that", on_click=update_selected_feature)


                #take average value between all of them / median value
            #erase useless categorical choices (appearing less time than the other), or merge them with other categories
            #for continuous: identical number of person in each category or identical range
        st.header("Dimensionality reduction")
