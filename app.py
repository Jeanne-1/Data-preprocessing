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
import other_functions as of
from sklearn.decomposition import PCA
from os import getcwd

threshold_cat = 40

if 'tab_key' not in st.session_state: #constant to create unique keys for checkboxes
    st.session_state['tab_key'] = 'tab0'
    
if 'new_df' not in st.session_state:
    st.session_state.new_df = None

def merge_and_update(selected_feature, merging_cat1, merging_cat2, condition):
    st.session_state["data"] = of.merge_features(
        selected_feature, merging_cat1, merging_cat2, st.session_state["data"], condition
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
    st.session_state['data'] = of.load_data(uploaded_file)

register = False #if the df changes, we need to register it in clean_dataset.csv
st.sidebar.selectbox("y : ",st.session_state['data'].columns, key="y")

if "reducing_desactivation" not in st.session_state:
    st.session_state.reducing_desactivation = {} #dictionnary used to deactivate the reducing possibilities when it already has been reduced
if "edited_types" not in st.session_state:
    st.session_state["edited_types"] = st.session_state["data"].dtypes.astype(str).to_dict()

#%% Data visualization
with tab1:
    st.session_state['tab_key'] = 'tab1'
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
        st.dataframe(of.summary_features(df), use_container_width=True)
        
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
            if user_input: of.apply_a_cond(user_input, df=st.session_state['data'], disp=True)
            else: st.error("Please enter a condition")

#%% Data Cleaning
with tab2:
    if st.session_state['data'] is not None:
        st.markdown("""
        ## Data Cleaning
        - [Anormal Values](#av)
        - [Missing Values](#mv)
        - [Redundancy](#red)
        """, unsafe_allow_html=True)

        st.write(f"This column has been chosen for y: {st.session_state.y}. You can change it in the sidebar.")
        col = st.session_state['data'].columns
       
        bdf.new_title("Handle anormal values", "av", is_hr=False)
        st.session_state['tab_key'] = 'tab2_av'

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
                    df = of.delete_feature(st.session_state.data, selected_feature)
                    update_selected_feature(df)
                    st.rerun()
            few_categories = True

        # ADD A CONDITION
        elif nb_cat_selected_feature<threshold_cat:
            st.write("Select anormal values :")
            selected_items = bdf.item_selection(df_selected_feature.unique(), multi_column=True)

        elif is_numeric_dtype(df_selected_feature):
            min_val = df_selected_feature.min()
            max_val = df_selected_feature.max()
            selected_items = st.slider(f"{selected_feature} is anormal when its value is not in between:", min_value = min_val, max_value = max_val, value=(min_val, max_val)) #get rid of the NaN
            of.apply_a_cond(f"{selected_feature}<{selected_items[0]} or {selected_feature}>{selected_items[1]}", st.session_state.data, disp=True, msg="Data concerned by the change:")
        
        col1, col2 = st.columns([1,3])
        with col1:
            if not few_categories and st.button("Erase the record(s)"):
                #erase lines values outside [selected_items[0], selected_items[1]] or selected_items
                df = of.erase_records(st.session_state.data, selected_feature, selected_items, threshold_cat)
                update_selected_feature(df)
                st.rerun() #To update the display
        with col2:
            if not few_categories and st.button("Replace with NaN"):
                #replace values outside [selected_items[0], selected_items[1]] or selected_items with NaN
                df = of.nan_replace(st.session_state.data, selected_feature, selected_items)
                update_selected_feature(df, selected_feature)
                st.rerun()
        
        bdf.new_title("Handle missing values", "mv")
        st.session_state['tab_key'] = 'tab2_mv'
        df = st.session_state.data.copy()

        nan_features = df.isnull().sum()>0
        nb_nan_features = nan_features.sum()
        if nb_nan_features>0:
            st.subheader("Removal")
            col1, col2 = st.columns([2,1], vertical_alignment="bottom")
            # Delete features
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
        
            # Delete records
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
            
            if st.button("Erase these records"):
                # erase values with more than max_nan_values NaN
                threshold = df.shape[1] - max_nan_values + 1
                df.dropna(thresh=threshold, inplace=True)
                st.session_state["data"] = df
                st.rerun()
            
            st.subheader("Imputation...")
            imputation_feature = (
                    df.columns[nan_features][0] if nb_nan_features==1
                    else st.selectbox("Which feature do you want to impute null values ?", df.loc[:,nan_features].columns)
                    )
            col1, col2 = st.columns([1,2])

            # Imputation with a constant
            with col1:
                st.subheader("... by a constant")
                st.write("")
                is_continuous = df[imputation_feature].nunique(dropna=True)>threshold_cat
                
                # The constant is a given value (median, mean or mode)
                if is_numeric_dtype(df[imputation_feature]):
                    imputation = df[imputation_feature].median()
                    name = "median"
                    ex_value = -1.0
                else:
                    imputation = df[imputation_feature].mode().iloc[0] 
                    name = "mode"
                    ex_value = "Other"

                st.write(f"Impute {imputation_feature} with...")

                if st.button(f"the {name}: {imputation}", use_container_width=True):
                    imputed_df = df[imputation_feature].fillna(imputation)
                    update_selected_feature(imputed_df, imputation_feature)
                    st.rerun()

                if is_continuous:
                    imputation_mean = df[imputation_feature].mean()
                    if st.button(f"the mean: {round(imputation_mean,2)}", use_container_width=True):
                        # TO ROUND CORRECTLY
                        imputed_df = df[imputation_feature].fillna(imputation_mean)
                        update_selected_feature(imputed_df, imputation_feature)
                        st.rerun()
                
                st.write("")

                # The constant is a value given by the user
                if is_numeric_dtype(df[imputation_feature]):
                    imputation_nb = st.number_input(
                        "A new value:", 
                        value=ex_value
                        )
                    st.button(
                        "Impute this value", 
                        on_click = update_selected_feature, 
                        args=(df[imputation_feature].fillna(imputation_nb), imputation_feature),
                        type="tertiary", 
                        use_container_width=True
                        )
                else:
                    st.text_input(
                        "A new value:", 
                        placeholder=ex_value, 
                        key="imputation", 
                        on_change=update_fill_na, 
                        args=(df[imputation_feature], imputation_feature)
                    )

                # GROUP-BY IMPUTATION TO ADD
            
            # Imputation with ML (DT or KNN)
            with col2:
                st.subheader("...with Machine Learning")
                ml_choice = st.selectbox("ML choice", {"Decision Tree", "KNN"}, label_visibility="collapsed")

                non_na_features = df.columns[df.notna().all()]
                selected_features = bdf.item_selection(non_na_features, default_value=True, is_popover=	True, popover_msg=f"Features used for imputation of {imputation_feature}")

                name = "Max depth :" if ml_choice == "Decision Tree" else "Number of neighbors :"
                arg = st.slider(name, min_value = 2, max_value = 10, value=4)
                df_copy = st.session_state.data.copy() # To avoid directly modifying the dataset
                
                if st.button("Visualize"):
                    if ml_choice == "Decision Tree":
                        imputed_values = of.decision_tree_imputation(df_copy, imputation_feature, selected_features, max_depth=arg, is_continuous=is_continuous)
                    else:
                        imputed_values = of.knn_imputation(st.session_state.data, imputation_feature, selected_features, k=arg)
                    
                    colA, colB = st.columns([4,1], vertical_alignment="bottom")
                    
                    # Visualization
                    with colA:
                        imputed_serie = pd.Series(imputed_values)
                        
                        if is_continuous:
                            dif_stat = pd.concat(
                                [st.session_state['data'][imputation_feature].describe(), imputed_serie.describe()],
                                axis=1,
                                keys=["Original values", "Imputed values"]
                            )
    
                            bdf.display_rounded_df(dif_stat)
                        else:
                            st.write(imputed_serie.value_counts())
    
                    with colB:
                        if st.button("Impute", type="tertiary"):
                            st.session_state.data.loc[st.session_state.data[imputation_feature].isna(), imputation_feature] = imputed_values
                            st.rerun()
        else:
            st.write("There are no missing value in the dataframe.")
        
        # deal with redundancy
        bdf.new_title("Redundancy", "red")
        st.session_state['tab_key'] = 'tab2_red'

        selected_items = df.columns
        col1,col2= st.columns(2)
        with col1:
            selected_items = bdf.item_selection(df.columns, default_value=True, is_popover=True, popover_msg="Which features should be considered ?")
        
        df = st.session_state.data
        # Nb of duplicate data (If 1 raw appears twice, duplicate_nb=1)
        duplicate_nb = df.duplicated(subset=selected_items).sum()
        
        with col2:
            if duplicate_nb==0:
                st.write("There are no duplicates according to this subset.")
            else:
                st.write(f"{duplicate_nb} ({round(100*duplicate_nb/df.shape[0],2)}%) record(s) concerned.")
        
        if duplicate_nb>0:
            with st.expander("See the concerned record(s)"):
                bdf.filtered_data_display(df.loc[df.duplicated(subset=selected_items)], details=True, stat=False)
            if st.button("Erase them"):
                # erase values with more than max_nan_values NaN
                threshold = df.shape[1] - max_nan_values + 1
                df.drop_duplicates(subset=selected_items, inplace=True)
                st.session_state["data"] = df
                st.rerun()

#%% Data Transformation and reduction
with tab3:
    st.session_state['tab_key'] = 'tab3'
    if st.session_state['data'] is not None:
        st.markdown("""
        ## Data Transformation and Reduction
        - [Type change](#tc)
        - [Normalization and Standardization](#ns)
        - [Numerosity Reduction](#num_red)
        - [Dimensionality Reduction](#dim_red)
        """, unsafe_allow_html=True)
        st.write(f"This column has been chosen for y: {st.session_state.y}. You can change it in the sidebar.")
        
        col = st.session_state['data'].columns
        
        bdf.new_title("Type change", "tc", is_hr=False)
        
         # Possible types
        possible_types = ["int64", "float64", "object", "category", "bool"]
        df = st.session_state.data.copy()
        
        st.data_editor(
            of.summary_features(st.session_state.data),
            column_config={
                "Feature Type": st.column_config.SelectboxColumn(
                    "Feature Type", options=possible_types
                )
            },
            disabled=["", "Unique Values", "Missing %"], #You can only modify feature_type
            use_container_width=True,
            key="edited_type",
        )
        
        if "edited_type" in st.session_state:
            edited_rows = st.session_state.edited_type.get("edited_rows")
            
            for row, feature_type in edited_rows.items():
                new_type = feature_type.get("Feature Type")
                col_name = col[row]
                old_type = str(df[col_name].dtype)
                if new_type != old_type:
                    with st.expander(f"Apply type change on {col_name}"):
                        nb_values = df[col_name].nunique()
                        try:
                            if new_type == "bool" and nb_values<3:
                                true_value = st.radio("True is:", df[col_name].unique(), horizontal=True)
                                false_values = [c for c in df[col_name].unique() if c != true_value]
                                df.loc[df[col_name] == true_value, col_name] = True
                                df.loc[df[col_name].isin(false_values), col_name] = False
                            elif old_type in ["object", "category"] and new_type not in ["object", "category"]:
                                new_col, le = of.encode(df[col_name])
                                df[col_name] = new_col
                                
                            if st.button(f"Confirm bool conversion for {col_name}"):
                                df[col_name] = df[col_name].astype(new_type)
                                update_selected_feature(df[col_name], col_name)
                                st.success(f"✅ {col_name} successfully converted to {new_type}")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Impossible to convert {col_name} into {new_type}: {e}")
        else:
            st.write("You can change your types by changing them in the tab.")
        
        # Data Transformation: Normalization and Standardization
        st.markdown("<a name=ns></a><hr>", unsafe_allow_html=True)
        selected_feature = st.selectbox("Feature you want to modify", col, key="selected_feature")

        is_continuous = df[selected_feature].nunique(dropna=True)>threshold_cat
        few_categories = df[selected_feature].nunique()<3
        if is_continuous: 
            col1, col2 = st.columns(2)
            with col1:
                st.write(st.session_state.data[selected_feature].describe())
            with col2:
                bdf.new_title("Transformation", is_hr=False)
                st.write("")
                st.button("Normalize", on_click= update_selected_feature, args=(of.feature_normalization(df[selected_feature]), selected_feature))
                st.write("")
                st.button("Standardize", on_click= update_selected_feature, args=(of.feature_standardization(df[selected_feature]), selected_feature))

        st.session_state['tab_key'] = 'num_red'
        bdf.new_title("Numerosity reduction", "num_red")
        
        col1, col2 = st.columns(2)
        
        # Numerosity reduction for a categorical feature
        if st.session_state['data'][selected_feature].nunique() < threshold_cat:
            with col1:
                st.markdown(f"Categorical feature with **{st.session_state['data'][selected_feature].unique().shape[0]}** unique data.")
            
            # Numerosity reduction not allowed if there are 1 or 2 features (useless)
            if few_categories:
                with col1:
                    st.dataframe(st.session_state['data'][selected_feature].value_counts(), use_container_width=True)
                with col2:
                    st.markdown("There are not enough values to reduce them. Would you like to erase the feature ?")
                    if st.button("Delete the whole feature"):
                        df = of.delete_feature(st.session_state.data, selected_feature)
                        update_selected_feature(df)
                        st.rerun()
            else:
                with col1:
                    st.dataframe(st.session_state['data'][selected_feature].value_counts(), use_container_width=True)
                    choice = st.radio("What would you like to do ?", {"Erase lines", "Merge categories"}, horizontal=True)
    
                with col2:
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
                        
                        if merging_cat1: #the rest is available only if some features are selected by the user to prevent errors
                            merging_choice = [cat for cat in n_unique if cat not in merging_cat1] + ["Create new..."] # add the choice to create a category
                            merging_cat2 = st.selectbox("Into which category ?", merging_choice)
                            if merging_cat2 == "Create new...":
                                merging_cat2 = st.text_input("Name of the category", placeholder="Other")
                                
                            is_condition = st.toggle("Add a condition")
                            if is_condition:
                                condition = st.text_input("Type the condition", placeholder="cat == a")
                                if st.button('Apply', key=st.session_state.tab_key):
                                    if condition: 
                                        of.apply_a_cond(f"{condition} and {selected_feature} in {merging_cat1}", df=st.session_state['data'])
                                    else: 
                                        st.error("You didn't enter a condition")
                            else:
                                condition = None
                                
                            st.button('Merge', on_click=merge_and_update, args=(selected_feature, merging_cat1, merging_cat2, condition))
    
        # Numerosity reduction for continuous features
        else:
            with col1:
                st.markdown(f"Continuous feature with **{st.session_state['data'][selected_feature].nunique()}** unique data.")
                # Optimal nb of bins calculation
                bin_edges = np.histogram_bin_edges(st.session_state['data'][selected_feature].dropna(), bins="auto")  # Supprime les NaN si besoin
                optimal_bins = len(bin_edges) - 1
                bin_choice = 2
                show_method = st.radio("What do you want to show ?", {"Categories with more than 1 occurance", "Repartition of occurance"})
            
            with col2:
                if show_method == "Repartition of occurance":
                    bin_choice = st.slider("Number of bins:", min_value=2, max_value=round(optimal_bins*1.5), value=optimal_bins)
                st.write("Before the reduction:")
                bdf.repartition_display(show_method, st.session_state['data'][selected_feature], bin_choice)
            
            
            if selected_feature not in st.session_state.reducing_desactivation:
                st.session_state.reducing_desactivation[selected_feature] = False
            
            # Continue only if it has not been already done
            if not st.session_state.reducing_desactivation[selected_feature]:
                with col1:
                    set_reduce_methods = {"Equal-width intervals":1, "Equal-frequency intervals":2}
                        
                    reduce_method = st.radio("How do you want to reduce your data ?", set_reduce_methods)
        
                    df = st.session_state['data'][selected_feature]
                    default_value = st.session_state['data'][selected_feature].nunique() - 1 #nb of unique value for the feature -1 to prevent any error
                    max_value = round(optimal_bins*1.5)
                    if max_value > default_value:
                        max_value = default_value
                    elif optimal_bins < default_value:
                        default_value = optimal_bins
    
                    nb_slices = st.slider("Number of slices:", min_value=2, max_value=max_value, value=default_value)
                    
                    if set_reduce_methods[reduce_method]==1:
                        binned_series = of.equal_width_reduction(df, nb_slices)
        
                    elif set_reduce_methods[reduce_method]==2:
                        binned_series = of.equal_freq_reduction(df, nb_slices)
        
                    # Summarize each bin
                    with st.expander("Bin Ranges & Summary Statistics"):
                        bin_summary = df.groupby(binned_series).agg(["min", "max", "mean", "median"])
                        bdf.display_rounded_df(bin_summary)
        
                    # To keep understandability
                    aggregate_type = st.selectbox("Which value would you like to keep foreach bin ?", {"Min", "Max", "Mean", "Median"})
                    bin_values = df.groupby(binned_series).agg(aggregate_type.lower())
                    df = binned_series.map(bin_values)
                    
                    dif_stat = pd.concat(
                        [st.session_state['data'][selected_feature].describe(), df.describe()],
                        axis=1,
                        keys=["Before", "After"]
                    )

                    bdf.display_rounded_df(dif_stat)
                    st.button("It's perfect like that", on_click=update_selected_feature, args=(df, selected_feature, True))

                with col2:
                    st.write("After the reduction:")
                    bdf.repartition_display(show_method, df, bin_choice)
        
            else: st.write("Discretization already done.")
        
        st.session_state['tab_key'] = 'dim_red'
        bdf.new_title("Dimensionality reduction", "dim_red")
        
        bdf.cor_mat(st.session_state['data'])
                
        df = st.session_state.data.copy()
        type_dim_red = st.selectbox("Type", {"Feature selection", "Feature construction"})
        
        if type_dim_red == "Feature selection":

            st.write("We will only deal with filter methods using mRMR. You can try wrapper or embedded methods if you're doing a ML algorithm.")
            
            name = "select"
            
        else:
            st.write("It's better to construct your features from scaled variables.")
            name="have"
        y = st.session_state.y
        cols = [c for c in df.columns if c != y]
            
        nb_features_selected = st.slider(f"How many features do you want to {name} ?", min_value = 1, max_value = len(cols), value = 5)
        selected_cols = bdf.item_selection(cols, min_non_selected=nb_features_selected, default_value = True, is_popover=True, popover_msg="Tested features")
        
        if len(selected_cols)==nb_features_selected:
            st.write("You don't need to use a method, you already have the wanted amount of features.")
            st.dataframe(selected_features)
            
        else:
            if type_dim_red == "Feature selection": 
                selected_cols.append(y)
                if st.button("mRMR"):
                    selected_features = of.mrmr_feature_selection(df[selected_cols], y)
                    st.dataframe(selected_features)
                    st.session_state.new_df = df[selected_features]
            #when do you stop ? when you don't lost any information VS when you have a certain nb of features
                
            #PCA
            elif type_dim_red == "Feature construction":
                df.dropna(inplace = True)
                
                X = df[selected_cols]
                y = df[y]
                
                y_encoded, le = of.encode(y)
                X_encoded, Xle = of.multiple_encode(X)
                
                if st.button("PCA"):
                    pca = PCA(n_components=nb_features_selected)  # Keeping only 4 components
                    new_df = pca.fit_transform(X_encoded)
                    st.session_state.new_df = pd.DataFrame(new_df)
                    st.dataframe(new_df)
            
            # MAKE IT WORK WITH OTHER DOWNLOADING FORMAT
            # PUT BACK ID
            # DOWNLOAD IN THE SIDE BAR AT ANY TIME YOUR FULL DATASET
            if st.session_state.new_df is not None:
                access_path = st.text_input("Access path", value=getcwd())
                col1,col2 =st.columns(2, vertical_alignment="bottom")
                with col1:
                    file_name = st.text_input("Name of your file", value="preprocessed_data")
                with col2:
                    file_path = f"{access_path}\{file_name}.csv"
                    if st.button("Download the obtained data"):
                        try:
                            st.session_state.new_df.to_csv(file_path, index=False)
                            bdf.success("The file has been downloading.")
                        except Exception as e:  
                            bdf.error(f"Error in downloading the file: {e}")