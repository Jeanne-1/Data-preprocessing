import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

st.set_page_config(
    page_title="Data Analysis Application",
    layout="centered",
    page_icon = "brain"
)

st.title("EDA")

tab1, tab2, tab3 = st.tabs(["Data Info", "Categorical Feature", "Studies with Prognosis"])


@st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    """
    Returns
    -------
    a dataframe corresponding to the uploaded file or a default dataframe if there are no uploaded file (healthcare dataset stroke data)
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv("disease_prediction_dataset.csv")

st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if st.sidebar.button('Clean Data Automatically'):
    st.session_state['data'] = load_data("clean_disease_prediction_dataset.csv")
    
if 'data' not in st.session_state or st.sidebar.button('Reload Data'):
    st.session_state['data'] = load_data(uploaded_file)

col = st.session_state['data'].columns
register = False #if the df changes, we need to register it in clean_disease_prediction_dataset.csv

with tab1:
    if st.session_state['data'] is not None:
        
        st.markdown("""
        ## Data Reading
        - [Basic Dataset Info](#introduction)
        - [Feature Info](#features)
        - [Filtered Data](#filtered)
        """, unsafe_allow_html=True)
        
        st.markdown('<a name="introduction"></a>', unsafe_allow_html=True)
        st.header("Basic Dataset Information")
        st.write(st.session_state['data'].head(10))
        col1, col2 = st.columns(2)
        with col1:
            st.write("Total Number of Records:", st.session_state['data'].shape[0])
        with col2:
            st.write("Total Number of Features:", st.session_state['data'].shape[1])
        
        st.markdown('<a name="features"></a>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Features Information")
        col3, col4 = st.columns(2)
        with col3:
            st.write("Name and Types of the Features:")
            st.dataframe(st.session_state['data'].dtypes, use_container_width=True)
        features_with_null = st.session_state['data'].columns[st.session_state['data'].isnull().any()].tolist()
        features_with_one = []
        for f in col:
            if st.session_state['data'][f].unique().shape[0]<2:
                features_with_one.append(f)
        with col4:
            st.write("Features with Empty Data:", '' if features_with_null else "None")
            for feature in features_with_null:
                st.write(feature,"with", round(st.session_state['data'].isnull().sum()[feature] / st.session_state['data'].shape[0] * 100, 2), '% empty values.')
                if st.button(f'Drop {feature}'):
                    st.session_state['data'] = st.session_state['data'].drop(columns=[feature])
                    col = st.session_state['data'].columns
                    st.write("Data drop successfully.")
                    register = True
                break
            st.write("Features with Only One Unique Data (which means useless):", ', '.join(features_with_one) if features_with_one else "None")
            for feature in features_with_one:
                u = st.session_state['data'][feature].unique()[0]
                if st.button(f'Drop {feature} (only {u})'):
                    st.session_state['data'] = st.session_state['data'].drop(columns=[feature])
                    col = st.session_state['data'].columns
                    st.write("Data drop successfully.")
                    register = True
                break
        
        if register == True:
            st.session_state['data'].to_csv("clean_disease_prediction_dataset.csv", index=False)
            st.write("Modification on the datas has been registered.")
            register = False
        
        with st.expander("Basic Statistics"):
            st.dataframe(round(st.session_state['data'].describe(), 2), use_container_width=True)
        feature = st.selectbox("Choose a feature", col)
        col5, col6 = st.columns(2)
        with col5:
            st.write('Statistical Info:')
            st.dataframe(st.session_state['data'][feature].describe(), use_container_width=True)
        with col6:
            st.write("Unique Data:", st.session_state['data'][feature].unique().shape[0])
            st.dataframe(st.session_state['data'][feature].value_counts(), use_container_width=True)

        st.markdown('<a name="filtered"></a>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header('Filtered Data')
        user_input = st.text_input("Enter your condition (in the format: feature == a):")
        if st.button('Apply'):
            if user_input: # si l'utilisateur a entré qch
                try:
                    filtered_data = st.session_state['data'].query(user_input)
                    st.write("Filtered Data:")
                    st.write(filtered_data)
                    col7, col8 = st.columns(2)
                    with col7:
                        st.write("Number of Records:", filtered_data.shape[0])
                    with col8:
                        st.write("Number of Features:", filtered_data.shape[1])
                    with st.expander("Basic statistics"):
                        st.dataframe(round(filtered_data.describe(), 2), use_container_width=True)
                except Exception as e:
                    st.error(f"Error : {e}")
            else:
                st.error("Please enter a condition.")

with tab2:
    if st.session_state['data'] is not None:
        st.markdown("""
        ## Data Reading
        - [Correlations](#correlation)
        - [Data visualization](#visualization)
        """, unsafe_allow_html=True)
        data = st.session_state['data'] #car le dataset n'a plus besoin d'être mis à jour
        
        st.markdown('<a name="correlation"></a>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.header("Correlations")
        mat_cor = data.corr()
        st.dataframe(mat_cor)
        same = [] #pour les corrélations = 1
        high_corr = []  #pour les corrélations > 0.8
        moderate_corr = []  #pour les corrélations > 0.5 et <= 0.8
        feature_group_map ={}
        
        # Itérer sur les paires de colonnes pour récupérer les indices et les valeurs de corrélation
        for i in range(len(mat_cor.columns)):
            for j in range(i+1, len(mat_cor.columns)):  # i+1 pour éviter la diagonale et les duplicatas
                if mat_cor.iloc[i, j] == 1:
                    feature_i = mat_cor.columns[i]
                    feature_j = mat_cor.columns[j]

                    # Trouver les groupes existants des deux caractéristiques
                    group_i = feature_group_map.get(feature_i)
                    group_j = feature_group_map.get(feature_j)

                    if group_i and group_j:
                        # Si les deux caractéristiques sont déjà dans des groupes différents, les fusionner
                        if group_i != group_j:
                            group_i.extend(group_j)
                            for feature in group_j:
                                feature_group_map[feature] = group_i
                            same.remove(group_j)
                        # Sinon, elles sont déjà dans le même groupe, rien à faire
                    elif group_i:
                        # Si seulement la caractéristique i est dans un groupe, ajouter j à ce groupe
                        group_i.append(feature_j)
                        feature_group_map[feature_j] = group_i
                    elif group_j:
                        # Si seulement la caractéristique j est dans un groupe, ajouter i à ce groupe
                        group_j.append(feature_i)
                        feature_group_map[feature_i] = group_j
                    else:
                        # Si aucune des caractéristiques n'est dans un groupe, créer un nouveau groupe
                        new_group = [feature_i, feature_j]
                        same.append(new_group)
                        feature_group_map[feature_i] = new_group
                        feature_group_map[feature_j] = new_group
                elif mat_cor.iloc[i, j] > 0.8:
                    high_corr.append((mat_cor.index[i], mat_cor.columns[j], mat_cor.iloc[i, j]))
                elif mat_cor.iloc[i, j] > 0.5:
                    moderate_corr.append((mat_cor.index[i], mat_cor.columns[j], mat_cor.iloc[i, j]))
        
        # Conversion en DataFrame
        df_high_corr = pd.DataFrame(high_corr, columns=['Feature 1', 'Feature 2', 'Correlation'])
        df_moderate_corr = pd.DataFrame(moderate_corr, columns=['Feature 1', 'Feature 2', 'Correlation'])
        
        if same:
            st.write("These features are correlated with a correlation of 1, which means if a patient has one of these symptoms, he has all the others and reciprocally: ")
            for features in same:
                html_s = '<li>'+' ; '.join(features)+'</li>'
                st.markdown(html_s, unsafe_allow_html = True)

            if st.button(f"Join in {len(same)} categories"):
                for cat in same:
                    txt = ', '.join(cat)
                    name = cat[0]
                    del(cat[0])
                    st.session_state['data'] = st.session_state['data'].drop(columns=cat)
                    col = st.session_state['data'].columns
                    st.write(f"All {txt} have been merged in {name}.")
                st.write("Name of the merged columns have been conserved in a variable.") #same
                register = True
                
        with st.expander("Correlation >0.8"):
            st.dataframe(df_high_corr, use_container_width=True)
            st.write(f"There are {df_high_corr.shape[0]} pairs of features with a very high correlation. If a patient have one of these symptoms, he has a very high chance to have the one associated.")
        with st.expander("Correlation between 0.5 and 0.8"):
            st.dataframe(df_moderate_corr, use_container_width=True)
            st.write(f"There are {df_moderate_corr.shape[0]} pairs of features with a high correlation. If a patient have one of these symptoms, he has a non neglectable chance to have the one associated.")
        with st.expander("Correlation between 2 chosen features"):
            features_cor = []
            st.write("Select your features :")
            for i in range(2):
                features_cor.append(st.selectbox(f"Feature {i+1}:", col, index=i))
            cor = data[features_cor].corr()
            st.write(f"Correlation between {features_cor[0]} and {features_cor[1]}:",cor.iloc[0,1])

        st.markdown('<a name="visualization"></a>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.header("Data Visualization")

        visualization_feature = st.selectbox("Choose a feature to visualize", col)

        with st.expander("Histogram"):
            plt.figure(figsize=(10, 6))
            sns.histplot(st.session_state['data'][visualization_feature], bins=2, kde=True)
            plt.title(f"Histogram of {visualization_feature}")
            st.pyplot(plt)

        with st.expander("Pie Chart"):
            if st.session_state['data'][visualization_feature].nunique() < 10:
                pie_data = st.session_state['data'][visualization_feature].value_counts()
                plt.figure()
                plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%',colors=sns.color_palette('Set2'))
                plt.title(f"Pie Chart of {visualization_feature}")
                st.pyplot(plt)
            else:
                st.error("Pie chart is suitable for categorical data or data with fewer unique values.")


        with st.expander("Box Plot of prognosis"):
            data_prognosis = st.session_state['data']['prognosis'].unique()
            for i in range((len(data_prognosis)//7)+1):
                start_idx = i*7 
                end_idx = min(start_idx + 7, len(data_prognosis))
                liste = data_prognosis[start_idx:end_idx]
                datas=st.session_state['data'][st.session_state['data']['prognosis'].isin(liste)]
                plt.figure(figsize=(10, 6))
                sns.violinplot(data=datas, y=visualization_feature, x = 'prognosis')
                plt.title(f"Box Plot of {visualization_feature}")
                st.pyplot(plt)
                #afficher mediane, moyenne pour chaque prognosis ?
        
        if register == True:
            st.session_state['data'].to_csv("clean_disease_prediction_dataset.csv", index=False)
            st.write("Modification on the datas has been registered.")
            register = False

with tab3:
    if st.session_state['data'] is not None:
        st.markdown("""
        ## Studies
        - [Per Prognosis](#prognosis)
        - [Per Symptom](#symptom)
        """, unsafe_allow_html=True)
        
        st.markdown('<a name="prognosis"></a>', unsafe_allow_html=True)
        st.header("Per Prognosis")
        chosen_prognosis = st.selectbox("Choose a prognosis", data_prognosis)
        data_chosen_prog = st.session_state['data'][st.session_state['data']['prognosis']==chosen_prognosis]
        always = []
        clean_chosen_prog = data_chosen_prog.copy(deep = True)
        for feature in data_chosen_prog.columns:
            if data_chosen_prog[feature].max()==0:
                always.append(feature)
                clean_chosen_prog.drop(columns = [feature], inplace = True)
        with st.expander(f'Features Unlikely Causing {chosen_prognosis}'):
            st.write(f"These features are very unlikely causing {chosen_prognosis}: {', '.join(always)}.")
        st.write('Mean :',np.mean(clean_chosen_prog))
        
        st.markdown('<a name="symptom"></a>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.header("Per Symptom")
        chosen_symptom = st.selectbox("Choose a symptom", col.drop('prognosis'))
        data_chosen_symptom = st.session_state['data'][st.session_state['data'][chosen_symptom]==1]
        with st.expander('Pie Chart'):
            plt.figure(figsize=(10, 6))
            plt.pie(data_chosen_symptom['prognosis'].value_counts(), autopct='%1.1f%%',colors=sns.color_palette('Set2'), labels = data_chosen_symptom['prognosis'].value_counts().index)
            plt.title(f"Repartition of {chosen_symptom} between prognosis:")
            st.pyplot(plt)
        with st.expander('Histogram'):
            plt.figure(figsize=(10, 6))
            sns.histplot(data_chosen_symptom['prognosis'])
            plt.title(f"Histogram of possible prognosis in case of {chosen_symptom}")
            plt.axhline(120, color = 'r')
            if data_chosen_symptom['prognosis'].nunique()>5:
                plt.xticks(rotation = 45, ha='right')
            st.pyplot(plt)
            st.write(f"The more a column of a prognosis is close to the red line (total number of people with {chosen_symptom}), the more we can discard it in case of absenteism of {chosen_symptom}.")