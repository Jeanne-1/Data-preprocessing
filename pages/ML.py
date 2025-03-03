import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

st.set_page_config(
    page_title="Data Analysis Application",
    layout="centered",
    page_icon = "brain"
)

st.title('ML')


st.sidebar.title("Import Dataset")
uploaded_data = st.sidebar.file_uploader("Choose a file")
def load_data(filename):
    data = pd.read_csv(filename)
    return data

if uploaded_data is not None:
    used_file = uploaded_data
else:
    used_file = "clean_disease_prediction_dataset.csv"
if st.sidebar.button('Reload Data'):
    used_file = "disease_prediction_dataset.csv"

df = load_data(used_file)
st.markdown(f"**Used file for this study :** {used_file} (*dim : {df.shape[0]}x{df.shape[1]}*).", unsafe_allow_html=True)
st.write("If you want to modify and clean it, please go to the EDA page.")
col = df.drop(['prognosis'], axis = 1).columns


tab1, tab2, tab3 = st.tabs(["By default", "Choose your own ML", "User Input"])
def model_evaluation(y_test, model):
    acc = accuracy_score(y_test, model)
    prec = precision_score(y_test, model, average='weighted')
    rec = recall_score(y_test, model, average='weighted')
    f1 = f1_score(y_test, model, average='weighted')
    st.markdown(f"<li>Accuracy: {acc}</li> <li>Precision: {prec}</li> <li>Recall: {rec}</li> <li>F1 Score: {f1}</li>", unsafe_allow_html=True)


with tab1:
        
    st.markdown("""
                ## Best Model Selection for Disease Prediction

Given the dataset characteristics, which gives a lot of binary features (symptoms), with the target variable (diagnosis) being categorical and no missing values, here's an analysis of the two best models:

### Random Forest Classifier
**Advantages:**
- **Binary Data Handling:** Highly effective with binary features as present in our case.
- **Feature Importance:** Capable of identifying which symptoms are most indicative of certain diseases, providing additional insights.
- **Complex Relationships:** Can model complex interactions between symptoms and diagnoses without a need for data transformation.
- **Overfitting Robustness:** Generally robust to overfitting, especially with a high number of features.

**Disadvantages:**
- **Interpretability:** Less interpretable than a linear model, though feature importance scores can offer insights.
 """)
    X = df.drop(['prognosis'], axis = 1)  # All columns except the last one are features
    y = df["prognosis"]   # The last column is the target
    
    # Encode the target variable (disease names) to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=4)  # Keeping only 4 components
    X_pca = pca.fit_transform(X_scaled)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=1234)
    
    # Initialize the models
    logistic_model = LogisticRegression(max_iter=1000)
    random_forest_model = RandomForestClassifier(n_estimators=20)
    
    # Train the models
    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    logistic_predictions = logistic_model.predict(X_test)
    random_forest_predictions = random_forest_model.predict(X_test)
    
    st.markdown("**Random Forest :**")
    model_evaluation(y_test, random_forest_predictions)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""  
### Logistic Regression
**Advantages:**
- **Simplicity and Interpretability:** Provides easily interpretable results.
- **Effectiveness:** Can be highly effective for binary or multiclass classification problems with a One-vs-Rest scheme.

**Disadvantages:**
- **Linearity Assumption:** Assumes a linear relationship between independent features and the log odds of the dependent variable, which might not be ideal for capturing complex relationships.
                """)
    st.markdown("**Logistic Regression :** ")
    model_evaluation(y_test, logistic_predictions)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""     
### Model Choice
In this scenario, **Random Forest Classifier appears to be the best choice** for several reasons:
- **Complexity of Relationships:** The dataset includes numerous features that may interact in complex ways to determine a diagnosis. Random Forest can capture these interactions without explicitly specifying interaction terms as needed in regression models.
- **Overfitting Robustness:** With all these features, there's a risk of overfitting, especially if the number of observations isn't exceedingly large. Random Forest manages this risk well through the ensemble of many decision trees. We also did a PCA to reduct these parameters to 4.
- **Binary Features Handling:** Decision trees, and by extension Random Forest, naturally handle binary features, making them particularly suited to our dataset.
""")
        
with tab2:
    st.title("ML Model")

    selected = st.selectbox("Choose a model", ['Linear Regression', "Decision Tree", "Random Forest", "SVM (support vector machine)", "Logistic Regression"])

    if selected == "Linear Regression":
        model = LinearRegression()
    elif selected == "Decision Tree":
        model = DecisionTreeRegressor()
    elif selected == "SVM (support vector machine)":
        model = SVC()
    elif selected == "Logistic Regression":
        n = st.slider("Select number of estimators for the model", min_value=100, max_value=1000, value=500, step=100)
        model = LogisticRegression(max_iter=n)
    else:
        n = st.slider("Select number of estimators for the model", min_value=5, max_value=100, value=20, step=5)
        model = RandomForestClassifier(n_estimators=n)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Trained data, obtained with a PCA")
        st.write(X_train)
    with col2:
        st.markdown("Corresponding encoded prognosis")
        st.write(y_train)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.metric(label="Model Score", value=score)
    
with tab3:
    y_n_tab = []
    st.write("Please check the symptoms you are suffering from :")
    col1,col2 = st.columns(2)
    for idx, symptom in enumerate(col):
        with col1 if idx%2==0 else col2:
            y_n_tab.append(st.checkbox(symptom))
    with st.expander("so ?"):
        user_symptoms = []
        for idx, y_n in enumerate(y_n_tab):
            if(y_n):
                user_symptoms.append(col[idx])
    # Supposons que 'col' est l'objet Index contenant les noms des colonnes (symptômes)
    col_list = col.tolist()  # Convertissez 'col' en une liste Python
    
    with st.expander("Your probable diagnosis :"):
        user_symptoms = []
        for idx, y_n in enumerate(y_n_tab):
            if y_n:
                user_symptoms.append(col_list[idx])  # Utilisez 'col_list' au lieu de 'col'
    
        # Initialisez un vecteur pour les entrées utilisateur basé sur le nombre de symptômes (features)
        user_input = np.zeros(len(col_list))
        
        # Mettez la valeur à 1 pour chaque symptôme coché par l'utilisateur
        for symptom in user_symptoms:
            if symptom in col_list:  # Utilisez 'col_list' ici aussi
                index = col_list.index(symptom)
                user_input[index] = 1
    
        # Appliquez les étapes de prétraitement identiques à celles de vos données d'entraînement
        user_input_scaled = scaler.transform([user_input])
        user_input_pca = pca.transform(user_input_scaled)
    
        # Prédisez la maladie en utilisant le modèle Random Forest
        prediction_encoded = random_forest_model.predict(user_input_pca)
        
        # Décodez la maladie prédite pour récupérer son nom original
        predicted_disease = le.inverse_transform(prediction_encoded)[0]
    
        # Affichez la maladie prédite
        st.write(f"The predicted disease based on the symptoms is: **{predicted_disease}**", unsafe_allow_html= True)
