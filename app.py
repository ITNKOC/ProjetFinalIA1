import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score

# 1. Téléchargement de Fichier
def upload_file():
    st.header("Upload your CSV file")
    sep_option = st.selectbox("Choose separator", [",", ";", "\t", " ", "|", "\\n", "Custom"])
    if sep_option == "Custom":
        sep_option = st.text_input("Enter custom separator")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=sep_option)
            st.write(df)
            return df
        except pd.errors.ParserError as e:
            st.error(f"Error reading the file: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please upload a CSV file.")
    return None

# 2. Statistiques Sommaires
def display_summary_statistics(df):
    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

# 3. Visualisation des Données
def visualize_data(df):
    st.subheader("Data Visualizations")
    
    # Distribution of numeric features
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        st.write(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    # Bar plots for categorical features
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"Bar plot for {col}")
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        st.pyplot(fig)

    # Heatmap of correlations
    if not numeric_columns.empty:
        st.write("Heatmap of correlations")
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_columns].corr(), annot=True, ax=ax)
        st.pyplot(fig)

# 4. Gestion des Valeurs Manquantes
def handle_missing_values(df):
    st.subheader("Handling Missing Values")
    
    missing_values = df.isnull().sum()
    st.write("Missing values in each column:")
    st.write(missing_values)

    if missing_values.any():
        method = st.selectbox("Choose a method to handle missing values", 
                              ["Drop rows with missing values", 
                               "Fill with mean", 
                               "Fill with median", 
                               "Fill with mode", 
                               "Fill with specific value"])
        
        if method == "Drop rows with missing values":
            df = df.dropna()
        elif method == "Fill with mean":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                df[col] = df[col].fillna(df[col].mean())
        elif method == "Fill with median":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                df[col] = df[col].fillna(df[col].median())
        elif method == "Fill with mode":
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif method == "Fill with specific value":
            fill_value = st.text_input("Enter the value to fill missing values with")
            if fill_value:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna(fill_value)
                    else:
                        df[col] = df[col].fillna(float(fill_value))
        
        st.write("Data after handling missing values:")
        st.write(df)
    else:
        st.success("No missing values found in the dataset.")
    
    return df


# 5. Prétraitement des Données
def preprocess_data(df, target_column):
    st.subheader("Data Preprocessing")
    
    # Séparer les caractéristiques et la cible
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Afficher les premières lignes
    st.write("Initial data sample:")
    st.write(X.head())
    
    # Encodage des variables catégorielles (chaînes)
    categorical_columns = X.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_columns = encoder.fit_transform(X[categorical_columns])
        X = X.drop(columns=categorical_columns)
        encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))
        X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
    # Convertir toutes les colonnes en numériques
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Remplacer les NaN restants
    X = X.fillna(0) 
    
    # Afficher les données après prétraitement
    st.write("Data sample after preprocessing:")
    st.write(X.head())
    st.write("Data types after preprocessing:")
    st.write(X.dtypes)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write("Training set:")
    st.write(X_train)
    st.write("Test set:")
    st.write(X_test)
    
    return X_train, X_test, y_train, y_test




# 6. Entraînement des Modèles
def train_models(X_train, y_train, X_test, y_test, problem_type="classification"):
    st.subheader("Model Training")
    
    # Verify that the data is not empty
    if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
        st.error("One or more data arrays are empty. Please check your data and try again.")
        return None
    
    models = {}
    if problem_type == "classification":
        models = {
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(),
            "Support Vector Machine": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "AdaBoost": AdaBoostRegressor()
        }

    model_performance = {}
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if problem_type == "classification":
                model_performance[model_name] = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted'),
                    "Recall": recall_score(y_test, y_pred, average='weighted'),
                    "F1 Score": f1_score(y_test, y_pred, average='weighted')
                }
            else:
                model_performance[model_name] = {
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "R2": r2_score(y_test, y_pred)
                }
        except Exception as e:
            st.error(f"Error training model {model_name}: {e}")
    
    st.write("Model Performance:")
    st.write(pd.DataFrame(model_performance).T)
    return models

# 7. Interface de Prédiction
def prediction_interface(models, X_train, problem_type):
    st.subheader("Make Predictions")
    
    # Select model
    model_choice = st.selectbox("Select a model", list(models.keys()))
    model = models[model_choice]
    
    # Input new data
    input_data = {}
    for col in X_train.columns:
        input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)
        
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)
            st.write(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# 8. Fonction Main
def main():
    st.title("Interactive ML Application")
    
    # Organize sections into tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Upload", "EDA", "ML", "Prediction"])
    
    with tab1:
        df = upload_file()
    
    if df is not None:
        with tab2:
            display_summary_statistics(df)
            visualize_data(df)
            df = handle_missing_values(df)
        
        with tab3:

            target_column = st.selectbox("Select the target variable", df.columns)
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
            if X_train is not None and not X_train.empty:
                problem_type = st.selectbox("Is this a classification or regression problem?", ["classification", "regression"])
                models = train_models(X_train, y_train, X_test, y_test, problem_type=problem_type)
        
        with tab4:
            if 'models' in locals() and models:
                prediction_interface(models, X_train, problem_type)
            else:
                st.warning("No models available. Please complete the training step first.")

if __name__ == "__main__":
    main()