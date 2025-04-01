import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyarrow import parquet as pq
from catboost import CatBoostClassifier, Pool
import joblib

MODEL_PATH = 'model/catboost_model.cbm'
DATA_PATH = 'data\data.parquet'

st.set_page_config(page_title='Chrun Prediction - Machine Learning')

@st.cache_resource
def load_data():
    data = pd.read_parquet(DATA_PATH)
    return data

def load_x_y(file_path):
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data

def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

def calculate_shap(model, X_train, X_test):
    explainer = shap.TreeExplainer(model=model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test

def plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_train, x_test):
    customer_index = x_test[x_test['customerID']==customer_id].index[0]
    fig, ax_2 =  plt.subplot(figsize=(6,6), dpi=200)
    shap.decision_plot(explainer.expected_values, shap_values_cat_test[customer_index], x_test[x_test['customerID'] == customer_id], link='logit')
    st.pyplot(fig)
    plt.close()

def display_shap_summary(shap_values_cat_train, X_train):
    shap.summary_plot(shap_values_cat_train, X_train, plot_type='bar', plot_size=(12,12))
    summary_fig, _ = plt.gcf(), plt.gcf()
    st.pyplot(summary_fig)
    plt.close()

# start from display_shap_waterfall function

def display_shap_waterfall_plot(explainer, expected_values, shap_values, feature_names, max_display=20):
    fig , ax = plt.subplots(figsize=(6,6), dpi=150)
    shap.waterfall_plot.waterfall_legacy(expected_values, shap_values, feature_names=feature_names, max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()

def summary(model , data, X_train, x_test):
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, x_test)
    display_shap_summary(shap_values_cat_train, X_train)


def plot_shap(model, data, customer_id, X_train, x_test):
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, x_test)
    plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_train, x_test)
    customer_index = x_test[x_test['customerID'] == customer_id].index[0]
    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_test[customer_index], feature_names=x_test.columns, max_display=20)


st.title('Telco customer churn prediction')


## main function for the streamlit app

def main():
    model = load_model()
    data = load_data()

    X_train = load_x_y('data/X_train.pkl')
    x_test = load_x_y('data/x_test.pkl')
    Y_train = load_x_y('data/Y_train')
    y_test = load_x_y('data/y_test')

    max_tenure = data['tenure'].max()
    max_monthly_charges = data['MonthlyCharges'].max()
    max_total_charges = data['TotalCharges'].max()
    ### Start from the radio button comment from the code.
    election = st.radio("Make Your Choice:", ("Feature Importance", "User-based SHAP", "Calculate the probability of CHURN"))





