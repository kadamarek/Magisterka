import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

st.header("Analiza reguł asocjacyjnych")

uploaded_file = st.file_uploader("Choose a file")  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    min_support = st.slider('Minimum support', min_value=0.0, max_value=1.0, value=0.1, step=0.01)  # Minimalne wsparcie
    min_confidence = st.slider('Minimum confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.01)  # Minimalne zaufanie

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Filtruj reguły według minimalnego zaufania
    rules = rules[rules['confidence'] >= min_confidence]

    st.write("Liczba reguł znalezionych: ", len(rules))

    st.write("Najważniejsze reguły:")
    st.table(rules)