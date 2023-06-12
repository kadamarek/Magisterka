import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


uploaded_file = st.file_uploader("Choose a file")   #Uploader plików
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) #Odczytanie zbioru danych
    st.write(df)   #Wypisanie data frame pliku
    column_list = list(df.columns)
    selected_columns = st.multiselect("Select columns for features", column_list[:-1],
                                      default=[column_list[0], column_list[1]])  # Wybór kolumn dla cech
    selected_labels = st.multiselect("Select columns for labels", column_list[-1:], default=[column_list[-1]])  # Wybór kolumn dla etykiet

    features = df[selected_columns]  # Wyodrębnienie części warunkowej danych
    labels = df[selected_labels]  # Wyodrębnienie kolumny decyzyjnej

    ts = st.number_input('Test size', min_value=0.0, max_value=1.0, value=0.6, step=0.01) #Ustalenie tablicy treningowej
    rs = st.number_input('Random State', min_value=1, max_value=10000, value=1234, step=1) #Ustalenie ziarna generatora liczb pseudolosowych

    datasets = train_test_split(features, labels, test_size=ts, random_state=rs)

    features_train = datasets[0]
    features_test = datasets[1]
    labels_train = datasets[2]
    labels_test = datasets[3]

    n_estimators = st.number_input('Number of Estimators', min_value=1, max_value=10000, value=100, step=1)  #Liczba estymatorów w drzewie losowym
    max_depth = st.number_input('Max Depth', min_value=1, max_value=10000, value=5, step=1)  #Maksymalna głębokość drzewa

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)  #Utworzenie obiektu klasyfikatora drzewa losowego
    model.fit(features_train, np.ravel(labels_train)) #Uczenie klasyfikatora na części treningowej

    labels_predicted = model.predict(features_test) #Generowanie decyzji dla części testowej

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)  #Policzenie jakości klasyfikacji

    st.write("Classification accuracy=", accuracy)
    st.write("========= FULL CLASSIFICATION RESULTS ================")
    rf_report = classification_report(labels_test, labels_predicted, output_dict=True)
    df_rf_report = pd.DataFrame(rf_report).transpose()
    st.write(df_rf_report)
    st.write("====== CONFUSION MATRIX =========")
    rf_conf_matrix = confusion_matrix(labels_test, labels_predicted)
    st.write(rf_conf_matrix)