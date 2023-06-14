from sklearn.cluster import DBSCAN
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

uploaded_file = st.file_uploader("Choose a file", key='2')  # Uploader plików

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Odczytanie zbioru danych
    st.write(df)  # Wypisanie data frame pliku

    column_list = list(df.columns)
    selected_columns = st.multiselect("Select columns for clustering", column_list)

    if len(selected_columns) >= 2:
        features = df[selected_columns]  # Wybór kolumn do grupowania
        e = st.number_input('The maximum distance between observations to be considered adjacent.', min_value=0,
                                max_value=1000, value=3, step=1)
        ms = st.number_input('Minimum samples', min_value=0, max_value=1000, value=5, step=1)
        met = st.selectbox('Select one of the available metrics', ('euclidean', 'manhattan', 'l1', 'l2'))

        db = DBSCAN(eps=e, min_samples=ms, metric=met)
        db.fit(features)

        clusters = db.fit_predict(features)

        x = np.ravel(features.iloc[:, [0]])
        y = np.ravel(features.iloc[:, [1]])

        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, s=10, c=clusters, alpha=0.9)

        # Dodanie etykiet wybranych kolumn
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])

        ax.legend([scatter], ['Data Points'])
        st.pyplot(fig)

        st.write("Przyporządkowanie poszczególnych obiektów do skupień:")
        data = {"Skupienie": [], "Obiekty": []}
        for i in range(len(np.unique(clusters))):
            cluster_indices = [index for index, cluster in enumerate(clusters) if cluster == i]
            objects = ", ".join([str(index + 1) for index in cluster_indices])
            data["Skupienie"].append(i)
            data["Obiekty"].append(objects)
        cluster_data = pd.DataFrame(data)
        st.table(cluster_data)
    else:
        st.write("Wybierz co najmniej dwie kolumny do grupowania.")