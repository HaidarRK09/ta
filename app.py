import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from math import sqrt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

# Konfigurasi halaman
st.set_page_config(
    layout="wide", page_title="Analisis Prediksi Dataset", page_icon="📊"
)
sns.set(style="whitegrid")

# Sidebar untuk kontrol
with st.sidebar:
    st.header("Pengaturan")
    uploaded_file = st.file_uploader("Unggah dataset CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            target_options = df.columns.tolist()
            target_column = st.selectbox(
                "Pilih kolom target",
                options=target_options,
                index=(
                    len(target_options) - 1 if "heart_attack" in target_options else 0
                ),
            )

            # Konversi kolom target ke biner jika perlu
            if df[target_column].nunique() > 2:
                st.warning(
                    "Kolom target memiliki lebih dari 2 nilai unik. Akan dikonversi ke biner (nilai paling sering = 0, lainnya = 1)"
                )
                most_common = df[target_column].mode()[0]
                df[target_column] = np.where(df[target_column] == most_common, 0, 1)

            # Batasi ukuran dataset untuk performa
            if len(df) > 10000:
                st.warning("Dataset terlalu besar, mengambil sample 5000 baris acak")
                df = df.sample(5000, random_state=42)

            st.session_state.df = df
            st.session_state.target_column = target_column

        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
            st.stop()

    test_size = st.slider("Ukuran data testing (%)", 10, 40, 20)
    random_state = st.number_input("Random State", 0, 100, 42)
    st.markdown("---")
    st.info(
        "Aplikasi ini menganalisis data dengan teknik SMOTE-Tomek dan membandingkan model Random Forest vs XGBoost."
    )

# Fungsi utama
def main():
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Silakan unggah file CSV dan pilih kolom target melalui sidebar")
        return

    df = st.session_state.df
    target_column = st.session_state.target_column

    # Tab untuk organisasi konten
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Eksplorasi Data",
            "⚙️ Preprocessing",
            "🔍 Analisis DBSCAN",
            "🤖 Model ML",
            "📈 Perbandingan",
        ]
    )

    with tab1:
        st.header("Eksplorasi Data Awal")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sample Data")
            st.dataframe(df.head())
            st.write(f"Dimensi dataset: {df.shape}")

        with col2:
            st.subheader("Informasi Dataset")
            buffer = st.container()
            with buffer:
                st.text(df.info())

        st.subheader(f"Distribusi Kolom Target: {target_column}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Countplot
        sns.countplot(data=df, x=target_column, ax=ax1)
        ax1.set_title(f"Distribusi {target_column}")

        # Pie chart
        counts = df[target_column].value_counts()
        ax2.pie(
            counts,
            labels=[f"Kelas {i}" for i in counts.index],
            autopct="%1.2f%%",
            startangle=140,
            colors=sns.color_palette("pastel"),
        )
        ax2.set_title("Proporsi Data")
        st.pyplot(fig)

        # Korelasi numerik
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(num_cols) > 1:  # Hanya tampilkan jika ada cukup kolom numerik
            st.subheader("Korelasi Variabel Numerik")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax
            )
            st.pyplot(fig)
        else:
            st.warning("Tidak cukup kolom numerik untuk menampilkan heatmap")

    with tab2:
        st.header("Preprocessing Data")

        # Pisah fitur dan target
        X = df.drop(columns=target_column)
        y = df[target_column]

        # Deteksi kolom
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # OneHot Encoding
        st.subheader("One-Hot Encoding")
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(X[cat_cols])
            encoded_df = pd.DataFrame(
                encoded, columns=encoder.get_feature_names_out(cat_cols)
            )

            # Normalisasi
            st.subheader("Normalisasi Fitur Numerik")
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(X[num_cols])
            scaled_df = pd.DataFrame(scaled, columns=num_cols)

            # Gabungkan hasil preprocessing
            X_processed = pd.concat(
                [scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)],
                axis=1,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.write("Sebelum Preprocessing:", X[cat_cols].head())
            with col2:
                st.write("Setelah Encoding:", encoded_df.head())

            col1, col2 = st.columns(2)
            with col1:
                st.write("Sebelum Normalisasi:", X[num_cols].head())
            with col2:
                st.write("Setelah Normalisasi:", scaled_df.head())

            st.success(f"Data berhasil diproses! Dimensi akhir: {X_processed.shape}")

            st.session_state.X_processed = X_processed
            st.session_state.y = y

        except Exception as e:
            st.error(f"Error dalam preprocessing: {str(e)}")
            st.stop()

    with tab3:
        st.header("Analisis DBSCAN untuk Data Minoritas")

        if "X_processed" not in st.session_state or "y" not in st.session_state:
            st.warning("Harap selesaikan preprocessing di tab sebelumnya")
            st.stop()

        try:
            # Pastikan index konsisten
            X_processed = st.session_state.X_processed.reset_index(drop=True)
            y = st.session_state.y.reset_index(drop=True)

            # Validasi final
            if len(X_processed) != len(y):
                raise ValueError(
                    f"Jumlah sampel tidak sama (X: {len(X_processed)}, y: {len(y)})"
                )

            if not set(y.unique()).issubset({0, 1}):
                raise ValueError(f"Target mengandung nilai selain 0 dan 1: {y.unique()}")

            # Pemisahan data yang aman
            mask = (y == 1).to_numpy()
            minority_X = X_processed[mask]
            majority_X = X_processed[~mask]
            minority_y = y[mask]
            majority_y = y[~mask]

            st.session_state.minority_X = minority_X
            st.session_state.majority_X = majority_X

        except Exception as e:
            st.error(f"Error persiapan data DBSCAN: {str(e)}")
            st.stop()

        # Parameter DBSCAN dengan session state untuk mempertahankan nilai
        if "eps" not in st.session_state:
            st.session_state.eps = 2.3
        if "min_samples" not in st.session_state:
            st.session_state.min_samples = 86

        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider(
                "Nilai EPS untuk DBSCAN",
                1.0,
                5.0,
                st.session_state.eps,
                0.1,
                key="eps_slider",
            )
        with col2:
            min_samples = st.slider(
                "Min Samples untuk DBSCAN",
                5,
                100,
                st.session_state.min_samples,
                key="min_samples_slider",
            )

        # Button untuk menjalankan DBSCAN
        if st.button("Jalankan DBSCAN"):
            st.session_state.eps = eps
            st.session_state.min_samples = min_samples

            with st.spinner("Menjalankan DBSCAN..."):
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    dbscan.fit(minority_X)
                    labels = dbscan.labels_

                    # Hitung titik
                    core_points = set(dbscan.core_sample_indices_)
                    noise_points = set(
                        [i for i, label in enumerate(labels) if label == -1]
                    )
                    border_points = (
                        set(range(len(minority_X))) - core_points - noise_points
                    )

                    # Visualisasi
                    st.subheader("Hasil Clustering DBSCAN")
                    pca = PCA(n_components=2)
                    minority_X_2d = pca.fit_transform(minority_X)

                    fig = plt.figure(figsize=(10, 6))
                    plt.scatter(
                        minority_X_2d[list(noise_points), 0],
                        minority_X_2d[list(noise_points), 1],
                        c="red",
                        s=20,
                        label="Noise Points",
                        alpha=0.9,
                    )
                    plt.scatter(
                        minority_X_2d[list(border_points), 0],
                        minority_X_2d[list(border_points), 1],
                        c="orange",
                        s=5,
                        label="Border Points",
                        alpha=0.6,
                    )
                    plt.scatter(
                        minority_X_2d[list(core_points), 0],
                        minority_X_2d[list(core_points), 1],
                        c="blue",
                        s=5,
                        label="Core Points",
                        alpha=0.6,
                    )
                    plt.title("Visualisasi PCA (DBSCAN pada Data Minoritas)")
                    plt.legend()
                    st.pyplot(fig)

                    # Info cluster
                    st.info(
                        f"""
                    **Hasil DBSCAN:**
                    - Total Data Minoritas: {len(minority_X)}
                    - Core Points: {len(core_points)} 
                    - Border Points: {len(border_points)}
                    - Noise Points: {len(noise_points)}
                    """
                    )

                    # Simpan hasil untuk tab berikutnya
                    st.session_state.core_points = core_points
                    st.session_state.minority_X = minority_X
                    st.session_state.majority_X = majority_X
                    st.session_state.minority_y = minority_y
                    st.session_state.majority_y = majority_y

                except Exception as e:
                    st.error(f"Error dalam DBSCAN: {str(e)}")
                    st.stop()
        else:
            if "core_points" in st.session_state:
                st.info(
                    "Gunakan tombol 'Jalankan DBSCAN' untuk memperbarui hasil dengan parameter baru"
                )

    with tab4:
        st.header("Pemodelan Machine Learning")

        # Gunakan hanya core points
        core_data = minority_X.iloc[list(core_points)]
        core_y = pd.Series([1] * len(core_data))

        # Gabungkan dengan mayoritas untuk SMOTE
        X_smote_input = pd.concat([core_data, majority_X], axis=0)
        y_smote_input = pd.concat([core_y, majority_y], axis=0)

        # SMOTE-Tomek
        smote_tomek = SMOTETomek(random_state=random_state)
        X_final, y_final = smote_tomek.fit_resample(X_smote_input, y_smote_input)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_final,
            y_final,
            test_size=test_size / 100,
            random_state=random_state,
            stratify=y_final,
        )

        # Pilih model
        model_option = st.selectbox("Pilih Model", ["Random Forest", "XGBoost"])

        if st.button("Train Model"):
            if model_option == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=100, random_state=random_state
                )
            else:
                model = XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=random_state,
                )

            # Training
            model.fit(X_train, y_train)

            # Evaluasi
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Tampilkan hasil
            st.subheader("Hasil Evaluasi Model")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

            with col2:
                st.write("Classification Report:")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

            # ROC Curve
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(
                fpr, tpr, color="darkorange", label=f"ROC curve (area = {roc_auc:.2f})"
            )
            ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            ax.legend(loc="lower right")
            st.pyplot(fig)

            # Feature Importance
            st.subheader("Feature Importance")
            if model_option == "Random Forest":
                importances = model.feature_importances_
            else:
                importances = model.feature_importances_

            indices = np.argsort(importances)[-10:][::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(indices)), importances[indices], color="skyblue")
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(X_final.columns[indices])
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title("Top 10 Feature Importance")
            st.pyplot(fig)

    with tab5:
        st.header("Perbandingan Model")

        if st.button("Bandingkan Kedua Model"):
            # Train Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100, random_state=random_state
            )
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

            # Train XGBoost
            xgb_model = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
            )
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

            # Hitung metrik
            metrics = {
                "Accuracy": [
                    accuracy_score(y_test, y_pred_rf),
                    accuracy_score(y_test, y_pred_xgb),
                ],
                "Precision (0)": [
                    precision_score(y_test, y_pred_rf, pos_label=0),
                    precision_score(y_test, y_pred_xgb, pos_label=0),
                ],
                "Recall (0)": [
                    recall_score(y_test, y_pred_rf, pos_label=0),
                    recall_score(y_test, y_pred_xgb, pos_label=0),
                ],
                "F1-score (0)": [
                    f1_score(y_test, y_pred_rf, pos_label=0),
                    f1_score(y_test, y_pred_xgb, pos_label=0),
                ],
                "Precision (1)": [
                    precision_score(y_test, y_pred_rf, pos_label=1),
                    precision_score(y_test, y_pred_xgb, pos_label=1),
                ],
                "Recall (1)": [
                    recall_score(y_test, y_pred_rf, pos_label=1),
                    recall_score(y_test, y_pred_xgb, pos_label=1),
                ],
                "F1-score (1)": [
                    f1_score(y_test, y_pred_rf, pos_label=1),
                    f1_score(y_test, y_pred_xgb, pos_label=1),
                ],
                "AUC-ROC": [
                    roc_auc_score(y_test, y_prob_rf),
                    roc_auc_score(y_test, y_prob_xgb),
                ],
            }

            # Tampilkan tabel perbandingan
            comparison_df = pd.DataFrame(metrics, index=["Random Forest", "XGBoost"])
            st.dataframe(
                comparison_df.style.format("{:.4f}").background_gradient(cmap="Blues")
            )

            # Visualisasi perbandingan
            st.subheader("Visualisasi Perbandingan")
            fig, ax = plt.subplots(figsize=(12, 6))
            comparison_df.T.plot(kind="bar", ax=ax)
            ax.set_ylabel("Score")
            ax.set_title("Perbandingan Model Random Forest vs XGBoost")
            ax.legend(title="Model")
            st.pyplot(fig)


# Simpan data di session state setelah upload
if uploaded_file is not None:
    st.session_state.df = df
    st.session_state.target_column = target_column

if __name__ == "__main__":
    main()
