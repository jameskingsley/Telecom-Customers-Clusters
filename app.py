import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import os
import shap
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

#App config
st.set_page_config(page_title="Telecom Customer Segmentation App", layout="wide")
st.title("Telecom Customer Segmentation with KMeans")

MODEL_PATH = "kmeans_model.joblib"
SCALER_PATH = "scaler.joblib"

#Required features
expected_features = ['age', 'gender', 'estimated_salary', 'num_dependents', 'calls_made', 'sms_sent', 'data_used']

#Segment labels for the clusters
segment_labels = {
    0: "Tech-Savvy High Users",
    1: "Low Engagement Seniors",
    2: "Balanced Users",
    3: "Call & SMS Centric Users",
    4: "Infrequent Young Users"
}

@st.cache_resource
def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

def save_model_and_scaler(model, scaler):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    st.success("Model and scaler saved.")

uploaded_file = st.file_uploader(f"Upload your customer CSV file with columns: {expected_features}", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")

    #Check if required columns exist
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
        st.stop()

    df_selected = df[expected_features].copy()

    #Encode gender if necessary
    if df_selected['gender'].dtype == 'object':
        df_selected['gender'] = LabelEncoder().fit_transform(df_selected['gender'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected)

    st.sidebar.header("Options")
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)

    if st.sidebar.checkbox("Show Elbow Method"):
        sse = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=42)
            km.fit(X_scaled)
            sse.append(km.inertia_)
        fig_elbow, ax = plt.subplots()
        ax.plot(range(1, 11), sse, marker='o')
        ax.set_title("Elbow Method - SSE")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("SSE")
        st.pyplot(fig_elbow)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df['cluster'] = labels

    df['segment'] = df['cluster'].map(segment_labels).fillna("Other")

    st.markdown("### Clustering Evaluation Metrics")
    st.write(f"- **Calinski-Harabasz Score:** {calinski_harabasz_score(X_scaled, labels):.2f}")
    st.write(f"- **Davies-Bouldin Score:** {davies_bouldin_score(X_scaled, labels):.2f}")

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = pca_components[:, 0], pca_components[:, 1]

    fig_pca = px.scatter(
        df, x='PCA1', y='PCA2', color='segment',
        title="PCA Clustering Visualization",
        labels={'color': 'Segment'},
        template="plotly_white"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    #Cluster summary table (only numeric columns)
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'cluster' in numeric_cols:
            numeric_cols.remove('cluster')

        summary = df.groupby('cluster')[numeric_cols].mean().round(2)
        summary['count'] = df['cluster'].value_counts().sort_index()
        summary['segment'] = summary.index.map(segment_labels).fillna("Other")

        st.markdown("### Cluster Summary Table")
        st.dataframe(summary)
    except Exception as e:
        st.error(f"Could not generate cluster summary. Error: {e}")
        summary = None

    if summary is not None:
        st.markdown("### Cluster Size Distribution")
        fig_bar = px.bar(
            summary.reset_index(), x='cluster', y='count', text='count',
            labels={"cluster": "Cluster", "count": "Customer Count"},
            color='segment', template='simple_white'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Download Clustered Dataset")
    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "clustered_customers.csv", "text/csv")

    if st.sidebar.button("Save Model and Scaler"):
        save_model_and_scaler(kmeans, scaler)

    st.markdown("### Real-Time Cluster Prediction")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", 10, 100, 30)
        gender = col2.selectbox("Gender", ["Male", "Female"])
        salary = col3.number_input("Estimated Salary", 1000, 1000000, 50000)
        dependents = col1.number_input("Number of Dependents", 0, 10, 1)
        calls = col2.number_input("Calls Made", 0, 1000, 50)
        sms = col3.number_input("SMS Sent", 0, 1000, 20)
        data_used = col1.number_input("Data Used (GB)", 0.0, 100.0, 5.0)

        submitted = st.form_submit_button("Predict Segment")
        if submitted:
            gender_enc = 1 if gender == "Male" else 0
            input_data = np.array([[age, gender_enc, salary, dependents, calls, sms, data_used]])

            model_loaded, scaler_loaded = load_model_and_scaler()
            if model_loaded and scaler_loaded:
                input_scaled = scaler_loaded.transform(input_data)
                prediction = model_loaded.predict(input_scaled)[0]
                st.success(f"Predicted Customer Segment: {segment_labels.get(prediction, prediction)}")
            else:
                st.warning("Model and scaler not found. Please save them first.")

    st.markdown("### Model Explainability with SHAP")
    model_loaded, scaler_loaded = load_model_and_scaler()
    if model_loaded and scaler_loaded:
        try:
            if X_scaled.shape[0] > 0:
                explainer = shap.KernelExplainer(model_loaded.predict, X_scaled[:100])
                shap_values = explainer.shap_values(X_scaled[:100])

                #Handle case if shap_values is a list (multi-output)
                if isinstance(shap_values, list):
                    shap_vals = np.array(shap_values).mean(axis=0)
                else:
                    shap_vals = shap_values

                shap.summary_plot(
                    shap_vals,
                    features=pd.DataFrame(X_scaled[:100], columns=expected_features),
                    show=False
                )
                plt.tight_layout()
                st.pyplot(plt.gcf())
            else:
                st.info("Not enough data for SHAP explainability.")
        except Exception as e:
            warnings.warn(f"SHAP plotting error: {e}")
            st.warning(f"SHAP explainability error: {e}")
    else:
        st.info("Save and load a model to see explainability.")

else:
    st.info(f"Please upload a CSV file containing these columns: {expected_features}")
