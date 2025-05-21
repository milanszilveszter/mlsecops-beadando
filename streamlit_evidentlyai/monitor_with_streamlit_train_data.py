import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("# Train Dataset ")
st.sidebar.markdown("# Train Dataset ")

def get_raw_data():
    df_train = pd.read_csv('data/crop_train.csv')
    df_test = pd.read_csv('data/crop_test.csv')
    return df_train, df_test

header = st.container()
dataset = st.container()
plot_area_code = st.container()
feature_distributions = st.container()
correlation_heatmap = st.container()
pairwise_relationship = st.container()

with header:
    st.title('Crop Recommendation Dataset Monitoring')
    st.text("A dashboard exploring agricultural input data.")

with dataset:
    st.header("Train Dataset Sample")
    df_train, df_test = get_raw_data()

with plot_area_code:
    st.header("Label Distribution (Most Suitable Crops)")
    label_counts = df_train["label"].value_counts()
    st.bar_chart(label_counts)

with feature_distributions:
    st.header("Feature Distributions")
    numeric_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    selected_feature = st.selectbox("Choose a feature to visualize its distribution", numeric_features)
    fig, ax = plt.subplots()
    sns.histplot(df_train[selected_feature], kde=True, bins=30, ax=ax)
    st.pyplot(fig)

with correlation_heatmap:
    st.header("Correlation Heatmap")
    corr = df_train[numeric_features].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with pairwise_relationship:
    st.header("Pairwise Feature Relationships")
    st.text("Colored by crop label (might be slow on large datasets)")
    selected_features = st.multiselect("Select up to 4 features for pairplot", numeric_features, default=numeric_features[:3])
    if len(selected_features) >= 2:
        fig = sns.pairplot(df_train, vars=selected_features, hue="label", diag_kind="kde", corner=True)
        st.pyplot(fig)
    else:
        st.warning("Please select at least two features.")

# streamlit run monitor_with_streamlit_train_data.py