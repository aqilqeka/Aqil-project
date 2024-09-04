import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gdown
import zipfile

# Set page config to wide layout for a dashboard-style page
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")

# Function to download and load data from Google Drive
@st.cache_data
def load_data():
    url = 'https://drive.google.com/uc?id=1-p2oSWs3P4HPmxZjZ0EeQyzVZRhU8Kb3'
    output = 'dataset2.zip'
    
    gdown.download(url, output, quiet=False)
    
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall()

    fraud_train = pd.read_csv('fraudTrain.csv')
    fraud_test = pd.read_csv('fraudTest.csv')

    combined_df = pd.concat([fraud_train, fraud_test], ignore_index=True)
    
    return combined_df

# Load the data
if 'df' not in st.session_state:
    st.session_state['df'] = load_data()

df = st.session_state['df']

# Subsetting to a smaller dataset if necessary
df_subset = df.head(50000)

# Convert transaction time to datetime
df_subset['trans_date_trans_time'] = pd.to_datetime(df_subset['trans_date_trans_time'])

# Adjusting the height of charts for smaller layout
chart_height = 250

# Creating sections using columns and containers
# Section 1: Transaction Amount Distribution
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1 = px.histogram(df_subset, x='amt', color='is_fraud', nbins=50)
        fig1.update_layout(title="Transaction Amount Distribution", height=chart_height, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        transactions_over_time = df_subset.groupby(df_subset['trans_date_trans_time'].dt.date).agg({'trans_num': 'count', 'is_fraud': 'sum'}).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=transactions_over_time['trans_date_trans_time'], y=transactions_over_time['trans_num'], mode='lines', name='Total Transactions'))
        fig2.add_trace(go.Scatter(x=transactions_over_time['trans_date_trans_time'], y=transactions_over_time['is_fraud'], mode='lines', name='Fraudulent Transactions', line=dict(color='firebrick')))
        fig2.update_layout(title="Transactions Over Time", xaxis_title="Date", yaxis_title="Number of Transactions", height=chart_height, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fraud_by_category = df_subset[df_subset['is_fraud'] == 1].groupby('category')['is_fraud'].count().sort_values(ascending=False).reset_index()
        fig3 = px.bar(fraud_by_category, x='category', y='is_fraud')
        fig3.update_layout(title="Fraudulent Transactions by Merchant Category", height=chart_height, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig3, use_container_width=True)

# Section 2: Geographic Distribution, Transaction Velocity, and Correlation Matrix
with st.container():
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Grouping the data by gender, latitude, and longitude for fraud cases
        fraud_by_location_gender = df_subset[df_subset['is_fraud'] == 1].groupby(['merch_lat', 'merch_long', 'gender']).size().reset_index(name='fraud_count')

        fig4 = px.scatter_geo(
            fraud_by_location_gender,
            lat='merch_lat',
            lon='merch_long',
            color='gender',
            size='fraud_count',
            hover_name='gender',
            title="Geographic Distribution of Fraud by Gender",
            scope="usa",
            labels={'fraud_count': 'Fraud Cases'}
        )
        fig4.update_layout(height=chart_height, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig4, use_container_width=True)
    
    with col5:
        df_subset['hour'] = df_subset['trans_date_trans_time'].dt.hour
        velocity = df_subset.groupby('hour').size().reset_index(name='transaction_count')
        fig5 = px.bar(velocity, x='hour', y='transaction_count')
        fig5.update_layout(title="Transaction Velocity by Hour", height=chart_height, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        numeric_df = df_subset.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        correlation = numeric_df.corr()
        fig6 = px.imshow(correlation, text_auto=True)
        fig6.update_layout(title="Correlation Matrix of Features", height=chart_height, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig6, use_container_width=True)
