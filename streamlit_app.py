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

# Adjusting the height of charts for smaller layout
chart_height = 250

st.header("Credit Card Fraud Detection 💳")

# Custom CSS for the background color behind the text in columns
st.markdown("""
    <style>
    .stat-card {
        background-color: #20B2AA;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Data Overview with background color in columns
with st.container():
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="stat-card">Total Transactions: {:,}</div>'.format(df.shape[0]), unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="stat-card">Total Fraudulent Transactions: {:,}</div>'.format(df['is_fraud'].sum()), unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="stat-card">Fraud Rate: {:.2f}%</div>'.format(df['is_fraud'].mean() * 100), unsafe_allow_html=True)

# Create Tabs
tab1, tab2 = st.tabs(["**📜 Main**", "🔎 **Prediction**"])

with tab1: 

    # Add space between the first container and the slider
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Slider for selecting subset size placed below the first container
    subset_size = st.slider('Select the number of rows to visualize:', min_value=1000, max_value=1852394, step=1000, value=100000)

    # Subsetting the data based on the slider value
    df_subset = df.head(subset_size)

    # Convert transaction time to datetime
    df_subset['trans_date_trans_time'] = pd.to_datetime(df_subset['trans_date_trans_time'])

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

    # Section 2: Fraud Cases by Hour, Transaction Velocity by Hour (Separated), and Fraud by Job
    with st.container():
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # Mapping 0 to 'Normal' and 1 to 'Fraud'
            df_subset['fraud_status'] = df_subset['is_fraud'].map({0: 'Normal', 1: 'Fraud'})
            df_subset['hour'] = df_subset['trans_date_trans_time'].dt.hour
            # Creating a histogram for transaction velocity by hour, separating normal and fraudulent transactions
            fig5 = px.histogram(
                df_subset, 
                x='hour', 
                color='fraud_status', 
                barmode='group',
                title="Transaction Velocity by Hour (Normal vs. Fraud)",
                nbins=24,  # One bin for each hour of the day
                labels={'hour': 'Hour of Day', 'count': 'Transaction Count', 'fraud_status': 'Transaction Type'},
            )

            fig5.update_layout(
                height=chart_height, 
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(tickmode='linear'),
                yaxis_title="Transaction Count"
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col5:
            # Filter the data to include only hours 22 and 23 and fraudulent transactions
            
            late_hours_fraud = df_subset[(df_subset['is_fraud'] == 1) & (df_subset['hour'].isin([22, 23]))]

            # Group by merchant category
            fraud_by_merchant = late_hours_fraud.groupby('category').size().reset_index(name='fraud_count')

            # Creating a bar chart for fraud cases by merchant category during hours 22 and 23
            fig4 = px.bar(fraud_by_merchant, x='category', y='fraud_count', title="Fraudulent Transactions by Merchant (Hours 22 & 23)")
            fig4.update_layout(
                height=chart_height, 
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Merchant Category",
                yaxis_title="Fraud Count",
                xaxis={'categoryorder':'total descending'},  # Order by descending fraud count
            )
            st.plotly_chart(fig4, use_container_width=True)
            



        with col6:
            # Grouping the data by job
            fraud_by_job = df_subset[df_subset['is_fraud'] == 1].groupby('job')['is_fraud'].count().reset_index().sort_values(by='is_fraud', ascending=False)

            # Creating a bar chart for fraud by job
            fig6 = px.bar(fraud_by_job, x='job', y='is_fraud', title="Fraudulent Transactions by Job")
            fig6.update_layout(height=chart_height, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig6, use_container_width=True)

with tab2:
    st.write('test')
