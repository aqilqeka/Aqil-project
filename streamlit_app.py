import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gdown
import zipfile
import pickle
import joblib
import sklearn
from io import BytesIO

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
# tab1, tab2 = st.tabs(["**📜 Main**", "🔎 **Prediction**"])

# with tab1: 

    # Add space between the first container and the slider
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Slider for selecting subset size placed below the first container
subset_size = st.slider('Select the number of rows to visualize:', min_value=1000, max_value=1852394, step=1000, value=100000)

# Subsetting the data based on the slider value
df_subset = df.head(subset_size)

# Convert transaction time to datetime using .loc to avoid SettingWithCopyWarning
df_subset.loc[:, 'trans_date_trans_time'] = pd.to_datetime(df_subset['trans_date_trans_time'])

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
        # Mapping 0 to 'Normal' and 1 to 'Fraud' using .loc
        df_subset.loc[:, 'fraud_status'] = df_subset['is_fraud'].map({0: 'Normal', 1: 'Fraud'})
        df_subset.loc[:, 'hour'] = df_subset['trans_date_trans_time'].dt.hour
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
        # Filter the data to include only hours 22 and 23 and fraudulent transactions using .loc
        late_hours_fraud = df_subset.loc[(df_subset['is_fraud'] == 1) & (df_subset['hour'].isin([22, 23]))]

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
        
# with tab2:
#     # Display the scikit-learn version
#     st.write(f"Scikit-learn version: {sklearn.__version__}")

#     # Function to download and load data from Google Drive (for the model)
#     @st.cache_data
#     def download_model():
#         url = 'https://drive.google.com/uc?id=1lxKmn_m_ETOWpIQWcK5kTPtRRy-ikPP4'  # Replace with your Google Drive model file link
#         output = 'project2.pkl'
#         gdown.download(url, output, quiet=False)
#         model = joblib.load(output)  # Use joblib to load the model
#         return model

#     # Load the model
#     if 'model' not in st.session_state:
#         st.session_state['model'] = download_model()

#     model = st.session_state['model']

#     # Creating the Prediction tab
#     st.header("Upload File for Prediction")
#     st.markdown("Upload a CSV or Excel file (Max 25MB) to predict fraudulent transactions.")

#     # File uploader - limiting the size to 25MB
#     uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], help="Limit: 25MB", accept_multiple_files=False)

#     if uploaded_file is not None:
#         # Check file size (limit to 25MB)
#         if uploaded_file.size > 25 * 1024 * 1024:  # Convert MB to bytes
#             st.error("File size exceeds the 25MB limit. Please upload a smaller file.")
#         else:
#             try:
#                 # If it's a CSV file
#                 if uploaded_file.name.endswith('.csv'):
#                     data = pd.read_csv(uploaded_file, header=None)
#                     # Define correct column headers
#                     correct_headers = ['trans_date_trans_time', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'dob']
#                     data.columns = correct_headers

#                 # If it's an Excel file
#                 elif uploaded_file.name.endswith('.xlsx'):
#                     data = pd.read_excel(uploaded_file, header=None)
#                     # Define correct column headers
#                     correct_headers = ['trans_date_trans_time', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'dob']
#                     data.columns = correct_headers

#                 st.write(f"File successfully loaded! Shape: {data.shape}")

#                 # Assuming the model requires these specific set of features to predict
#                 features = ['trans_date_trans_time', 'category', 'amt', 'city', 'state', 'lat', 'long', 'city_pop', 'dob']  # Use your actual feature names
#                 X = data[features]  # Extract features for prediction

#                 # Making predictions
#                 predictions = model.predict(X)
#                 data['Predictions'] = predictions

#                 # Display the predicted results
#                 st.write("Predictions added to the dataset:")
#                 st.dataframe(data.head())  # Display the first few rows with predictions

#                 # Allow users to download the results with predictions
#                 csv = data.to_csv(index=False).encode('utf-8')
#                 st.download_button(
#                     label="Download predictions as CSV",
#                     data=csv,
#                     file_name='predictions.csv',
#                     mime='text/csv',
#                 )

#             except Exception as e:
#                 st.error(f"An error occurred while processing the file: {e}")
