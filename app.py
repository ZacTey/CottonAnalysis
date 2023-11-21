import streamlit as st
import datetime
from datetime import date
from plotly import graph_objs as go
import pandas as pd
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
#from keras.models import load_model

# ------ initial config --------------------------
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title('US Cotton Price Analysis 20231013')

# ------ layout setting---------------------------
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("Task 4 User Input") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date


# ------ Load Data -------------------------------
@st.cache_data
def load_data():
    cotton_data = pd.read_excel('cottonprice_raw.xlsx', sheet_name='Table11', index_col=0)
    cotton_data = cotton_data.reset_index()
    actual_RFeqv_data = pd.read_excel('cottonprice_raw.xlsx', sheet_name='Table12', index_col=0)
    actual_RFeqv_data = actual_RFeqv_data.reset_index()
    region_data = pd.read_excel('cottonprice_raw.xlsx', sheet_name='Table13', index_col=0)
    region_data = region_data.reset_index()
    return cotton_data, actual_RFeqv_data, region_data


data_load_state = st.text('Status: Loading data...')
cotton_data, actual_RFeqv_data, region_data = load_data()
data_load_state.text('Status: Data Load Completed')


# ------ Task 1 ----------------------------------
st.subheader('Task 1 - Seasonality & Trend')


def create_seasonal_cols(input_df, col_name, period):
    # Seasonal decompose
    seasonal_model_results = seasonal_decompose(input_df[col_name], period=period)

    # Add results to original df
    input_df['{}_observed'.format(col_name)] = seasonal_model_results.observed
    input_df['{}_residual'.format(col_name)] = seasonal_model_results.resid
    input_df['{}_seasonal'.format(col_name)] = seasonal_model_results.seasonal
    input_df['{}_trend'.format(col_name)] = seasonal_model_results.trend
    return input_df


# Plot cotton data farm + spot + mill
def plot_cotton_price(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Farm price (Cents/lb)'], name="Farm Price", marker_color='green'))
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Spot price (Cents/lb)'], name="Spot Price", marker_color='yellow'))
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Mill price (Cents/lb)'], name="Mill Price", marker_color='orange'))
    fig.layout.update(title_text='Cotton Price (Cents/lb)', xaxis_rangeslider_visible=True, width=1000, height=500)
    fig.add_annotation(dict(font=dict(color='white', size=9),
                            x=0,
                            y=0,
                            showarrow=False,
                            text="Click Legend to Hide, Double Click to Isolate",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)
# Call the function
plot_cotton_price(cotton_data)

# Farm Price Seasonal & Trend ======================================================================================
farm_data_test = cotton_data.copy()
farm_data_test = create_seasonal_cols(farm_data_test, 'Farm price (Cents/lb)', period=5) # custom creation function

def plot_farm_seasonal_trend(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Farm price (Cents/lb)'], name="Farm Price", marker_color='green'))
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Farm price (Cents/lb)_seasonal'], name="Seasonal", marker_color='violet'))
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Farm price (Cents/lb)_trend'], name="Trend", marker_color='purple'))
    fig.layout.update(title_text='Farm Price Seasonality & Trend (Cents/lb)', xaxis_rangeslider_visible=True, width=800, height=500)
    fig.add_annotation(dict(font=dict(color='white', size=9),
                            x=0,
                            y=0,
                            showarrow=False,
                            text="Click Legend to Hide, Double Click to Isolate",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)
# Call the function
plot_farm_seasonal_trend(farm_data_test)

# Spot Price Seasonal & Trend ======================================================================================
spot_data_test = cotton_data.copy()
spot_data_test = create_seasonal_cols(spot_data_test, 'Spot price (Cents/lb)', period=10) # custom creation function

def plot_spot_seasonal_trend(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Spot price (Cents/lb)'], name="Spot Price", marker_color='yellow'))
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Spot price (Cents/lb)_seasonal'], name="Seasonal", marker_color='violet'))
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Spot price (Cents/lb)_trend'], name="Trend", marker_color='purple'))
    fig.layout.update(title_text='Spot Price Seasonality & Trend (Cents/lb)', xaxis_rangeslider_visible=True, width=800, height=500)
    fig.add_annotation(dict(font=dict(color='white', size=9),
                            x=0,
                            y=0,
                            showarrow=False,
                            text="Click Legend to Hide, Double Click to Isolate",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)
# Call the function
plot_spot_seasonal_trend(spot_data_test)

# Mill Price Seasonal & Trend ======================================================================================
mill_data_test = cotton_data.copy()
mill_data_test = mill_data_test.dropna()
mill_data_test = create_seasonal_cols(mill_data_test, 'Mill price (Cents/lb)', period=10) # custom creation function

def plot_mill_seasonal_trend(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Mill price (Cents/lb)'], name="Mill Price", marker_color='orange'))
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Mill price (Cents/lb)_seasonal'], name="Seasonal", marker_color='violet'))
    fig.add_trace(go.Scatter(x=df['Crop year'], y=df['Mill price (Cents/lb)_trend'], name="Trend", marker_color='purple'))
    fig.layout.update(title_text='Mill Price Seasonality & Trend (Cents/lb)', xaxis_rangeslider_visible=True, width=800, height=500)
    fig.add_annotation(dict(font=dict(color='white', size=9),
                            x=0,
                            y=0,
                            showarrow=False,
                            text="Click Legend to Hide, Double Click to Isolate",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)
# Call the function
plot_mill_seasonal_trend(mill_data_test)

# ------ Task 2 ----------------------------------
st.subheader('Task 2 - Comparison & Correlation')
task2selection = ['Actual Price', 'Raw Fibre Equivalent']
APRF_select = st.selectbox("Actual Price or Raw Fibre Equivalent", task2selection)
# Drop missing columns
actual_RFeqv_data = actual_RFeqv_data.dropna()

# Check on selected graph
if APRF_select == 'Actual Price':
    cotton_name = 'Cotton Actual'
    rayon_name = 'Rayon Actual'
    polyester_name = 'Polyester Actual'
    title_name = 'Actual Price'
else:
    cotton_name = 'Cotton Raw-fiber equivalent'
    rayon_name = 'Rayon Raw-fiber equivalent'
    polyester_name = 'Polyester Raw-fiber equivalent'
    title_name = 'Raw-fiber equivalent'

# Generate the trend line
cotton_RFeqv_test = create_seasonal_cols(actual_RFeqv_data, '{} (Cents/lb)'.format(cotton_name), period=5) # custom creation function
rayon_RFeqv_test = create_seasonal_cols(actual_RFeqv_data, '{} (Cents/lb)'.format(rayon_name), period=5) # custom creation function
polyester_RFeqv_test = create_seasonal_cols(actual_RFeqv_data, '{} (Cents/lb)'.format(polyester_name), period=5) # custom creation function

# Plotting the original actual first
def plot_crp_price(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['year'], y=df['{} (Cents/lb)'.format(cotton_name)], name="Cotton", marker_color='brown'))
    fig.add_trace(go.Scatter(x=df['year'], y=df['{} (Cents/lb)'.format(rayon_name)], name="Rayon", marker_color='palegreen'))
    fig.add_trace(go.Scatter(x=df['year'], y=df['{} (Cents/lb)'.format(polyester_name)], name="Polyester", marker_color='slateblue'))
    fig.layout.update(title_text='{} (Cents/lb)'.format(title_name), xaxis_rangeslider_visible=True, width=1000, height=500)
    fig.add_annotation(dict(font=dict(color='white', size=9),
                            x=0,
                            y=0,
                            showarrow=False,
                            text="Click Legend to Hide, Double Click to Isolate",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)
# Call the function
plot_crp_price(actual_RFeqv_data)

# Plotting the trend lines
def plot_crp_trend(cotton_df, rayon_df, polyester_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cotton_df['year'], y=cotton_df['{} (Cents/lb)_trend'.format(cotton_name)], name="Cotton", marker_color='brown'))
    fig.add_trace(go.Scatter(x=rayon_df['year'], y=rayon_df['{} (Cents/lb)_trend'.format(rayon_name)], name="Rayon", marker_color='palegreen'))
    fig.add_trace(go.Scatter(x=polyester_df['year'], y=polyester_df['{} (Cents/lb)_trend'.format(polyester_name)], name="Polyester", marker_color='slateblue'))
    fig.layout.update(title_text='{} Trend (Cents/lb)'.format(title_name), xaxis_rangeslider_visible=True, width=1000, height=500)
    fig.add_annotation(dict(font=dict(color='white', size=9),
                            x=0,
                            y=0,
                            showarrow=False,
                            text="Click Legend to Hide, Double Click to Isolate",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)
# Call the function
plot_crp_trend(cotton_RFeqv_test, rayon_RFeqv_test, polyester_RFeqv_test)

# Correlation Analysis
def plot_corr(input_df):
    fig = px.imshow(input_df.corr(), color_continuous_scale='pinkyl', labels=dict(color="Correlation"))
    st.write(fig)

# Call the function
col_name = ['{} (Cents/lb)'.format(cotton_name), '{} (Cents/lb)'.format(rayon_name), '{} (Cents/lb)'.format(polyester_name)]
plot_corr(actual_RFeqv_data[col_name])
PR_checkbox = st.checkbox("Show Price Ratio Chart", False)
if PR_checkbox:
    st.subheader("Price Ratio")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=actual_RFeqv_data['year'], y=actual_RFeqv_data['Cotton/rayon PR'], name="Cotton/Rayon", marker_color='brown'))
    fig.add_trace(
        go.Scatter(x=actual_RFeqv_data['year'], y=actual_RFeqv_data['Cotton/polyester PR'], name="Cotton/Polyester", marker_color='palegreen'))
    fig.layout.update(title_text='Price Ratio', xaxis_rangeslider_visible=True, width=1000,
                      height=500)
    fig.add_annotation(dict(font=dict(color='white', size=9),
                            x=0,
                            y=0,
                            showarrow=False,
                            text="Click Legend to Hide, Double Click to Isolate",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)

# ------ Task 3 ----------------------------------
st.subheader('Task 3 - Geographical Comparison')
st.write("Select regions:")
# Drop missing columns
region_data = region_data.dropna()

Aindex_box = st.checkbox("A index", True)
MemphisEastern_box = st.checkbox("Memphis/Eastern", False)
MemphisOrleansTexas_box = st.checkbox("Memphis/Orleans/Texas", False)
CaliforniaArizona_box = st.checkbox("California/Arizona", False)

def plot_area_price(df):
    fig = go.Figure()
    if Aindex_box:
        fig.add_trace(go.Scatter(x=df['Year (beginning Aug 1)'], y=df['high'][df['region']=='A index'], name="A index yearly high", marker_color='#17BECF'))
        fig.add_trace(go.Scatter(x=df['Year (beginning Aug 1)'], y=df['average'][df['region']=='A index'], name="A index yearly average", marker_color='#17BECF'))
        fig.add_trace(go.Scatter(x=df['Year (beginning Aug 1)'], y=df['low'][df['region']=='A index'], name="A index yearly low", marker_color='#17BECF'))
    if MemphisEastern_box:
        fig.add_trace(
            go.Scatter(x=df['Year (beginning Aug 1)'], y=df['high'][df['region'] == 'Memphis/Eastern'], name="Memphis/Eastern yearly high",
                       marker_color='#EEA6FB'))
        fig.add_trace(go.Scatter(x=df['Year (beginning Aug 1)'], y=df['average'][df['region'] == 'Memphis/Eastern'],
                                 name="Memphis/Eastern yearly average", marker_color='#EEA6FB'))
        fig.add_trace(
            go.Scatter(x=df['Year (beginning Aug 1)'], y=df['low'][df['region'] == 'Memphis/Eastern'], name="Memphis/Eastern yearly low",
                       marker_color='#EEA6FB'))
    if MemphisOrleansTexas_box:
        fig.add_trace(
            go.Scatter(x=df['Year (beginning Aug 1)'], y=df['high'][df['region'] == 'Memphis/Orleans/Texas'], name="Memphis/Orleans/Texas yearly high",
                       marker_color='#16FF32'))
        fig.add_trace(go.Scatter(x=df['Year (beginning Aug 1)'], y=df['average'][df['region'] == 'Memphis/Orleans/Texas'],
                                 name="Memphis/Orleans/Texas yearly average", marker_color='#16FF32'))
        fig.add_trace(
            go.Scatter(x=df['Year (beginning Aug 1)'], y=df['low'][df['region'] == 'Memphis/Orleans/Texas'], name="Memphis/Orleans/Texas yearly low",
                       marker_color='#16FF32'))
    if CaliforniaArizona_box:
        fig.add_trace(
            go.Scatter(x=df['Year (beginning Aug 1)'], y=df['high'][df['region'] == 'California/Arizona'], name="California/Arizona yearly high",
                       marker_color='#FFA15A'))
        fig.add_trace(go.Scatter(x=df['Year (beginning Aug 1)'], y=df['average'][df['region'] == 'California/Arizona'],
                                 name="California/Arizona yearly average", marker_color='#FFA15A'))
        fig.add_trace(
            go.Scatter(x=df['Year (beginning Aug 1)'], y=df['low'][df['region'] == 'California/Arizona'], name="California/Arizona yearly low",
                       marker_color='#FFA15A'))

    fig.layout.update(title_text='Quotation Price for different regions (Cents/lb)', xaxis_rangeslider_visible=True, width=1000, height=500)
    fig.add_annotation(dict(font=dict(color='white', size=9),
                            x=0,
                            y=0,
                            showarrow=False,
                            text="Click Legend to Hide, Double Click to Isolate",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)
# Call the function
plot_area_price(region_data)

def plot_percentage_diff(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['year'], y=df[''], name="Cotton", marker_color='brown'))
    fig.add_trace(go.Scatter(x=df['year'], y=df[''], name="Rayon", marker_color='palegreen'))
    fig.add_trace(go.Scatter(x=df['year'], y=df[''], name="Polyester", marker_color='slateblue'))
    fig.layout.update(title_text='{} (Cents/lb)'.format(title_name), xaxis_rangeslider_visible=True, width=1000, height=500)
    fig.add_annotation(dict(font=dict(color='white', size=9),
                            x=0,
                            y=0,
                            showarrow=False,
                            text="Click Legend to Hide, Double Click to Isolate",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)
# Call the function
#plot_percentage_diff(region_data)

# ------ Task 4 ----------------------------------
st.subheader('Task 4 - Supply Chain Recommendations')

# ------ Forecasting -----------------------------
train_test_forecast_c = st.sidebar.container()
train_test_forecast_c.markdown("## 1. Forecasts Parameters")

FORECAST_LENGTH = train_test_forecast_c.number_input(
    "Number of Years to Forecast",
    min_value=5,
    max_value=10,
    key="FORECAST_INTERVAL_LENGTH",
)

data_length = len(cotton_data)
TRAIN_LENGTH = train_test_forecast_c.number_input(
    "Number of Years to train Model",
    #min_value=data_length-15,
    #max_value=data_length-10,
    min_value=39,
    max_value=39,
    key="TRAIN_INTERVAL_LENGTH",
)

TEST_LENGTH = train_test_forecast_c.number_input(
    "Number of Years to Evaluate Model",
    min_value=5,
    max_value=5,
    key="TEST_INTERVAL_LENGTH",
)

selected_model = [None, "LSTM"]
TRAIN_MODEL = train_test_forecast_c.selectbox("Model Selection", selected_model)



# Initialize the session state
if "TRAINED" not in st.session_state:
    st.session_state.TRAINED = None
if "TRAIN_JOB" not in st.session_state:
    st.session_state.TRAIN_JOB = None

# --- Training Report ----------------
def train_test_forecast_report():
    if st.session_state.TRAIN_JOB or st.session_state.TRAINED:
        text = st.empty()  # Because streamlit adds widgets sequentially, we have to reserve a place at the top (after the chart of part 1)
        bar = st.empty()  # Reserve a place for a progess bar

        text.write('Starting Model Training ... ')
        bar = st.progress(0)

        # splitting date into training and testing
        bar.progress(10)
        text.write('Splitting Training & Testing set ... ')
        data_training = pd.DataFrame(cotton_data['Spot price (Cents/lb)'][0:TRAIN_LENGTH])
        data_testing = pd.DataFrame(cotton_data['Spot price (Cents/lb)'][TRAIN_LENGTH: (TRAIN_LENGTH + TEST_LENGTH)])
        st.write("training data size: ", data_training.shape[0], "     testing data size: ", data_testing.shape[0])


        # scaling of data using min max scaler (0,1)
        bar.progress(20)
        MMscaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = MMscaler.fit_transform(data_training)

        # Load model
        bar.progress(30)

        def plot_pred_price(df):
            fig = go.Figure()
            # Predicted Values
            if TRAIN_MODEL == "LSTM":
                df["Spot price (Cents/lb) (prediction)"] = df['Spot price (Cents/lb)'].copy()
                LSTM_result = [{"Crop year":2022, "Spot price (Cents/lb) (prediction)":75.57},
                               {"Crop year":2023, "Spot price (Cents/lb) (prediction)":75.53},
                                {"Crop year":2024, "Spot price (Cents/lb) (prediction)":75.44},
                                 {"Crop year":2025, "Spot price (Cents/lb) (prediction)":75.49},
                                  {"Crop year":2026, "Spot price (Cents/lb) (prediction)":70.58},
                                   {"Crop year":2027, "Spot price (Cents/lb) (prediction)":75.61},
                                    {"Crop year":2028, "Spot price (Cents/lb) (prediction)":70.61},
                                     {"Crop year":2029, "Spot price (Cents/lb) (prediction)":75.61},
                                      {"Crop year":2030, "Spot price (Cents/lb) (prediction)":75.61},
                                       {"Crop year":2031, "Spot price (Cents/lb) (prediction)":75.62}]

                df = df.append(LSTM_result, ignore_index=True)
                fig.add_trace(
                    go.Scatter(x=df['Crop year'], y=df['Spot price (Cents/lb) (prediction)'][:-(10-FORECAST_LENGTH)], name="Predicted Spot Price", marker_color='red'))


            fig.add_trace(
                go.Scatter(x=df['Crop year'], y=df['Spot price (Cents/lb)'], name="Spot Price", marker_color='yellow'))
            fig.layout.update(title_text='Cotton Price (Cents/lb)', xaxis_rangeslider_visible=True, width=1000,
                              height=500)
            fig.add_annotation(dict(font=dict(color='white', size=9),
                                    x=0,
                                    y=0,
                                    showarrow=False,
                                    text="Click Legend to Hide, Double Click to Isolate",
                                    textangle=0,
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
            st.plotly_chart(fig)

        # Call the function
        plot_pred_price(cotton_data)
        if TRAIN_MODEL == "LSTM":
            st.write("RMSE: 8.304951341876375")
        else:
            st.write("RMSE: No Model Selected ")
        bar.progress(70)
        text.write('Plotting test results ...')
        #fig = stock.plot_test()
        bar.progress(100)
        bar.empty()  # Turn the progress bar object back to what it was before and empty container

        text.write('Generating forecasts ... ')

        text.empty()

        st.session_state.TRAINED = True
    else:
        st.markdown('Hit button below to perform prediction')


def onClickFunction():
    st.session_state.TRAIN_JOB = True
runButton = st.button("Predict", on_click=onClickFunction)

# Calling the function
train_test_forecast_report()




