''' Import the Libraries '''
''' ************************************************'''
# For Dashboard
import dash
from dash_bootstrap_components._components.CardImg import CardImg
from dash_bootstrap_components._components.Row import Row
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from dash_html_components.Br import Br
from dash_html_components.Img import Img
from numpy.core.numeric import outer

# for Plot 
import plotly.express as px
import plotly.graph_objs as go

# use boostrap
import dash_bootstrap_components as dbc   

# For read the dataset 
import pandas_datareader.data as web
import pandas as pd

# For mathematical operations 
import numpy as np   
 
# To draw a heatmap
import seaborn as sns
sns.set_style('whitegrid')

# To get information from Yahoo
import yfinance as yf 
from yahoo_fin import stock_info as si #To get an instant price

# For time stamps
from datetime import datetime

# For the LSTM
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


''' download and read dataset from Yahoo and Stooq'''
''' ************************************************'''

# Select time range
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# read stocks dataset from 'stooq'
df_stock = web.DataReader(['META','AAPL','AMZN','NFLX','GOOGL'],
                    'stooq', start = start, end = end)
df_stock = df_stock.stack().reset_index()
df_stock.to_csv("mystocks.csv", index = False)
dfs = pd.read_csv("mystocks.csv")    # read dataset
print(dfs[:5])

# read google dataset from 'yahoo' 
df_goog = yf.download("GOOG", start, end)
df_goog.to_csv('google.csv')

'''---------- LSTM ----------'''
''' ************************************************'''

# read the stock Price
df = pd.read_csv("google.csv")
df["Date"] = pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index = df['Date']
    
# Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
    
# Converting the dataframe to a numpy array
dataset = data.values
    
# Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8)
    
# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
    
# Create the scaled training data set
train_data = scaled_data[0:training_data_len  , : ]
    
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
    
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
        
# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
    
# Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
# Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
    
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
    
# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
    
# Test data set
test_data = scaled_data[training_data_len - 60: , : ]
    
# Create the x_test and y_test data sets
x_test = []
    
    
y_test =  dataset[training_data_len : , : ] 
    
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
        
# Convert x_test to a numpy array 
x_test = np.array(x_test)
    
# Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    
# Getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling
    
# Calculate/Get the value of RMSE
rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
#rmse
    
# Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions



''' ************************************************'''
#app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Stock Price Prediction"


''' --------- Layout ---------  '''
''' ************************************************'''

app.layout = dbc.Container([
    # Row-01 ( Header )
    dbc.Row([
        dbc.Col(
             html.H1("Stock Market Dashboard" ,className='text-center text-dark, my-4'),
                          width=12)
    ]),

    # Tabs
    dcc.Tabs(id="tabs", children=[
        # Tab Analytic Stock
        dcc.Tab(label='FAANG Stocks',children=[

            # Row-0 ( Drodown list )
            dbc.Row([
                # Col-0
                dbc.Col([
                    # title
                    html.H3("Close Price USD ($) " ,className='text-center text-dark, my-4'),

                    # dropdown
                    dbc.Card([
                        dcc.Dropdown(id = 'my-dpdn', multi = True , value = '',
                        options = [{'label':x, 'value':x}
                                  for x in sorted(dfs['Symbols'].unique())
                        ], style = {"width": "95%" , "margin": "auto" })

                    # style css code
                    ], style = {"width": "85%", "height":"6rem", 
                                 "box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.18)",
                                 "margin": "auto" , "padding": "10px"})      
                ])
            ]),
            
            # Row-01 ( Close Chart )
            dbc.Row([
                dbc.Col([
                    # Chart
                    dcc.Graph(id = 'fig',
                    figure = {
                            "layout": {"title": "Close Prise"} 
                    # style
                     }, className = "my-4",
                        style = {"box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.15)",
                                  "width": "85%", "margin": "auto"}) # style code
                ])
            ]),
            
            # Row-02 
            dbc.Row([
                dbc.Col([
                    # titel ( histogram )
                    html.H1("Histogram Chart",className='text-center text-dark, my-4')
                ])
            ]),
            
            # Row-03
            dbc.Row([
                # Show histogeram figure
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            # title
                            html.H6("Select the Compony:", className = "font-weight-bolder"),
                            dcc.Checklist(id = 'my-chek', value = '',
                            # select compony name from database
                            options=[{'label':x, 'value':x}
                                   for x in sorted(dfs['Symbols'].unique())]
                                   , labelClassName="mr-3"),
                        # Chart
                        dcc.Graph(id='my-hist', figure={}),
                        ])
                    ], style = {"width": "85%", "margin": "auto" ,   # style css
                               "box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.18)"}),
                    html.Br()
                ])
            ])

        ]), # end tab1
        
        # Tab google
        dcc.Tab(label='Google Stock',children=[
            # Row-00 ( Close figure )
            dbc.Row([
                dbc.Col([
                    # title 
                    html.H2("Close figure", className = 'text-center blockquote, my-4'),

                    # Chart 
                    dcc.Graph(id = "Close",
                    figure= { 
                        "data":[
                            go.Scatter(
                                x=train.index,
                                y=valid["Close"],
                                mode = 'lines',
                                line = dict(shape = 'linear', color = 'rgb(10, 120, 24)', dash = 'dot'),
                                connectgaps = True
                        )],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )}
                        
                        # style
                        , className = "my-4",
                        style = {"box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.15)",
                                  "width": "85%", "margin": "auto"}
                       )
                ])
            ]),
            
            # Row-01 ( predict Chart )
            dbc.Row([
                dbc.Col([ 
                    #  Header and Description
                     html.H2("Predicted figure", className = 'text-center blockquote, my-4'),
                     html.P("This program uses an artificial recurrent neural network called"
                            " Long Short Term Memory (LSTM)", 
                              className="text-decoration-none text-center"),
                    
                    # Chart
                    dcc.Graph(id = "Predicted",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid.index,
                                y=valid["Predictions"],
                                mode = 'lines',
                                line = dict(shape = 'linear', color = 'rgb(100, 10, 100)', dash = 'dot'),
                                connectgaps = True
                            )],

                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )}
                        , className = "my-4",

                        # style code css
                        style = {"box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.15)",
                                  "width": "85%", "margin": "auto"}),
                    # next line
                    html.Br(),
                    html.Br()
                ])
            ])

        ]) # end tab2
    ])
], style = {"background-color": "#F7F7F7"}) # style body




''' ---------- Callback ---------- '''
''' ************************************************'''

# Close Price Chart
@app.callback(
    Output('fig', 'figure'),
    Input('my-dpdn', 'value')
)
def close_graph(stock_slctd):
    dff = dfs[dfs['Symbols'].isin(stock_slctd)]
    figln = px.line(dff, x = 'Date', y = 'Close', color = 'Symbols')
    return figln

# Histogram Chart
@app.callback(
    Output('my-hist', 'figure'),
    Input('my-chek', 'value')
)
def update_graph(stock_slctd):
    dff = dfs[dfs['Symbols'].isin(stock_slctd)]
    dff = dff[dff['Date']=='2021-09-01']  # Date Today
    fighist = px.histogram(dff, x = 'Symbols', y = 'Close', color = 'Symbols')
    return fighist

''' -------- Run Server -------- '''
''' ************************************************'''

if __name__ == "__main__":
    app.run_server()