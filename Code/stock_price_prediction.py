''' Import the Libraries '''
import pandas as pd
import numpy as np 

# To draw a diagram
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
#%matplotlib inline

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



''' Download stock data from yahoo then export as CSV '''
''' ------------------ FAANG STOCKS ----------------- '''

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Download stock data then export as CSV
df_fb = yf.download("FB", start, end)
df_fb.to_csv('facebook.csv')

df_aapl = yf.download("AAPL", start, end)
df_aapl.to_csv('apple.csv')

df_amzn = yf.download("AMZN", start, end)
df_amzn.to_csv('amazon.csv')

df_nflx = yf.download("NFLX", start, end)
df_nflx.to_csv('netflix.csv')

df_goog = yf.download("GOOG", start, end)
df_goog.to_csv('google.csv')



''' Analyze the closing prices from dataset '''

# Plot stock company['Close']
def figure_close(stockname , name):
    df = pd.read_csv(stockname)
    df["Date"] = pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index = df['Date']
    
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot()
    ax.set_title(name)
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Close Price USD ($)', fontsize=16)
    plt.plot(df["Close"],label='Close Price history')



''' The total volume of stock being traded each day: '''

# Plot stock compony['Volume']
def figure_Sales(stockname , name):
    df = pd.read_csv(stockname)
    df["Date"] = pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index = df['Date']
    
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot()
    ax.set_title(name)
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Close Price USD ($)', fontsize=16)
    plt.plot(df["Volume"],label='Close Price history')



''' Use sebron for a quick correlation plot for the daily returns '''

# The tech stocks we'll use for this analysis
tech_list = ['FB', 'AAPL', 'AMZN', 'NFLX', 'GOOG']

# Download stock data then export as CSV
df_close = yf.download(tech_list, start, end)['Adj Close']
df_close.to_csv('closing.csv')

# Plot all the close prices
((df_close.pct_change()+1).cumprod()).plot(figsize=(16,8))

# Show the legend
plt.legend()

# Define the label for the title of the figure
plt.title("Returns", fontsize=16)
plt.ylabel('Cumulative Returns', fontsize=14)
plt.xlabel('Year', fontsize=14)

# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()

# Heatmap
sns.heatmap(df_close.corr(), annot=True, cmap='summer')




''' Real-Time stock price '''

# import stock_info module from yahoo_fin
def get_Price(stock , name):
    price = si.get_live_price(stock)
    print(name,"stock price now: " , price)
    


''' Predicting the closing price stock price with "LSTM" '''

# For Warning 
pd.options.mode.chained_assignment = None  # default='warn'

def LSTM_Model(stockname , name):
    # read the stock Price
    df = pd.read_csv(stockname)
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
    
    # Visualize the data
    plt.figure(figsize=(16,8))
    plt.title(name)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


''' Print the result Function '''

print()
print('Apple')
get_Price('aapl' , 'Apple Inc.')
figure_close('apple.csv' , 'Apple')
figure_Sales('apple.csv' , 'Sales Volume for Apple')
LSTM_Model('apple.csv' , 'Apple')

''' For other companies, 
    the output is as above, 
    just select the name of the company
    (Facebook Inc./ Apple Inc./ Amazon Inc./ Netflix Inc./ Google Inc.) 
    The names of the csv files:
    (facebook.csv, apple.csv, amazon.csv, netflix.csv, google.csv)''' 