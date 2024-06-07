import requests
import urllib.request
import json
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta
import numpy as np
from hmm import MarketPhasePredictor
from scipy.signal import argrelextrema



################################################################################################################################################################
###### Get Top 100 coins by market cap
################################################################################################################################################################


def calculate_market_cap(symbol):
    # Send a GET request to the Binance API
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    response = requests.get(url)
    data = response.json()

    # Extract the volume and weighted average price from the response
    volume = float(data['volume'])
    weighted_avg_price = float(data['weightedAvgPrice'])

    # Calculate the product of the volume and weighted average price
    result = volume * weighted_avg_price

    return result


def get_top_100_coins_by_market_cap():
    
    # Get all coins from the Binance API
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)
    data = response.json()

    # Filter the coins that are traded against USDT
    usdt_coins = [coin['symbol'] for coin in data if coin['symbol'].endswith('USDT')]

    # Calculate the market cap for each USDT coin
    market_caps = {}
    for coin in usdt_coins:
        market_caps[coin] = calculate_market_cap(coin)
        time.sleep(0.1)

    # Sort the coins by their market cap
    coins = list(market_caps.items())
    coins.sort(key=lambda x: x[1], reverse=True)

    # Get the top 100 coins
    top_100_coins = [coin[0] for coin in coins[:100]]

    filename = './data/meta/top_100_coins.json'
    with open(filename, 'w') as f:
        json.dump(top_100_coins, f)

    return top_100_coins


################################################################################################################################################################
###### Get first timestamps of Top 100 coins
################################################################################################################################################################

def find_first_timestamp(interval, symbol, start_ts=1499990400000):
    '''
    Returns the first kline from an interval and start timestamp and symbol
    :param interval:  1w, 1d, 1m etc - the bar length to query
    :param symbol:    BTCUSDT or LTCBTC etc
    :param start_ts:  Timestamp in miliseconds to start the query from
    :return:          The first open candle timestamp
    '''

    url_stub = "http://api.binance.com/api/v1/klines?interval="

    #/api/v1/klines?interval=1m&startTime=1536349500000&symbol=ETCBNB
    addInterval   = url_stub     + str(interval) + "&"
    addStarttime  = addInterval   + "startTime="  + str(start_ts) + "&"
    addSymbol     = addStarttime + "symbol="     + str(symbol)
    url_to_get = addSymbol

    # debug
    print(url_to_get)

    kline_data = urllib.request.urlopen(url_to_get).read().decode("utf-8")
    kline_data = json.loads(kline_data)

    return kline_data[0][0]


def get_first_timestamps_list(coins):

    first_timestamps = {}
    interval = "1d"
    binance_start = 1499990400000

    # File name
    file_name = f"./data/meta/coin_first_timestamps_{interval}.json"

    # Check if the file exists
    if os.path.exists(file_name):
        # Load the existing data
        with open(file_name, 'r') as f:
            existing_data = json.load(f)
    else:
        # If the file doesn't exist, create an empty dictionary
        existing_data = {}

    # Update the existing data with the new data, but only for coins that are not already in the file
    for coin in coins:
        if coin not in existing_data:
            
            if interval == "1w":
                first_timestamps[coin] = find_first_timestamp(interval, coin, binance_start )
            elif interval == "1d":
                week_start = find_first_timestamp(interval, coin, binance_start )
                first_timestamps[coin] = find_first_timestamp(interval, coin, week_start )
            elif interval == "1m":
                week_start = find_first_timestamp(interval, coin, binance_start )
                day_start = find_first_timestamp(interval, coin, week_start )
                first_timestamps[coin] = find_first_timestamp(interval, coin, day_start )
            else:
                print("Unknown interval: 1w, 1d, 1m are valid.")
            
            existing_data[coin] = first_timestamps[coin]

    # Save the updated data to the file
    with open(file_name, 'w') as f:
        json.dump(existing_data, f)



################################################################################################################################################################
###### Get historical data from Binance
################################################################################################################################################################

def get_historical_data(symbol, data_interval, request_interval, start_time, end_time):
    url = "https://api.binance.com/api/v3/klines"
    df = pd.DataFrame()

    while start_time < end_time:
        params = {
            'symbol': symbol,
            'interval': data_interval,
            'startTime': int(start_time.timestamp() * 1000),  # Binance uses milliseconds
            'endTime': int((start_time + timedelta(minutes=request_interval)).timestamp() * 1000),  # Binance uses milliseconds
        }
        response = requests.get(url, params=params)
        data = response.json()

        if 'code' in data:
            print(f"Error code: {data['code']}, Error message: {data['msg']}")
            break

        temp_df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Close time', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        temp_df.drop(['Ignore', 'Close time'], axis=1, inplace=True)

        df = pd.concat([df, temp_df])

        # Update start_time to the time of the last kline plus interval
        start_time += timedelta(minutes=request_interval)

    filename1 = f'./data/raw/{symbol}.csv'
    filename2 = f'./data/raw/{symbol}_raw.csv'      # Save a copy of the raw data for producing different data sets
    df.to_csv(filename1, index=False)
    df.to_csv(filename2, index=False)

    return df


################################################################################################################################################################
###### Calculate Technical Indicators
################################################################################################################################################################

def calculate_indicators(data_file):
    # Load the data from the file
    data = pd.read_csv(data_file)

    # Calculate the 7-day moving average of the 'Close' price
    data['Close_MA_7'] = data['Close'].rolling(window=7).mean()

    # Calculate the 14-day moving average of the 'Close' price
    data['Close_MA_14'] = data['Close'].rolling(window=14).mean()
    
    # Calculate the 25-day moving average of the 'Close' price
    data['Close_MA_25'] = data['Close'].rolling(window=25).mean()

    # Calculate the 50-day moving average of the 'Close' price
    data['Close_MA_50'] = data['Close'].rolling(window=50).mean()

    # Calculate MACD
    data['macd'] = ta.trend.MACD(data['Close']).macd()

    # Calculate RSI
    data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()

    # Calculate Aroon Indicator
    aroon_indicator = ta.trend.AroonIndicator(low=data['Low'], high=data['High'])
    data['aroon_down'] = aroon_indicator.aroon_down()
    data['aroon_up'] = aroon_indicator.aroon_up()

    # Calculate On Balance Volume
    data['obv'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()

    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['bollinger_hband'] = bollinger.bollinger_hband()
    data['bollinger_lband'] = bollinger.bollinger_lband()

    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
    data['stoch'] = stoch.stoch()

    data.to_csv(data_file, index=False)
    
    return data


################################################################################################################################################################
###### Calculate Target Class
################################################################################################################################################################

def add_target_class(symbol):
    
    # Call the function for the symbol.csv data as well as symbol_phases.csv data
    data = target_class_helper(symbol)
    data_phase = target_class_helper(symbol + '_phases')

    return data, data_phase

def target_class_helper(symbol):
        
    data = pd.read_csv(f"./data/raw/{symbol}.csv")
   
    # Check for local minima and maxima
    data['class'] = is_extrema(data['Close'])
    data['class'] = data['class'].astype('category')

    # Save the data with the new 'class_swing' column appended
    data.to_csv(f"./data/raw/{symbol}_min_max_ti.csv", index=False)

    # Create a normalized data set in addition
    data_normalized = data.copy()
    scaler = MinMaxScaler()
    normalize_columns = [col for col in data.columns if col not in ['Open time','phase', 'class']]
    data_normalized[normalize_columns] = scaler.fit_transform(data_normalized[normalize_columns])
    data_normalized.to_csv(f"./data/raw/{symbol}_min_max_ti_n.csv", index=False)

    return data, data_normalized

################################################################################################################################################################
###### Handle Empty Values
################################################################################################################################################################

def replace_empty_values(data_file):
    
    data = pd.read_csv(data_file)
    
    # Fill missing values with the mean of the column
    data.fillna(data.mean(), inplace=True)

    data.to_csv(data_file, index=False)
    
    return data

################################################################################################################################################################
###### Data Set Preparation
################################################################################################################################################################

# A) Data Set with Indicators

def prepare_ti_data(symbol, data_interval='1d'):
    
    with open('./data/meta/coin_first_timestamps_1d.json', 'r') as f:
        start_dates = json.load(f)

    # Get the start date for the symbol
    start_date = start_dates.get(symbol)
    # convert from int to datetime
    start_date = datetime.fromtimestamp(start_date / 1000)

    # get data one year at a time to not run into API limits
    request_interval = 24 * 60 * 365
    request_interval = int(request_interval)

    end_date = datetime.now()
    get_historical_data(symbol, data_interval=data_interval, request_interval=request_interval, start_time=start_date, end_time=end_date)

    data_file = f'./data/raw/{symbol}.csv'
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                data = pd.read_csv(data_file)
            else:
                print(f"No data to read from {data_file}")
                data = pd.DataFrame()
    else:
        print(f"{data_file} does not exist")
        data = pd.DataFrame()

    data = calculate_indicators(data_file)
    data = replace_empty_values(data_file)
    
    # Predict market phases with HMM
    predictor = MarketPhasePredictor(n_components=9, covariance_type="full", n_iter=140, symbol=symbol)
    predictor.train()
    predictor.phase_recognition()
    
    data = add_target_class(symbol)
    
    
    return data

# B) Minimal Data Set with Min-Max-Class and Regression Coefficients

def calculate_regression_coef(window):
    
    # Calculate the regression coefficient for the data in the window
    coef = np.polyfit(range(len(window)), window, 1)[0]
    return coef

def is_extrema(x):
    
    # Find the indices of the local maxima
    maxima = argrelextrema(x.values, np.greater)
    # Find the indices of the local minima
    minima = argrelextrema(x.values, np.less)
    # Create an array of zeros with the same length as x
    extrema = np.zeros_like(x)
    # Set the values at the maxima indices to 1
    extrema[maxima] = 1
    # Set the values at the minima indices to -1
    extrema[minima] = -1

    extrema = extrema.astype(int)
    
    return extrema

def move_class_to_end(data_path, symbol):
    
    data = pd.read_csv(f'{data_path}/{symbol}.csv')
    # Create a new list of column names with 'class' at the end
    cols = [col for col in data if col != 'class'] + ['class']
    # Reorder the columns
    data = data[cols]
    # Save the data
    data.to_csv(f'{data_path}/{symbol}.csv', index=False)

def prepare_min_data(raw_data_path, output_data_path, symbol):
    
    data_file = f'{raw_data_path}/{symbol}.csv'
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                data = pd.read_csv(data_file)
                
                # Check for local minima and maxima
                data['class'] = is_extrema(data['Close'])
                data['class'] = data['class'].astype('category')

                # Calculate the normalized price column
                # If it is close to 1, the coin has closed near the high of the day, if it is close to 0, the coin has closed near the low of the day
                data['price_normalized'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])

                # Calculate the regression coefficients for the last 3, 5, 10, and 20 days - represents the direction of price development
                for days in [3, 5, 10, 20]:
                    data[f'{days}_reg'] = data['price_normalized'].rolling(days).apply(calculate_regression_coef)
                data.fillna(0, inplace=True)    # the first few values will be NaN, fill them with 0
            
                # Keep only desired columns for minimal data set
                data = data[['Open time', 'Volume', 'price_normalized', '3_reg', '5_reg', '10_reg', '20_reg', 'class']]	
            else:
                print(f"No data to read from {data_file}")
                pass
    else:
        print(f"{data_file} does not exist")
        pass
    
    save_path = f'{output_data_path}/{symbol}_min_max.csv'
    data.to_csv(save_path, index=False)

    # Predict market phases with HMM
    hmm_symbol = symbol + '_min_max'
    hmm_input_path = output_data_path
    predictor = MarketPhasePredictor(n_components=9, covariance_type="full", n_iter=140, symbol=hmm_symbol, input_path=hmm_input_path)      # optimal parameters from optimization test
    predictor.train()
    predictor.phase_recognition()

    hmm_symbol_phases = hmm_symbol + '_phases'
    move_class_to_end(output_data_path, hmm_symbol_phases)


   
################################################################################################################################################################
###### MAIN
################################################################################################################################################################

# Get the top 100 coins
if not os.path.exists('./data/meta/top_100_coins.json'):
    top_100_coins = get_top_100_coins_by_market_cap()

# Get the first timestamps for the top 100 coins
if not os.path.exists('./data/meta/coin_first_timestamps_1d.json'):
    get_first_timestamps_list(top_100_coins)

# Data preparation for BTCUSDT
prepare_ti_data('BTCUSDT', data_interval='1d') # Data set with technical indicators
prepare_min_data(raw_data_path='./data/raw', output_data_path='./data/raw', symbol='BTCUSDT') # Minimal data set with normalized price and regression coefficients