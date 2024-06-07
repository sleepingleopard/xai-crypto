import glob
import os
import pandas as pd
from binance.client import Client
from cn2 import CN2Predictor
from lstm_class import LSTMPredictor
from rulefit import RuleFitModel


class trader:
    
    def __init__(self, symbol='BTCUSDT', base_currency='USDT', risk_per_trade=0.01, trading_strat='min_max_day'):
        
        self.symbol = symbol
        self.base_currency = base_currency
        self.risk = risk_per_trade
        self.trading_strat = trading_strat
        self.balance = 0
        self.portfolio = {}
        #self.client = Client('your_api_key', 'your_api_secret')
        self.prepare_strats()
        #self.get_balance()
        #self.current_price()

        '''
        The trading_strat can be either 'moderate_hold' or 'swing'. The params for single trades are set automatically based on the trading_strat, so that it fits the respective model approach. 
        The moderate_hold is accepting little drawdowns and holds for a maximum of 10d, while the swing trader is not accepting losses and moves on to the next opportunity while holding a maximum of 21d.
        With the risk_per_trade parameter, you can set the percentage of your balance that you want to risk per trade. 1% is a good starting point, to bear possible loosing trades. 
        '''

    def get_balance(self):
        
        self.balance = float(self.client.get_asset_balance(asset=self.base_currency)['free'])

    def current_price(self):
        
        ticker = self.client.get_symbol_ticker(symbol=self.symbol)
        self.current_price = float(ticker['price'])
    
    def prepare_strats(self):
          
        trade_parameters = {
            'moderate_hold': {
                'max_trade_duration': 10, 'max_trade_loss': 0.03, 'max_trade_gain': 0.03
            },
            'swing': {
                'max_trade_duration': 21, 'max_trade_loss': 0.003, 'max_trade_gain': 0.04
            },
            'min_max_day': {
                'max_trade_duration': 1, 'max_trade_loss': 0.003, 'max_trade_gain': 0.04
            }
        }

        self.max_trade_duration = trade_parameters[self.trading_strat]['max_trade_duration']
        self.max_trade_loss = trade_parameters[self.trading_strat]['max_trade_loss']
        self.max_trade_gain = trade_parameters[self.trading_strat]['max_trade_gain']
    
    
    def calculate_prices(self, is_buying):
        
        if is_buying:
            stop_loss_price = self.current_price * (1 - self.max_trade_loss)
            take_profit_price = self.current_price * (1 + self.max_trade_gain)
        else:  # selling
            stop_loss_price = self.current_price * (1 + self.max_trade_loss)
            take_profit_price = self.current_price * (1 - self.max_trade_gain)
        return stop_loss_price, take_profit_price

    def create_order(self, is_buying, quantity, stop_loss_price, take_profit_price):
        
        order_type = self.client.SIDE_BUY if is_buying else self.client.SIDE_SELL
        return self.client.create_order(
            symbol=self.symbol,
            side=order_type,
            type=self.client.ORDER_TYPE_LIMIT,
            timeInForce=self.client.TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=self.current_price,
            stopPrice=stop_loss_price,
            icebergQty=take_profit_price
        )
    
    def buy_and_sell_today(self, prediction):
        
        self.get_balance()
        self.current_price()
        quantity = self.balance * self.risk

        if self.model_strat == 'day_price':
            is_buying = prediction == 1
        elif self.model_strat == 'min_max':
            if prediction == 0: # don't trade when 0 is predicted
                return
            is_buying = prediction == -1
        else:
            return

        stop_loss_price, take_profit_price = self.calculate_prices(prediction, is_buying)
        order = self.create_order(is_buying, quantity, stop_loss_price, take_profit_price)

        # Update balance
        self.get_balance()

    def backtesting(self, df, trading_period=480):
        
        ''' 
        Annahmen: Eine Prediction wird auf des Daten des vergangenen Tages getroffen. Ein Trade wird dann mit dem Open Price des nächsten Tages gesetzt. 
        Wenn die maximale Haltedauer erreicht wird, wird zum Close Price des Tages verkauft.
        '''
        
        # Set Parameters
        data = df
        max_gain = self.max_trade_gain
        max_loss = self.max_trade_loss
        max_duration = self.max_trade_duration

        # Add OHLC info if not present
        if not set(['Open', 'High', 'Low', 'Close']).issubset(data.columns):
            
            ohlc_data = pd.read_csv(f'./data/raw/{self.symbol}_raw.csv')
            # Convert Open time to datetime
            data['Open time'] = pd.to_datetime(data['Open time'])
            ohlc_data['Open time'] = pd.to_datetime(ohlc_data['Open time'])
            # Merge OHLC data with prediction data
            data = pd.merge(data, ohlc_data, how='left', on='Open time')
        
        open_trades = []
        completed_trades = []
        total_profit = 0
        testing_balance_base = 100000
        testing_balance = testing_balance_base

        # Loop over the data day by day
        for i in range(trading_period):
            # Check if there is a new trading opportunity
            prediction = data['class'][i]
            
            if prediction == 0:
                is_buying = None
            elif prediction == -1:
                is_buying = True
            elif prediction == 1:
                is_buying = False

            trade_amount = testing_balance * self.risk
            quantity = trade_amount / data['Open'][i]

            if is_buying is not None:
                if is_buying:
                    # Add the new buying trade to the list of open trades
                    open_trades.append({'start_day': i, 'start_price': data['Open'][i], 'quantity': quantity, 'trade_amount': trade_amount, 'is_buying': True})
                    testing_balance -= trade_amount
                else:
                    # Add the new selling trade to the list of open trades
                    open_trades.append({'start_day': i, 'start_price': data['Open'][i], 'quantity': quantity, 'trade_amount': trade_amount, 'is_buying': False})
                    testing_balance -= trade_amount

            # Loop over the open trades
            trades_to_remove = []
            for trade in open_trades:
                profit = 0
                # Check if the conditions for closing the trade are met
                if trade['is_buying']: # buying trade

                    if data['High'][i] >= trade['start_price'] * (1 + max_gain):
                        price = trade['start_price'] * (1 + max_gain)
                    elif data['Low'][i] <= trade['start_price'] * (1 - max_loss):
                        price = trade['start_price'] * (1 - max_loss)
                    elif i - trade['start_day'] >= max_duration:
                        price = data['Close'][i]
               
                else: # selling trade
                    
                    if data['Low'][i] <= trade['start_price'] * (1 - max_gain):
                        price = trade['start_price'] * (1 + max_gain)
                    elif data['High'][i] >= trade['start_price'] * (1 + max_loss):
                        price = trade['start_price'] * (1 - max_loss)
                    elif i - trade['start_day'] >= max_duration:
                        price = data['Close'][i]

                if 'price' in locals():
                    
                    profit = (price - trade['start_price']) * trade['quantity']
                    # Add the profit to the total profit and update the testing balance
                    total_profit += profit
                    testing_balance += profit + trade['trade_amount']
                    # Log completed trade
                    completed_trades.append({
                        'start_day': trade['start_day'],
                        'end_day': i,
                        'duration': i - trade['start_day'],
                        'profit': profit,
                        'is_buying': trade['is_buying'],
                        'trade_amount': trade['trade_amount'] 
                    })
                    # Remove the trade from the list of open trades
                    trades_to_remove.append(trade)
            
            for trade in trades_to_remove:
                open_trades.remove(trade)

        # Calculate Sharpe Ratio

        completed_trades_df = pd.DataFrame(completed_trades)
        # Calculate the returns of each trade
        completed_trades_df['returns'] = completed_trades_df['profit'] / completed_trades_df['trade_amount']
        # Assume a risk-free rate of 0
        risk_free_rate = 0
        # Calculate the excess returns
        excess_returns = completed_trades_df['returns'] - risk_free_rate
        # Calculate the Sharpe Ratio
        sharpe_ratio = excess_returns.mean() / excess_returns.std()

        # Prepare summary
        total_profit_percentage = testing_balance * 100 / testing_balance_base - 100
        winning_trades = len(completed_trades_df[completed_trades_df['profit'] > 0])
        losing_trades = len(completed_trades_df[completed_trades_df['profit'] < 0])
        total_trades = len(completed_trades_df)
        traded_days_percentage = total_trades / len(data)
        win_rate = winning_trades / total_trades


        summary_df = pd.DataFrame([{
            'total_profit_percentage': total_profit_percentage, 
            'total_profit': total_profit,
            'testing_balance': testing_balance,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'traded_days_percentage': traded_days_percentage,
            'win_rate': win_rate
        }])
        completed_trades_df.to_csv(f'./data/metrics/backtesting/{self.symbol}_completed_trades.csv', index=False)
        summary_df.to_csv(f'./data/metrics/backtesting/{self.symbol}_summary.csv', index=False)
        
        return total_profit, testing_balance, completed_trades
        

    def create_prediction_dfs(self, prediction_symbol='BTCUSDT'):
        
        directories = ['./data/proc', './data/raw']
        search_keys = ['min_max']
        models = ['lstm', 'cn2', 'rulefit']
        
        def get_files(directories, search_keys, prediction_symbol, model):
            
            file_dict = {}
            
            for file in glob.glob(os.path.join(directories[0], f'*{prediction_symbol}*{model}*.csv')):
                if os.path.isfile(file):
                    filename = os.path.basename(file)
                    filename = os.path.splitext(filename)[0]
                    if filename not in file_dict:
                        file_dict[filename] = []
                    file_dict[filename].append(directories[0])

            for search_key in search_keys:
                for file in glob.glob(os.path.join(directories[1], f'*{prediction_symbol}*{search_key}*.csv')):
                    if os.path.isfile(file):
                        filename = os.path.basename(file)
                        filename = os.path.splitext(filename)[0]
                        if filename not in file_dict:
                            file_dict[filename] = []
                        file_dict[filename].append(directories[1])
            
            return file_dict
        
        model_file_dicts = {}

        for model in models:
            model_file_dicts[model] = get_files(directories, search_keys, prediction_symbol, model)

        for model, file_dict in model_file_dicts.items():
            for file, path in file_dict.items():
                # get path as string
                path = path[0]
                
                # get optimal model params
                opt_files = glob.glob(f'./data/metrics/model_opt/{model}/*{file}*.csv')
                if model == 'cn2':
                    opt_files = glob.glob(f'./data/metrics/model_opt/{model}/CN2SDUnorderedLearner*{file}*.csv')
                opt_dict = {}
                if opt_files:
                    df = pd.read_csv(opt_files[0])
                    # Sort the DataFrame by accuracy
                    df = df.sort_values('accuracy', ascending=False)
                    opt_dict = df.iloc[0].to_dict()
                        
                if model == 'cn2':
                    cn2 = CN2Predictor()
                    cn2.prepare_data(symbol=file, input_path=path)
                    cn2.create_classifier(min_covered_examples=opt_dict['min_covered_examples'], beam_width=opt_dict['beam_width'], max_rule_length=opt_dict['max_rule_length'])
                    cn2.predict_df()
                elif model == 'lstm':
                    data = pd.read_csv(f'{path}/{file}.csv')
                    machine = LSTMPredictor(symbol=file, input_data=data)
                    machine.lstm_class(n_timesteps=opt_dict['timesteps'], epochs=opt_dict['epochs'], units=opt_dict['units'], optimizer=opt_dict['optimizer'])
                    machine.predict_df()
                elif model == 'rulefit':
                    rft = RuleFitModel(symbol=file, input_path=path)
                    rft.train_model(max_rule_length=opt_dict['max_rule_length'], n_trees=opt_dict['n_trees'])
                    rft.predict_df()
              


################################################################################################################################################################
###### EXAMPLES
################################################################################################################################################################

### Generate prediction dataframes

# trader = trader()
# trader.create_prediction_dfs()


### Backtesting for a certain CSV file

df = pd.read_csv('data/pred/BTCUSDT_min_max_phases_rulefit_predicted.csv')

trader = trader('BTCUSDT', risk_per_trade=0.01, trading_strat='min_max_day')
profit, balance, completed_trades = trader.backtesting(df)
print(completed_trades)
print(f'Total profit: {profit}')
print(f'Final balance: {balance}')




