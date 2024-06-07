import numpy as np
import pandas as pd
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib import pyplot as plt



class LSTMPredictor():
    
    def __init__(self, symbol, input_data, timesteps=60, training_size=0.8):
        
        self.symbol = symbol
        self.input_data = input_data
        self.timesteps = timesteps
        self.training_size = training_size

    ################################################################################################################################################################
    ###### MODEL CREATION
    ################################################################################################################################################################

    def prepare_data(self, data=None, timesteps=None):

        if data is None:
          data = self.input_data
        if timesteps is None:
          timesteps = self.timesteps
        
        timesteps = self.timesteps

        # Remove open times if existent
        if 'Open time' in data.columns:
          data = data.drop('Open time', axis=1)
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1)) # Scale the data between 0 and 1
        data = scaler.fit_transform(data)
        print('Data shape: ' + str(data.shape))

        '''
        Each data point in the input data array is not a single value, but a vector of features. So, each of these sequences of 60 data points is actually a 2D array with shape (60, number_of_features).
        '''

        # Prepare the data for LSTM
        X, y = [], []
        for i in range(timesteps, len(data)):    # for every index: get 60 data points before
          X.append(data[i-timesteps:i, :-1])      # -1: Exclude the last column, which is the class
          y.append(data[i, -1])  # Class is at last row and that's what we want to predict
        X, y = np.array(X), np.array(y)
        
        print('X.shape, y.shape: ' + str(X.shape) + ', ' + str(y.shape))

        '''
        The resulting numpy array for X is three-dimensional:
        
        X.shape[0]: This is the number of samples. Each sample is a sequence of 60 data points, so the number of samples is the total number of such sequences.

        X.shape[1]: This is the number of time steps. Each time step is a data point in the sequence, and since each sequence has 60 data points, the number of time steps is 60.

        X.shape[2]: This is the number of features per step. Each data point has a certain number of features (for example, the open price, close price, volume, etc. in a stock price dataset), so this is the number of features in each data point.
        
        The resulting array for Y is one-dimensional, with each element being the class label for the corresponding sequence in X.
        '''

        # check for number of target classes and convert y adequately
        num_classes = len(np.unique(y))
        y = to_categorical(y, num_classes)
        
        # split the data set
        train_size = int(len(X) * self.training_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print('X_train.shape, X_test.shape, y_train.shape, y_test.shape: ' + str(X_train.shape) + str(X_test.shape) + str(y_train.shape) + str(y_test.shape))
        print('Number of classes: ' + str(num_classes))
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = num_classes
        
        return X_train, X_test, y_train, y_test, num_classes

    def lstm_class(self, data=None, n_timesteps=None, units=50, optimizer='adam', epochs=1):

        if data is None:
          data = self.input_data
        if n_timesteps is None:
          n_timesteps = self.timesteps

        # Remove open times if existent
        if 'Open time' in data.columns:
          data = data.drop('Open time', axis=1)

        X_train, X_test, y_train, y_test, num_classes = self.prepare_data(data, n_timesteps)

        # Define the LSTM model for classification
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(LSTM(units=units))
        model.add(Dense(num_classes, activation='softmax'))  

        # choose loss function according to the number of classes
        if num_classes == 2:
          loss_function = 'binary_crossentropy'
        else:
          loss_function = 'categorical_crossentropy'
        
        # Compile and train the model
        model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=2)

        model.save('./models/lstm_class.keras')

        print('Training Accuracy: ', model.history.history['accuracy'][-1])


        # Evaluate the model on the test data
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        print('Test Accuracy: ', accuracy)
        print('Test Loss: ', loss)

        self.model = model

        return model, X_train, X_test, y_train, y_test, num_classes

    ################################################################################################################################################################
    ###### CLASS PREDICTION
    ################################################################################################################################################################

    def class_prediction(self, threshold=None):
      
        class_probs = self.model.predict(self.X_test)
        # Apply threshold and get the class with the highest probability
        if threshold is not None:
          class_pred = [np.argmax(probs) if np.max(probs) > threshold else 0 for probs in class_probs]
        else:
          class_pred = np.argmax(class_probs, axis=1)

        # Translate indices to classes
        class_pred = [pred - 1 for pred in class_pred]
        
        print('Class probabilities: ' + str(class_probs))
        print('Class predictions: ' + str(class_pred))

        return class_probs, class_pred

    def predict_df(self, pred_output_path='./data/pred', threshold=None): 
      
        _, predictions = self.class_prediction(threshold=threshold)
        input_data = self.input_data.iloc[self.timesteps:].copy()     # remove the first 60 data points, they will not be processed in prepare data function because of the timestep window
        df = input_data.iloc[int(self.training_size * len(input_data)):].copy()
        assert len(predictions) == len(df)
        df['class'] = predictions
        df.to_csv(f"{pred_output_path}/{self.symbol}_lstm_predicted.csv", index=False)

        return df

    ################################################################################################################################################################
    ###### OPTIMIZATION 
    ################################################################################################################################################################

    # Wrap model definition in a function for optimization
    def create_model(self, n_timesteps, n_features, num_classes, units, optimizer='adam'):
      
        if num_classes == 2:
          loss_function = 'binary_crossentropy'
        else:
          loss_function = 'categorical_crossentropy'
        
        model = Sequential()
        model.add(Input(shape=(n_timesteps, n_features)))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(LSTM(units=units))
        model.add(Dense(num_classes, activation='softmax'))  
        model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        
        return model


    def optimize(self, input_data_path, symbol=None):

        if symbol is None:
            symbol = self.symbol
        filename = f'{input_data_path}/{symbol}.csv'
        data = pd.read_csv(filename)

        # Remove open times if existent
        if 'Open time' in data.columns:
          data = data.drop('Open time', axis=1)

        '''
        Define the grid of parameters to search:
        Units: LSTM-Zellen in jeder Schicht. Eine größere Anzahl von Einheiten kann komplexere Muster erfassen, kann aber auch zu Overfitting führen.
        Epochs: Anzahl der Iterationen über den gesamten Datensatz.
        Batch Size: Anzahl der Beispiele, die gleichzeitig in einer Iteration verarbeitet werden. Eine größere Batch-Größe kann das Training beschleunigen, kann aber auch mehr Speicher benötigen und zu schlechteren Ergebnissen führen.
        Timesteps: Anzahl der Zeitpunkte in jedem Eingabedatensatz. Dies ist die Anzahl der Datenpunkte, die in jedem Schritt des Modells betrachtet werden.
        '''
        param_grid = {
            'units': [50, 100, 150],  # 50, 100, 150
            'optimizer': ['adam', 'rmsprop', 'sgd'], # 'adam', 'rmsprop', 'sgd'
            'epochs': [1, 3, 5], # 1, 5
            'n_timesteps': [30, 60, 90], # 60, 90
            'batch_size': [1],
            'diff_order': [1, 2, 3] # 1, 2, 3
        }

        results_file = f'./data/metrics/model_opt/lstm/lstm_grid_results_{symbol}.csv'
        try:
          results = pd.read_csv(results_file)
        except:
          results = pd.DataFrame(columns=['units', 'optimizer', 'epochs', 'timesteps', 'diff_order', 'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'])
      
        
        # Iterate over all combinations of hyperparameters
        for n_timesteps in param_grid['n_timesteps']:
            for units in param_grid['units']:
                for optimizer in param_grid['optimizer']:
                    for epochs in param_grid['epochs']:
                        for batch_size in param_grid['batch_size']:
                            for diff_order in param_grid['diff_order']:
                          
                                # Check if this combination already exists in the results
                                if ((results['optimizer'] == optimizer) & 
                                    (results['units'] == units) & 
                                    (results['epochs'] == epochs) & 
                                    (results['timesteps'] == n_timesteps) & 
                                    (results['diff_order'] == diff_order)).any():
                                    print(f'Combination already exists: {units}, {optimizer}, {epochs}, {n_timesteps}, {diff_order}. Skipping...')
                                    continue
                                
                                # Differentiation on data
                                diff_data = data.diff(diff_order).dropna()
                                
                                # Prepare the data
                                X_train, X_test, y_train, y_test, num_classes = self.prepare_data(diff_data, n_timesteps)

                                # Create and train the model with the current hyperparameters
                                n_features = X_train.shape[2]
                                model = self.create_model(n_timesteps, n_features, num_classes, units, optimizer)
                                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
                                
                                # Evaluate the model on the test data
                                y_pred = model.predict(X_test)
                                y_pred = np.argmax(y_pred, axis=1)
                                if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                                  y_test = np.argmax(y_test, axis=1)
                                # beide Datensätze müssen mit argmax in eine konkrete Klasse umgewandelt werden, da sie vorher die verschiedenen Klassen und ihre Wahrscheinlichkeiten enthalten
                                
                                # Calculate metrics for evaluation:
                                accuracy = accuracy_score(y_test, y_pred)
                                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                                recall = recall_score(y_test, y_pred, average='weighted')
                                f1 = f1_score(y_test, y_pred, average='weighted')

                                # Calculate AUC-ROC if binary or multilabel classification
                                if num_classes == 2 or y_test.ndim > 1:
                                    y_test_bin = label_binarize(y_test, classes=[0, 1, 2]) # binary matrix is expected
                                    auc_roc = roc_auc_score(y_test_bin, model.predict(X_test), multi_class='ovr')
                                else:
                                    auc_roc = None

                                # Append the results with Series to keep the order of the columns
                                new_entry = pd.Series({
                                    'units': units, 
                                    'optimizer': optimizer, 
                                    'epochs': epochs, 
                                    'timesteps': n_timesteps, 
                                    'diff_order': diff_order, 
                                    'accuracy': accuracy, 
                                    'precision': precision, 
                                    'recall': recall, 
                                    'f1_score': f1, 
                                    'auc_roc': auc_roc
                                })                          
                                results = results._append(new_entry, ignore_index=True)
                                results = results.sort_values(by=['units', 'optimizer', 'epochs', 'timesteps', 'diff_order'])
                                results.to_csv(results_file, index=False)

       

    ################################################################################################################################################################
    ###### FEATURE IMPORTANCE
    ################################################################################################################################################################

    def integrated_gradients(self, model, input_data, baseline=None, num_steps=100, plot=True, save_path="./data/metrics/feature_importance/plots", symbol=None):
      
        # If no baseline is provided, use a zero input
        if baseline is None:
            baseline = np.zeros(input_data.shape)
        
        # Ensure that the baseline and the input have the same shape
        assert baseline.shape == input_data.shape

        # Create a series of linearly interpolated inputs between the baseline and the actual input
        interpolated_inputs = [baseline + (input_data - baseline) * i / num_steps for i in range(num_steps + 1)]

        # Precompute the gradients of the model output with respect to each interpolated input
        gradients = []
        for input in interpolated_inputs:
            input_tensor = tf.convert_to_tensor(input)
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                prediction = model(input_tensor)
            gradient = tape.gradient(prediction, input_tensor)
            gradients.append(gradient.numpy())

        # Compute the average of the gradients
        avg_gradients = np.average(gradients, axis=0)

        # Compute the integrated gradients by subtracting the baseline from the input and multiplying by the average gradients
        integrated_gradients = (input_data - baseline) * avg_gradients

        # Plot the integrated gradients if requested
        avg_integrated_gradients = np.mean(integrated_gradients, axis=(0, 1))
        if plot:
          positions = range(len(avg_integrated_gradients))
          labels = range(1, len(avg_integrated_gradients) + 1)
          
          plt.figure(figsize=(10, 5))
          plt.bar(positions, avg_integrated_gradients)
          plt.xlabel('Feature')
          plt.ylabel('Importance')
          plt.title('Feature importances via Integrated Gradients')
          plt.xticks(positions, labels)
          # Hide every third label
          for label in plt.gca().xaxis.get_ticklabels():
            if (int(label.get_text())) % 5 != 0:
              label.set_visible(False)
          # Save plot
          if not os.path.exists(save_path):
            os.makedirs(save_path)
          plt.savefig(f'{save_path}/{symbol}_lstm_feature_importances.png')
          plt.clf()

        return integrated_gradients


    def feature_selection(self, symbol, input_path='./data/raw', output_path='./data/proc', top_feature_percentage=0.8):
      
        df = pd.read_csv(f'{input_path}/{symbol}.csv')
        
        # Store timestamps separately to not influence model training
        if 'Open time' in df.columns:
          open_time = df['Open time']
          df = df.drop('Open time', axis=1)

        # Keep phases in every case
        phases = None
        if 'phase' in df.columns:
          phases = df['phase']

        model, X_train, _, _, _, _ = self.lstm_class(data=df)

        # Compute the integrated gradients for the input data
        integrated_grads = self.integrated_gradients(model, X_train, symbol=symbol)

        # Compute the absolute sum of the integrated gradients for each feature
        feature_importances = np.sum(np.abs(integrated_grads), axis=(0, 1))

        # Sort the feature importances in descending order and compute the cumulative sum
        sorted_importances = np.sort(feature_importances)[::-1]
        cumulative_importances = np.cumsum(sorted_importances)

        # Normalize the cumulative sum to get the cumulative proportions
        cumulative_proportions = cumulative_importances / np.sum(sorted_importances)

        # Get the number of top features needed to reach the desired cumulative proportion
        num_top_features = np.argmax(cumulative_proportions >= top_feature_percentage) + 1

        # Ensure num_top_features does not exceed the number of available features
        num_top_features = min(num_top_features, len(feature_importances))

        # Get the indices of the features with the highest importance
        top_feature_indices = np.argsort(feature_importances)[-num_top_features:]    
        top_feature_indices = np.sort(top_feature_indices)

        # Create a new DataFrame that only contains the top features
        x_df = df.drop(columns=['class'])

        top_features_df = x_df.iloc[:, top_feature_indices].copy()  

        # Add the class column to the DataFrame
        top_features_df.loc[:, 'class'] = df['class']

        # Add timestamps again as first column
        top_features_df.insert(0, 'Open time', open_time)

        # Add phases again as second last column
        if phases is not None and 'phase' not in top_features_df.columns:
          top_features_df.insert(len(top_features_df.columns)-1, 'phase', phases)

        # Save the DataFrame to a CSV file
        top_features_df.to_csv(f'{output_path}/{symbol}_lstm_selected.csv', index=False)

        # Save feature importances to a CSV file
        metrics_output_path = './data/metrics/feature_importance'
        if not os.path.exists(metrics_output_path):
            os.makedirs(metrics_output_path)
        feature_importances_df = pd.DataFrame({'feature': x_df.columns, 'importance': feature_importances})
        feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
        feature_importances_df.to_csv(f'{metrics_output_path}/{symbol}_lstm_feature_importances.csv', index=False)

        return top_features_df

  


################################################################################################################################################################
###### EXAMPLE USAGE
################################################################################################################################################################

### Single functions

# data_file = './data/raw/BTCUSDT_phases_min_max_ti.csv'
# data = pd.read_csv(data_file)
# machine = LSTMPredictor(symbol='BTCUSDT', input_data=data)
#machine.lstm_class()
#machine.predict_df(threshold=0.5)
#machine.feature_selection('BTCUSDT_phases_min_max_ti')
#machine.optimize('./data/raw', 'BTCUSDT_phases_min_max_ti')

### Feature Selection

# data_files_long = ['BTCUSDT_min_max_ti', 'BTCUSDT_phases_min_max_ti_n', 'BTCUSDT_min_max_ti_n']
# data_files_short = ['BTCUSDT_min_max_phases', 'BTCUSDT_min_max']
# for file in data_files_long:
#    machine.feature_selection(file, top_feature_percentage=0.2)
# for file in data_files_short:
#    machine.feature_selection(file, top_feature_percentage=0.6)	


### Model Optimization

# data_files_long = ['BTCUSDT_min_max_ti', 'BTCUSDT_phases_min_max_ti', 'BTCUSDT_phases_min_max_ti_n', 'BTCUSDT_min_max_ti_n']
# data_files_long_sel = ['BTCUSDT_min_max_ti_lstm_selected', 'BTCUSDT_phases_min_max_ti_n_lstm_selected', 'BTCUSDT_min_max_ti_n_lstm_selected', 'BTCUSDT_phases_min_max_ti_lstm_selected']
# data_files_short = ['BTCUSDT_min_max_phases', 'BTCUSDT_min_max']
# data_files_short_sel = ['BTCUSDT_min_max_phases_lstm_selected', 'BTCUSDT_min_max_lstm_selected']

# data_files = [(data_files_long, './data/raw'), 
#         (data_files_short, './data/raw'), 
#         (data_files_long_sel, './data/proc'), 
#         (data_files_short_sel, './data/proc')]

# for file_list, path in data_files:
#   for file in file_list:
#     machine.optimize(path, file)

