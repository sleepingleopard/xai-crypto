import h2o
import numpy as np
import os
import pandas as pd
import shap
from h2o.estimators import H2ORuleFitEstimator
from h2o.utils.model_utils import reset_model_threshold
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



class RuleFitModel:
    
    def __init__(self, input_path="./data/raw", symbol=None):
        
        self.input_data_file = f"{input_path}/{symbol}.csv"
        self.symbol = symbol
        self.train = None
        self.test = None
        self.x = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.prepare_data()
        
    
    def prepare_data(self):
        
        data = pd.read_csv(self.input_data_file)

        # Check if 'Open time' column exists in the DataFrame
        if 'Open time' in data.columns:
            self.open_time = data['Open time']
            data = data.drop('Open time', axis=1)

        if 'class' in data.columns:
            X_data = data.drop('class', axis=1)
        self.x = x = X_data.columns.tolist() # feature names
        self.y = y = 'class' # target class

        h2o.init()
        df = h2o.import_file(path=self.input_data_file, col_types={"class": "categorical"})
        df[y] = df[y].asfactor()
        self.train, self.test = train, test = df.split_frame(ratios=[0.8], seed=1)

        print(train)

        self.train = train
        self.test = test
        self.X_train = train[x]
        self.X_test = test[x]
        self.y_train = train[y]
        self.y_test = test[y]

        return train, test, x, y
    
    def train_model(self, max_rule_length=10, n_trees=50):
        
        rfit = H2ORuleFitEstimator(max_rule_length=max_rule_length, n_trees=n_trees, seed=1)
        rfit.train(x=self.x, y=self.y, training_frame=self.train)
        print(rfit.rule_importance())
        print('Predictions:')
        print(rfit.predict(self.test))

        self.rfit = rfit
        
        return rfit
    
    def predict_df(self, pred_output_path='./data/pred', threshold=None):

        print("Threshold of RuleFit model is ", self.rfit.default_threshold())
        rfit_mod = self.rfit
        # Change the prediction threshold if needed
        if threshold is not None:
            reset_model_threshold(rfit_mod, threshold)
            print("Custom threshold of RuleFit model was set to ", threshold)
        
        predictions = rfit_mod.predict(self.test)  # predictions have columns: predict (class), p0 and p1 

        with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
            predictions_df = predictions.as_data_frame() 

        # Only take the class column
        predictions = predictions_df['predict']

        with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
            df = pd.DataFrame(self.test.as_data_frame())

        df['class'] = predictions
        df.to_csv(f"{pred_output_path}/{self.symbol}_rulefit_predicted.csv", index=False)

        return df
    
    def optimize(self):
        
        param_grid = {
            'max_rule_length_values': [3, 6, 10], # default = 3 
            'seed_values': [1],
            'n_trees': [50, 100, 150, 200] # default = 50; 
            # lambda not needed, because RuleFit calculates the default value for lambda using a heuristic based on the training data
        }
        
        # Load existing results
        results_file = f'./data/metrics/model_opt/rulefit/rulefit_grid_results_{self.symbol}.csv'
        try:
            results = pd.read_csv(results_file)
        except FileNotFoundError:
            results = pd.DataFrame(columns=['max_rule_length', 'seed', 'n_trees', 'auc_roc', 'accuracy', 'auc_pr', 'f1_score', 'precision', 'recall'])
        
        # Iterate over all combinations of parameters
        for max_rule_length in param_grid['max_rule_length_values']:
            for seed in param_grid['seed_values']:
                for n_trees in param_grid['n_trees']:
                    
                    if not results[(results['max_rule_length'] == max_rule_length) &
                           (results['seed'] == seed) &
                           (results['n_trees'] == n_trees)].empty:
                        print(f"Skipping existing combination: max_rule_length={max_rule_length}, seed={seed}, n_trees={n_trees}")
                        continue
                    
                    # Create and train a RuleFit model
                    rfit = H2ORuleFitEstimator(max_rule_length=max_rule_length, seed=seed, algorithm='AUTO', rule_generation_ntrees=n_trees)
                    rfit.train(x=self.x, y=self.y, training_frame=self.train)

                    # Get performance metrics on the validation set
                    # ... from H2O model performance
                    perf = rfit.model_performance(self.test)
                    auc_roc = float(perf.auc())
                    auc_pr = float(perf.aucpr())

                    # ... and manually from predictions
                    with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):    
                        predictions = rfit.predict(self.test).as_data_frame()
                    # Convert predictions to binary labels
                    predictions_binary = [1 if p > 0.5 else 0 for p in predictions['predict']]
                    # Get actual values
                    with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
                        actual = self.test['class'].as_data_frame()  
                    # Calculate metrics
                    accuracy = accuracy_score(actual, predictions_binary)
                    f1 = f1_score(actual, predictions_binary, average='weighted', zero_division=0)
                    precision = precision_score(actual, predictions_binary, average='weighted', zero_division=0)
                    recall = recall_score(actual, predictions_binary, average='weighted', zero_division=0)
                    
                    new_results = pd.Series({
                        'max_rule_length': max_rule_length,
                        'seed': seed,
                        'n_trees': n_trees,
                        'auc_roc': auc_roc,
                        'accuracy': accuracy,
                        'auc_pr': auc_pr,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall
                    })
                    results = results._append(new_results, ignore_index=True)
                    # Save the results
                    results = results.sort_values(by=['max_rule_length', 'seed', 'n_trees'])
                    results.to_csv(results_file, index=False)

    
    def feature_selection(self, output_path='./data/proc', selection_percentile=80, data_samples=10):
        
        assert isinstance(selection_percentile, int) and 0 <= selection_percentile <= 100, "selection_percentile must be an integer between 0 and 100"
        
        rfit = H2ORuleFitEstimator(max_rule_length=10, max_num_rules=100, seed=1)
        self.prepare_data()
        rfit.train(x=self.x, y=self.y, training_frame=self.train) 
        
        # Prepare the data for SHAP
        with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
            shap_train_data = self.train[self.x].as_data_frame().values
        background_data = shap.sample(shap_train_data, data_samples)

        # Helper function for using SHAP with H2O Rulefit
        # SHAP needs Numpy arrays as input, H2O RuleFit needs H2O frames 
        def predict_fn(x):

            # Convert np array to pandas dataframe and then to H2O frame
            x = pd.DataFrame(x, columns=self.x)
            x = h2o.H2OFrame(x)
            preds = rfit.predict(x)
            
            # Convert H2O frame back to Pandas dataframe (multi-threading)
            with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
                preds = preds.as_data_frame().iloc[:, 1:]
            print ("Predictions: ", preds)
            return preds.values.reshape(-1, 3) # 3 classes, 3 columns

        explainer = shap.KernelExplainer(predict_fn, background_data)
        shap_values = explainer.shap_values(background_data)
        feature_names = list(self.x)
        
        # Calculate importance treshold based on the selection percentile
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        importance_treshold = np.percentile(mean_shap_values, selection_percentile)
        
        # Get feature importances and save them
        feature_importances = list(zip(feature_names, mean_shap_values))

        metrics_output_path = './data/metrics/feature_importance'
        if not os.path.exists(metrics_output_path):
            os.makedirs(metrics_output_path)
        feature_importance_df = pd.DataFrame(feature_importances, columns=['Feature', 'Score'])
        feature_importance_df['MeanScore'] = feature_importance_df['Score'].apply(np.mean)
        feature_importance_df = feature_importance_df.sort_values(by='MeanScore', ascending=False)
        feature_importance_df.to_csv(f'{metrics_output_path}/{self.symbol}_rulefit_feature_importance.csv', index=False)
        
        # Select important features
        important_features = [(feature, score) for feature, score in feature_importances if np.mean(score) > importance_treshold]
        
        # Print the important features
        for feature, score in important_features:
            print(f"Feature: {feature}, Score: {np.mean(score)}")
        
        ### Plot results

        n_features = len(feature_names)
        plt.figure(figsize=(10, 5))
        colors = ['blue', 'pink', 'green']
        for i in range(n_features):
            for j in range(3):  # 3 probabilities for each feature
                # Create a bar for the j-th probability of the i-th feature
                # The left edge of the bar is the sum of the previous probabilities
                plt.barh(feature_names[i], mean_shap_values[i, j], left=np.sum(mean_shap_values[i, :j]), color=colors[j])
        # Create patches for the legend
        patches = [mpatches.Patch(color=colors[i], label=f'Class {i}') for i in range(3)]
        plt.legend(handles=patches)
        # Add labels and title
        plt.xlabel('Average Score')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        # Invert the y-axis to have the highest score at the top
        plt.gca().invert_yaxis()
        # Save plot
        plot_output_path = './data/metrics/feature_importance/plots'
        if not os.path.exists(plot_output_path):
            os.makedirs(plot_output_path)
        plt.savefig(f'{plot_output_path}/{self.symbol}_rulefit_shap_values.png')
        plt.clf()
        
        # Load the data from the input file
        input_file = self.input_data_file
        data = pd.read_csv(input_file)

        # Remove the unimportant features from the data, but keep 'phases' and 'class' column in every case
        if 'phases' in feature_names:
            important_feature_names = ['phases'] + [feature for feature, _ in important_features] + ['class']
        else:
            important_feature_names = [feature for feature, _ in important_features]
        
        data_selected = data[important_feature_names]

        # Add timestamps again as first column
        data_selected.insert(0, 'Open time', self.open_time)

        # Save the resulting data to the output file
        output_file = f'{output_path}/{self.symbol}_rulefit_selected.csv'
        data_selected.to_csv(output_file, index=False)

        return data_selected



################################################################################################################################################################
###### EXAMPLE USAGE ######
################################################################################################################################################################

### Single Functions

# rf = RuleFitModel(input_path="./data/raw", symbol="BTCUSDT_phases_min_max_ti")
# rf.train_model()

# rf.predict_df(threshold=0.333)
# rf.optimize()
# rf.feature_selection(data_samples=3, selection_percentile=90)
 
### Feature Selection

# data_files_short = ['BTCUSDT_min_max_phases', 'BTCUSDT_min_max']
# for data_file in data_files_short:
#     rf = RuleFitModel(input_path="./data/raw", symbol=data_file)
#     rf.feature_selection(data_samples=40, selection_percentile=40)

# data_files_long = ['BTCUSDT_min_max_ti', 'BTCUSDT_phases_min_max_ti_n', 'BTCUSDT_min_max_ti_n']
# for data_file in data_files_long:
#     rf = RuleFitModel(input_path="./data/raw", symbol=data_file)
#     rf.feature_selection(data_samples=40, selection_percentile=80)


### Model Optimization

# data_files_long = ['BTCUSDT_min_max_ti', 'BTCUSDT_phases_min_max_ti', 'BTCUSDT_min_max_ti_n', 'BTCUSDT_phases_min_max_ti_n'] 
# data_files_long_sel = ['BTCUSDT_min_max_ti_rulefit_selected', 'BTCUSDT_phases_min_max_ti_rulefit_selected', 'BTCUSDT_min_max_ti_n_rulefit_selected', 'BTCUSDT_phases_min_max_ti_n_rulefit_selected']
# data_files_short = ['BTCUSDT_min_max_phases', 'BTCUSDT_min_max']
# data_files_short_sel = ['BTCUSDT_min_max_phases_rulefit_selected', 'BTCUSDT_min_max_rulefit_selected']

# for data_file in data_files_long:
#     rf = RuleFitModel(input_path="./data/raw", symbol=data_file)
#     rf.optimize()
#     h2o.cluster().shutdown()
# for data_file in data_files_long_sel:
#     rf = RuleFitModel(input_path="./data/proc", symbol=data_file)
#     rf.optimize()
#     h2o.cluster().shutdown()
# for data_file in data_files_short:
#     rf = RuleFitModel(input_path="./data/raw", symbol=data_file)
#     rf.optimize()
#     h2o.cluster().shutdown()
# for data_file in data_files_short_sel:
#     rf = RuleFitModel(input_path="./data/proc", symbol=data_file)
#     rf.optimize()
#     h2o.cluster().shutdown()



