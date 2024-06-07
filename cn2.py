import numpy as np
import Orange
import os
import pandas as pd
import shap
from Orange.evaluation import AUC, CrossValidation
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



class CN2Predictor:
    def __init__(self):
        self.symbol = None
        self.input_path = None
        self.output_path = None
        self.labeled_data = None
        self.unlabeled_data = None
        self.classifier = None

    def prepare_data(self, symbol, input_path):
        
        self.symbol = symbol
        self.input_path = input_path

        # Helper function for df processing
        def df_preparation(df, is_labeled=True):
            
            # Fill NaN values with the median of the column
            df.fillna(df.median(), inplace=True)

            if is_labeled:
                
                # Take the first 80% of the data
                df = df.head(int(len(df) * 0.8))

                # Handle timestamps
                if 'Open time' in df.columns:
                    open_time = df['Open time']
                    df = df.drop('Open time', axis=1)

                # Create a new second row for 'continuous' and 'discrete'
                new_row1 = {col: 'continuous' for col in df.columns}
                new_row1[df.columns[-1]] = 'categorical'  # Assuming the class variable is the last column

                # Create a new third row for 'class'
                new_row2 = {col: '' for col in df.columns}
                new_row2[df.columns[-1]] = 'class'  # Assuming the class variable is the last column

                # Append the new rows at the top
                df = pd.concat([pd.DataFrame(new_row1, index=[0]), pd.DataFrame(new_row2, index=[1]), df])
            else:
                # Take the last 20% of the data
                df = df.tail(int(len(df) * 0.2))

                # Handle timestamps
                if 'Open time' in df.columns:
                    open_time = df['Open time']
                    df = df.drop('Open time', axis=1)

                # Drop the content of the class column
                df[df.columns[-1]] = ''  # Assuming the class variable is the last column

            return df, open_time
        
        ### Data Preparation

        df = pd.read_csv(f"{input_path}/{symbol}.csv")

        # Prepare the labeled data
        df_labeled, open_time_labeled = df_preparation(df.copy())
        labeled_path = f"./tmp/{symbol}_cn2.csv"
        df_labeled.to_csv(labeled_path, index=False) # Orange can't convert directly from DataFrame to Table
        self.labeled_data = Orange.data.Table(labeled_path)
        self.open_time_labeled = open_time_labeled
        os.remove(labeled_path)

        # Prepare the unlabeled data
        df_unlabeled, open_time_unlabeled = df_preparation(df.copy(), is_labeled=False)
        unlabeled_path = f"./tmp/{symbol}_cn2_unlabeled.csv"
        df_unlabeled.to_csv(unlabeled_path, index=False)
        self.unlabeled_data = Orange.data.Table(unlabeled_path)
        self.open_time_unlabeled = open_time_unlabeled
        os.remove(unlabeled_path)
        
        

    def create_classifier(self, beam_width=10, min_covered_examples=80, max_rule_length=6):
        
        # Create a CN2Learner
        learner = Orange.classification.rules.CN2SDUnorderedLearner()
        learner.rule_finder.search_algorithm.beam_width = beam_width
        learner.rule_finder.search_strategy.constrain_continous = True
        learner.rule_finder.search_algorithm.min_covered_examples = min_covered_examples
        learner.rule_finder.general_validator.max_rule_length = max_rule_length
        
        # Train the learner to create a classifier
        self.classifier = learner(self.labeled_data)

        return self.classifier

    def predict_df(self, pred_output_path='./data/pred'):
        
        # Use the classifier to predict the classes of the unlabeled data
        predicted_classes = self.classifier(self.unlabeled_data)
        # Save unlabeled data + predicted classes as CSV
        df = pd.DataFrame(data=self.unlabeled_data)
        df.columns = [var.name for var in self.labeled_data.domain.attributes] + ['class']
        df['class'] = predicted_classes
        # Add Open Time column
        open_time_unlabeled_df = pd.DataFrame(self.open_time_unlabeled, columns=['Open time']) 
        open_time_unlabeled_df= open_time_unlabeled_df['Open time'].astype(str)
        open_time_unlabeled_df = open_time_unlabeled_df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        df = pd.concat([open_time_unlabeled_df, df], axis=1)
        # Save pred df
        df.to_csv(f"{pred_output_path}/{self.symbol}_cn2_predicted.csv", index=False)

        return predicted_classes
    
    def get_feature_probs(self, instances):
        
        # Create a new domain with only the feature variables
        domain = Orange.data.Domain(self.labeled_data.domain.attributes)
        # Convert instances to Orange.data.Table
        instances = Orange.data.Table(domain, instances)
        # Get the probabilities of the positive class
        probs = self.classifier(instances, 1)

        return probs
    
    def optimize(self, param_grid):

        data = self.labeled_data
        
        models = [Orange.classification.rules.CN2Learner, Orange.classification.rules.CN2UnorderedLearner, Orange.classification.rules.CN2SDLearner, Orange.classification.rules.CN2SDUnorderedLearner]

        for model in models:
            output_file = f'./data/metrics/model_opt/cn2/{model.__name__}_grid_results_{self.symbol}.csv'
            try:
                results = pd.read_csv(output_file)
            except FileNotFoundError:
                results = pd.DataFrame(columns=['beam_width', 'min_covered_examples', 'max_rule_length', 'auc'])

            best_score = 0
            
            for beam_width in param_grid['beam_width']:
                for min_covered_examples in param_grid['min_covered_examples']:
                    for max_rule_length in param_grid['max_rule_length']:
                        
                        # Check if result is already obtained
                        if not results[(results['beam_width'] == beam_width) &
                           (results['min_covered_examples'] == min_covered_examples) &
                           (results['max_rule_length'] == max_rule_length)].empty:
                                print("Skipping existing parameters: beam_width={}, min_covered_examples={}, max_rule_length={}".format(beam_width, min_covered_examples, max_rule_length))
                                continue
                        
                        # Create and train the CN2 classifier
                        learner = model()
                        learner.rule_finder.search_algorithm.beam_width = beam_width
                        learner.rule_finder.search_strategy.constrain_continous = True
                        learner.rule_finder.search_algorithm.min_covered_examples = min_covered_examples
                        learner.rule_finder.general_validator.max_rule_length = max_rule_length

                        # Evaluate the classifier
                        # Start with cross validation
                        cross_val = CrossValidation(k=5)
                        cv_results = cross_val(data, [learner])
                        print("CV Results:", cv_results.actual, cv_results.predicted) # dev
                        print(len(cv_results.actual), len(cv_results.predicted))  # dev
                        auc_score = AUC(cv_results)

                        # Prepare the classification report
                        y_true = cv_results.actual
                        y_pred = cv_results.predicted[0]
                        
                        report = classification_report(y_true, y_pred, output_dict=True)
                        
                        # Get other metrics
                        accuracy = report['accuracy']
                        precision = report['weighted avg']['precision']
                        recall = report['weighted avg']['recall']
                        f1 = report['weighted avg']['f1-score']

                        # Document the results
                        new_entry = pd.Series({
                        'beam_width': beam_width,
                        'min_covered_examples': min_covered_examples,
                        'max_rule_length': max_rule_length,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': auc_score
                        })
                        results = results._append(new_entry, ignore_index=True)
                        results = results.sort_values(by=['beam_width', 'min_covered_examples', 'max_rule_length'])
                        results.to_csv(output_file, index=False)

            

def feature_selection(input_path='./data/raw', output_path='./data/proc', symbol=None, selection_percentile=80, data_samples=100):
    
    assert isinstance(selection_percentile, int) and 0 <= selection_percentile <= 100, "selection_percentile must be an integer between 0 and 100"
    
    cn2 = CN2Predictor()
    cn2.prepare_data(symbol, input_path, output_path)
    cn2.create_classifier()

    x = cn2.unlabeled_data.X
    shap_x = np.delete(x, -1, axis=1)
    feature_names= [var.name for var in cn2.labeled_data.domain.attributes]
    background_data = shap.sample(shap_x, data_samples)

    def predict_fn(data):
        
        data = np.array(data)
        preds = cn2.classifier.predict_proba(data)
        preds = pd.DataFrame(data=preds)

        return preds.values
    
    explainer = shap.KernelExplainer(predict_fn, background_data)
    shap_values = explainer.shap_values(background_data)
    
    # Convert shap_values to a list of arrays for plotting
    shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    # Plot and save the shap values
    shap.summary_plot(shap_values_list, background_data, feature_names, show=False)
    plot_output_path = './data/metrics/feature_importance/plots'
    if not os.path.exists(plot_output_path):
        os.makedirs(plot_output_path)
    plt.savefig(f'{plot_output_path}/{symbol}_cn2_shap_values.png')
    plt.clf()
    
    print("Shap values: ", shap_values)

    # Calculate importance treshold based on the selection percentile
    mean_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))
    importance_treshold = np.percentile(mean_shap_values, selection_percentile)
    
    # Select important features
    feature_importances = list(zip(feature_names, mean_shap_values))
    important_features = [(feature, score) for feature, score in feature_importances if score > importance_treshold]
    
    # Print the important features
    for feature, score in important_features:
        print(f"Feature: {feature}, Score: {score}")
    
    # Load the data from the input file
    input_file = f'{input_path}/{symbol}.csv'
    data = pd.read_csv(input_file)

    # Separate timestamps
    if 'Open time' in data.columns:
        open_time = data['Open time']
        data = data.drop('Open time', axis=1)

    # Remove the unimportant features from the data, but keep phases and class in every case
    important_feature_names = [feature for feature, _ in important_features] + ['class']
    if 'phase' in data.columns and 'phase' not in important_feature_names:
        important_feature_names.insert(0, 'phase')
    
    data_selected = data[important_feature_names]

    # Add timestamps again as first column
    data_selected.insert(0, 'Open time', open_time)

    # Save the resulting data to the output file
    output_file = f'{output_path}/{symbol}_cn2_selected.csv'
    data_selected.to_csv(output_file, index=False)

    # Save feature importances as CSV
    metrics_output_path = './data/metrics/feature_importance'
    if not os.path.exists(metrics_output_path):
        os.makedirs(metrics_output_path)
    feature_importances = pd.DataFrame(data=feature_importances, columns=['Feature', 'Importance'])
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    feature_importances.to_csv(f'{metrics_output_path}/{symbol}_cn2_feature_importances.csv', index=False)

    return data_selected
    


################################################################################################################################################################
###### EXAMPLE USAGE ######
################################################################################################################################################################

### Single Functions

# input_path = './data/raw'
# symbol = 'BTCUSDT_phases_min_max_ti'
# param_grid = {
#     'beam_width': [5, 10, 15], 
#     'min_covered_examples': [40, 80, 120], 
#     'max_rule_length': [3, 6, 10] 
# }

# cn2 = CN2Predictor()
# cn2.prepare_data(symbol, input_path)
# cn2.create_classifier()
# cn2.predict_df()

# cn2.optimize(param_grid)

### Feature Selection

# data_files_long = ['BTCUSDT_min_max_ti', 'BTCUSDT_phases_min_max_ti_n', 'BTCUSDT_min_max_ti_n']
# data_files_short = ['BTCUSDT_min_max_phases', 'BTCUSDT_min_max']
# for symbol in data_files_long:
#     feature_selection(input_path, output_path, symbol, selection_percentile=80, data_samples=100)
# for symbol in data_files_short:
#     feature_selection(input_path, output_path, symbol, selection_percentile=40, data_samples=100)

### Model Optimization with Multiprocessing

# data_files_long = ['BTCUSDT_phases_min_max_ti', 'BTCUSDT_min_max_ti', 'BTCUSDT_phases_min_max_ti_n', 'BTCUSDT_min_max_ti_n']
# data_files_long_sel = ['BTCUSDT_phases_min_max_ti_cn2_selected', 'BTCUSDT_min_max_ti_cn2_selected', 'BTCUSDT_phases_min_max_ti_n_cn2_selected', 'BTCUSDT_min_max_ti_n_cn2_selected']
# data_files_short = ['BTCUSDT_min_max_phases', 'BTCUSDT_min_max']
# data_files_short_sel = ['BTCUSDT_min_max_phases_cn2_selected', 'BTCUSDT_min_max_cn2_selected']

# def process_file(file, path, param_grid):
#     cn2 = CN2Predictor()
#     cn2.prepare_data(file, path)
#     cn2.create_classifier()
#     cn2.optimize(param_grid)

# if __name__ == '__main__':
#     with multiprocessing.Pool() as pool:
#         for file in data_files_long:
#             pool.apply_async(process_file, (file, './data/raw', param_grid))
#         for file in data_files_short:
#             pool.apply_async(process_file, (file, './data/raw', param_grid))
#         for file in data_files_long_sel:
#             pool.apply_async(process_file, (file, './data/proc', param_grid))
#         for file in data_files_short_sel:
#             pool.apply_async(process_file, (file, './data/proc', param_grid))
#         pool.close()
#         pool.join()