import numpy as np
import pandas as pd
from hmmlearn import hmm
from hmmlearn.base import ConvergenceMonitor
from sklearn.preprocessing import StandardScaler

class MarketPhasePredictor:
  def __init__(self, input_path="./data/raw", output_path="./data/raw", n_components=1, covariance_type="diag", n_iter=100, symbol=None):
    
    self.model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)
    self.scaler = StandardScaler()
    self.symbol = symbol 
    self.file_path = f"{input_path}/{self.symbol}.csv" 
    self.output_file_path = f"{output_path}/{self.symbol}_phases.csv"

  def train(self):

    df = pd.read_csv(self.file_path)
    # Drop timestamps to not influence model training
    df = df.drop('Open time', axis=1)
    data = df.values

    # fill NaN values with the mean of the column
    data = np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=0), data)
    
    # Scale data
    data = self.scaler.fit_transform(data)

    # Train the HMM
    self.model.fit(data)

  def predict(self):
    
    # Load data
    df = pd.read_csv(self.file_path)
    data = df.values

    # fill NaN values with the mean of the column
    data = np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=0), data)
    
    # Scale data
    data = self.scaler.transform(data)

    # Predict the optimal sequence of internal hidden state
    hidden_states = self.model.predict(data)

    print("Means and variances of hidden states:")
    for i in range(self.model.n_components):
        print(f"Hidden state {i+1}: mean = {round(self.scaler.inverse_transform([self.model.means_[i]])[0][0], 2)}, variance = {round(np.diag(self.scaler.inverse_transform(self.model.covars_[i]))[0], 2)}")

    return hidden_states
  
  def phase_recognition(self):
    
    df = pd.read_csv(self.file_path)
    
    # Store timestamps separately to not influence model training
    self.open_time = df['Open time']
    df = df.drop('Open time', axis=1)
    
    data = df.values

    # fill NaN values with the mean of the column
    data = np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=0), data)

    # Scale data
    data = self.scaler.transform(data)

    # Predict the optimal sequence of internal hidden state
    hidden_states = self.model.predict(data)

    # Add the hidden states as a new column 'phase' to the DataFrame
    df.insert(len(df.columns), 'phase', hidden_states + 1)  # Adding 1 to make phases 1-6 instead of 0-5
    # Add timestamps again as first column
    df.insert(0, 'Open time', self.open_time)

    # Save the DataFrame to the output folder
    df.to_csv(self.output_file_path, index=False)
  

class ThresholdMonitor(ConvergenceMonitor):
    @property
    def converged(self):
        # Check if the maximum number of iterations has been reached
        if self.iter == self.n_iter:
            return True

        # Check if the log likelihood has improved
        if len(self.history) < 2:
            return False
        return abs(self.history[-1] - self.history[-2]) < self.tol

def optimize(data, n_components_range, n_iter_range, covariance_types, tol=5):
  
  best_n_components = None
  best_n_iter = None
  best_covariance_type = None

  # Load existing results from the CSV file into a DataFrame
  result_file = './data/metrics/model_opt/hmm/hmm_optimization_results.csv'
  try:
      results = pd.read_csv(result_file)
  except FileNotFoundError:
      results = pd.DataFrame(columns=['n_components', 'n_iter', 'covariance_type', 'bic', 'error'])

  # Test all combinations of n_components, n_iter, and covariance_type
  for n_components in n_components_range:
    for n_iter in n_iter_range:
      for covariance_type in covariance_types:
        # Check if these parameters are already in the results
        if not results[(results['n_components'] == n_components) & (results['n_iter'] == n_iter) & (results['covariance_type'] == covariance_type)].empty:
            continue
        
        # Create and train the HMM
        model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, tol=tol)
        model.monitor_ = ThresholdMonitor(model.monitor_.tol, model.monitor_.n_iter, model.monitor_.verbose)
        model.fit(data)

        # verbose monitor output for every model
        print(model.monitor_)
        print(model.monitor_.converged)
        
        # Check if the model has converged
        if not model.monitor_.converged:
            # If the model did not converge, add a row to the results indicating this
            results = results._append({'n_components': n_components, 'n_iter': n_iter, 'covariance_type': covariance_type, 'bic': np.nan, 'error': "not converging"}, ignore_index=True)
            continue

        # Calculate the BIC
        log_likelihood = model.score(data)
        n_features = data.shape[1]
        n_params = n_components ** 2 + n_components * n_features * 2
        bic = np.log(len(data)) * n_params - 2 * log_likelihood

        # Add the results to the DataFrame
        print(f"n_components = {n_components}, n_iter = {n_iter}, covariance_type = {covariance_type}, BIC = {bic}")
        results = results._append({'n_components': n_components, 'n_iter': n_iter, 'covariance_type': covariance_type, 'bic': bic}, ignore_index=True)       


  # Save the results to a CSV file
  results.to_csv(result_file, index=False)

  # Find the index of the row with the lowest BIC
  best_index = results['bic'].idxmin()

  # Get the best parameters from the row with the lowest BIC
  best_n_components = results.loc[best_index, 'n_components']
  best_n_iter = results.loc[best_index, 'n_iter']
  best_covariance_type = results.loc[best_index, 'covariance_type']

  # Print the best parameters
  print(f"Best parameters: n_components = {best_n_components}, n_iter = {best_n_iter}, covariance_type = {best_covariance_type}")

  return best_n_components, best_n_iter, best_covariance_type


################################################################################################################################################################
###### EXAMPLE USAGE ######
################################################################################################################################################################

### Model Optimization

# n_components_range = range(2, 10)   
# n_iter_range = range(10, 1300, 100) 
# covariance_types = ['spherical', 'tied', 'diag', 'full']
# file_path = './data/raw/BTCUSDT.csv'
# df = pd.read_csv(file_path)
# data = df.values
# data = np.where(np.isnan(data), np.ma.array(data, mask=np.isnan(data)).mean(axis=0), data)
# scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)

# optimize(data, n_components_range, n_iter_range, covariance_types)

'''
Erklärung des Outputs:
Der Convergence Monitor prüft, ob die absolute Distanz zwischen den Log-Likelihoods der letzten beiden Iterationen kleiner als die Toleranz ist. Wenn dies der Fall ist, wird converged auf True gesetzt.
HMM-Bibliothek vergleicht nach jeder Iteration die Log-Likelihood. Wenn die von Iteration n kleiner ist als die von Iteration n-1, wird folgender Fehler ausgegeben: Model is not converging.  "Current: 79121.67405873556 (log-p von n) is not greater than 82086.32626690302 (log-p von n-1). Delta is -2964.652208167463"
Das Modelltraining wird allerdings nicht abgebrochen, sondern fortgeführt bis es konvergiert. Wenn letztendlich bei einer Iteration der Abstand zwischen n-1 und n kleiner als tol ist, ist das Training abgeschlossen. 
'''