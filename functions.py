import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from joblib import Parallel, delayed

import pmdarima as pm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima_process import arma_generate_sample, ArmaProcess
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import statsmodels.api as sm

from scipy.stats import shapiro, spearmanr

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error




class ExogenousVariableBuilder:
    
    def __init__(self, df_y, df_exo, rates_config):
        self.df_y = df_y.copy()
        self.df_exo = df_exo.copy()
        self.rates_config = rates_config
        
    def build_exogenous_matrix (self):
        X = pd.DataFrame(index = self.df_y.index)
        
        for account, exog_dict in self.rates_config.items():
            for exog_variable, lags in exog_dict.items():
                if exog_variable in self.df_exo.columns:
                    # Add the base exogenous variable if lag 0 is present
                    if 0 in lags:
                        X[(account, exog_variable)] = self.df_exo[exog_variable]
                    
                    for lag in lags:
                        if lag > 0:
                            X[(account, f"{exog_variable}_lag_{lag}")] = self.df_exo[exog_variable].shift(lag)
                            
        X.columns = pd.MultiIndex.from_tuples(X.columns)
        return X







def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def display_error(y, y_predict, decimals=3, full=False):
    if full:
        print(f"Mean Squared Error: \t\t{mean_squared_error(y, y_predict)}")
        #print(f"Mean Absolute Error: \t\t{mean_absolute_error(y, y_predict)}")
        print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y, y_predict)}%")
        
    else:
        print(f"Mean Squared Error: \t\t{mean_squared_error(y, y_predict):.{decimals}f}")
        #print(f"Mean Absolute Error: \t\t{mean_absolute_error(y, y_predict):.{decimals}f}")
        print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y, y_predict):.3f}%")
        
        
def is_stationary_adf_test(series, display = True, significance_level = 0.05):
    result = adfuller(series)
    if result[1] < significance_level:
        if display:
            print("Augmented Dickey-Fuller: Series is Stationary")
        return True
    else:
        if display:
            print("Augmented Dickey-Fuller: Series is Non-Stationary")
        return False
    
    
    
def sarima_best_parameters(y : pd.DataFrame, X : pd.DataFrame, 
                           stationarity : dict, test_size : int = 12, title : str = "",
                           **kargs):
    
    if X is not None:
        y = y.dropna()
        X = X.dropna()

        common_index = y.index.intersection(X.index)
        y = y.loc[common_index]
        X = X.loc[common_index]
    
    train_size = len(y) - test_size
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    if X is not None:
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    else:
        X_train, X_test = None, None
        
    print(f"Train size: {len(y_train)}")
    
    best_models = {}
    
    for i, column in enumerate(y.columns):
        print(f"Finding best parameters for {column}")
        
        if X_train is not None:
            model = pm.auto_arima(y_train[column], X = X_train[column], 
                                    d=stationarity[column],
                                    trace = False, stepwise=True, suppress_warnings=True, 
                                    error_action="ignore", n_jobs=-1,
                                    **kargs
            )
        else:
            model = pm.auto_arima(y_train[column], 
                                    d=stationarity[column],
                                    trace = False, stepwise=True, suppress_warnings=True, 
                                    error_action="ignore", n_jobs=-1, 
                                    **kargs
            )
            
        best_models[column] = model
    
    print()
    for column in y.columns:
        model = best_models[column]
        print(f"The best model for {column} is {model.order}, {model.seasonal_order}")
    print()
     
    ##############################
    # Writing summary in a file
    
    if not os.path.exists("txt"):
        os.makedirs("txt")
    
    with open(f"txt/{title}.txt", "w") as f:
        f.write(f"Summary for {title}\n\n")
        for column in y.columns:
            model = best_models[column]
            f.write(f"The best model for {column} is {model.order}, {model.seasonal_order}\n\n")
            f.write(model.summary().as_text())
            f.write("\n\n")      
     
     
    ################################ 
    # Forecasting
    
    predictions = {}
    conf_intervals = {}

    for column, model in best_models.items():
        if X_test is not None:
            forecast, conf_int = model.predict(n_periods=len(y_test), X=X_test[column], return_conf_int=True, alpha=0.05)
        else:
            forecast, conf_int = model.predict(n_periods=len(y_test), return_conf_int=True, alpha=0.05)
            
        predictions[column] = forecast
        conf_intervals[column] = conf_int
        print(f"Predictions for {column}")
        display_error(y_test[column], forecast, decimals=5)

    df_predictions = pd.DataFrame(predictions)

    conf_dfs = []
    for column, conf in conf_intervals.items():
        conf_df = pd.DataFrame(conf, columns=[(column, "lower"), (column, "upper")])
        conf_df.index = y_test.index
        conf_dfs.append(conf_df)
        
    df_conf = pd.concat(conf_dfs, axis=1)
    df_conf.columns = pd.MultiIndex.from_tuples(df_conf.columns)

    ###############################


    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Share of Savings [%]")
    
    
    df_predictions.plot(ax = ax, label='Predictions')
    y.plot(ax = ax)

    for column in y.columns:
        lower = (column, 'lower')
        upper = (column, 'upper')
        ax.fill_between(df_conf.index, df_conf[lower], df_conf[upper], color='k', alpha=0.1)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())
    
    plt.savefig(os.path.join("imgs", f"{title}.png"))
    plt.show()
    
    
    ##############################
    
    # Plotting them independently    
    plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    for i, column in enumerate(y.columns):
        plt.subplot(3, 2, i+1)
        plt.title(column)
        plt.plot(y[column], label='True')
        plt.plot(predictions[column], label='Predictions')
        plt.fill_between(y_test.index, conf_intervals[column][:, 0], conf_intervals[column][:, 1], alpha=0.1)
        plt.ylabel("Share of Savings [%]")
        plt.xlabel("Date")
        plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join("imgs", f"{title}_multiple.png"))
    plt.show()
    
    return best_models










def sarima_chat_gpt(y: pd.DataFrame, X: pd.DataFrame, 
                    sarima_params: dict, test_size: int = 12, title: str = "",
                    **kargs):
    """
    Trains SARIMAX models with unique hyperparameters for each time series column.
    
    Parameters:
    - y : pd.DataFrame -> The dependent time series (accounts balances).
    - X : pd.DataFrame -> The exogenous variables.
    - sarima_params : dict -> Dictionary specifying SARIMA settings for each column.
        Example:
        sarima_params = {
            'CSL': {'d': 1, 'seasonal': True, 'm': 12, 'p_range': (0,3), 'q_range': (0,3)},
            'DAT': {'d': 2, 'seasonal': False, 'm': 1, 'p_range': (1,2), 'q_range': (1,2)},
        }
    - test_size : int -> Number of time steps for testing.
    - title : str -> Title for saving results.

    Returns:
    - best_models : dict -> Dictionary of fitted models.
    """
    
    if X is not None:
        y = y.dropna()
        X = X.dropna()

        # Ensure X and y have matching indices
        common_index = y.index.intersection(X.index)
        y = y.loc[common_index]
        X = X.loc[common_index]

        # Drop fully empty columns (i.e., when an account has no exogenous variables)
        X = X.dropna(axis=1, how='all')

        # Drop MultiIndex levels that became empty (if needed)
        if isinstance(X.columns, pd.MultiIndex):
            X.columns = X.columns.remove_unused_levels()
    
    train_size = len(y) - test_size
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    if X is not None:
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    else:
        X_train, X_test = None, None
        
    print(f"Train size: {len(y_train)}")
    
    best_models = {}
    
    for column in y.columns:
        print(f"Finding best parameters for {column}")
        
        # Get specific SARIMA settings for this column
        params = sarima_params.get(column, {})
        d_value = params.get('d', 1)
        seasonal = params.get('seasonal', True)
        m_value = params.get('m', 12)
        p_range = params.get('p_range', (0, 3))
        q_range = params.get('q_range', (0, 3))

        # Define p, q search space
        p_values = list(range(p_range[0], p_range[1] + 1))
        q_values = list(range(q_range[0], q_range[1] + 1))
        
        # Check if X_train has exogenous variables for this column
        if X_train is not None and column in X_train.columns.get_level_values(0).unique():
            exog_vars = X_train[column].dropna(axis=1, how='all')  # Remove empty exog variables
            if exog_vars.shape[1] > 0:
                model = pm.auto_arima(y_train[column], X=exog_vars, 
                                      d=d_value, seasonal=seasonal, m=m_value,
                                      #start_p=p_range[0], max_p=p_range[1],
                                      #start_q=q_range[0], max_q=q_range[1],
                                      trace=False, stepwise=True, suppress_warnings=True, 
                                      error_action="ignore", n_jobs=-1,
                                      **kargs)
            else:
                print(f"No exogenous variables for {column}, fitting without X")
                model = pm.auto_arima(y_train[column], 
                                      d=d_value, seasonal=seasonal, m=m_value,
                                      #start_p=p_range[0], max_p=p_range[1],
                                      #start_q=q_range[0], max_q=q_range[1],
                                      trace=False, stepwise=True, suppress_warnings=True, 
                                      error_action="ignore", n_jobs=-1, 
                                      **kargs)
        else:
            print(f"No exogenous variables for {column}, fitting without X")
            model = pm.auto_arima(y_train[column], 
                                  d=d_value, seasonal=seasonal, m=m_value,
                                  #start_p=p_range[0], max_p=p_range[1],
                                  #start_q=q_range[0], max_q=q_range[1],
                                  trace=False, stepwise=True, suppress_warnings=True, 
                                  error_action="ignore", n_jobs=-1, 
                                  **kargs)
        best_models[column] = model
    
    print()
    for column in y.columns:
        model = best_models[column]
        print(f"The best model for {column} is {model.order}, {model.seasonal_order}")
    print()
     
    ##############################
    # Writing summary in a file
    
    if not os.path.exists("txt"):
        os.makedirs("txt")
    
    with open(f"txt/{title}.txt", "w") as f:
        f.write(f"Summary for {title}\n\n")
        for column in y.columns:
            model = best_models[column]
            f.write(f"The best model for {column} is {model.order}, {model.seasonal_order}\n")
            f.write(model.summary().as_text())
            f.write("\n\n\n")      
     
    ################################ 
    # Forecasting
    
    predictions = {}
    conf_intervals = {}

    for column, model in best_models.items():
        if X_test is not None and column in X_test.columns.get_level_values(0).unique():
            exog_vars_test = X_test[column].dropna(axis=1, how='all')
            if exog_vars_test.shape[1] > 0:
                forecast, conf_int = model.predict(n_periods=len(y_test), X=exog_vars_test, return_conf_int=True, alpha=0.05)
            else:
                forecast, conf_int = model.predict(n_periods=len(y_test), return_conf_int=True, alpha=0.05)
        else:
            forecast, conf_int = model.predict(n_periods=len(y_test), return_conf_int=True, alpha=0.05)
            
        predictions[column] = forecast
        conf_intervals[column] = conf_int
        print(f"Predictions for {column}")
        display_error(y_test[column], forecast, decimals=5)

    df_predictions = pd.DataFrame(predictions)

    conf_dfs = []
    for column, conf in conf_intervals.items():
        conf_df = pd.DataFrame(conf, columns=[(column, "lower"), (column, "upper")])
        conf_df.index = y_test.index
        conf_dfs.append(conf_df)
        
    df_conf = pd.concat(conf_dfs, axis=1)
    df_conf.columns = pd.MultiIndex.from_tuples(df_conf.columns)

    ###############################

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Share of Savings [%]")
    
    df_predictions.plot(ax=ax, label='Predictions')
    y.plot(ax=ax)

    for column in y.columns:
        lower = (column, 'lower')
        upper = (column, 'upper')
        ax.fill_between(df_conf.index, df_conf[lower], df_conf[upper], color='k', alpha=0.1)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())
    
    plt.savefig(os.path.join("imgs", f"{title}.png"))
    plt.show()

    ##############################
    
    # Plotting them independently    
    plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    for i, column in enumerate(y.columns):
        plt.subplot(3, 2, i+1)
        plt.title(column)
        plt.plot(y[column], label='True')
        plt.plot(predictions[column], label='Predictions')
        plt.fill_between(y_test.index, conf_intervals[column][:, 0], conf_intervals[column][:, 1], alpha=0.1)
        plt.ylabel("Share of Savings [%]")
        plt.xlabel("Date")
        plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join("imgs", f"{title}_multiple.png"))
    plt.show()
    
    return best_models





