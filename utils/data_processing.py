import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

def load_demand_data(file_path, start_date=None, end_date=None):
    """
    Load demand data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: DataFrame with demand data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
    
    # Filter by date range if provided
    if start_date is not None:
        data = data[data.index >= start_date]
    if end_date is not None:
        data = data[data.index <= end_date]
    
    return data

def preprocess_demand_data(data, normalize=True, add_time_features=True, 
                           add_lags=False, n_lags=24):
    """
    Preprocess demand data for modeling.
    
    Args:
        data (pd.DataFrame): DataFrame with demand data
        normalize (bool): Whether to normalize the data
        add_time_features (bool): Whether to add time features
        add_lags (bool): Whether to add lagged values
        n_lags (int): Number of lags to add
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Add time features
    if add_time_features and isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Add cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Add lagged values
    if add_lags and 'demand' in df.columns:
        for i in range(1, n_lags + 1):
            df[f'demand_lag_{i}'] = df['demand'].shift(i)
        
        # Drop rows with NaN values from lagging
        df.dropna(inplace=True)
    
    # Normalize data
    if normalize:
        if 'demand' in df.columns:
            scaler = MinMaxScaler()
            df['demand'] = scaler.fit_transform(df[['demand']])
            
            # Store scaler for inverse transformation
            df.attrs['demand_scaler'] = scaler
    
    return df

def create_sequences(data, target_col='demand', seq_length=24, forecast_horizon=24):
    """
    Create sequences for time series forecasting.
    
    Args:
        data (pd.DataFrame): DataFrame with time series data
        target_col (str): Name of the target column
        seq_length (int): Length of input sequences
        forecast_horizon (int): Number of steps to forecast
        
    Returns:
        tuple: (X, y) where X is input sequences and y is target sequences
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        # Input sequence
        X.append(data.iloc[i:i+seq_length].values)
        
        # Target sequence
        if target_col in data.columns:
            y.append(data[target_col].iloc[i+seq_length:i+seq_length+forecast_horizon].values)
        else:
            y.append(data.iloc[i+seq_length:i+seq_length+forecast_horizon].values)
    
    return np.array(X), np.array(y)

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X (np.ndarray): Input data
        y (np.ndarray): Target data
        test_size (float): Fraction of data for testing
        val_size (float): Fraction of data for validation
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Second split: train and val
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, shuffle=False
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train_val, None, X_test, y_train_val, None, y_test

def calculate_energy_mix(generation_data):
    """
    Calculate energy mix percentages from generation data.
    
    Args:
        generation_data (pd.DataFrame): DataFrame with generation data by source
        
    Returns:
        pd.DataFrame: DataFrame with energy mix percentages
    """
    # Create a copy to avoid modifying the original
    df = generation_data.copy()
    
    # Calculate total generation
    df['total'] = df.sum(axis=1)
    
    # Calculate percentages
    for col in df.columns:
        if col != 'total':
            df[f'{col}_pct'] = df[col] / df['total'] * 100
    
    return df

def calculate_emissions(generation_data, emission_factors):
    """
    Calculate emissions from generation data.
    
    Args:
        generation_data (pd.DataFrame): DataFrame with generation data by source
        emission_factors (dict): Dictionary mapping source names to emission factors
        
    Returns:
        pd.DataFrame: DataFrame with emissions by source
    """
    # Create a copy to avoid modifying the original
    df = generation_data.copy()
    
    # Calculate emissions for each source
    emissions = pd.DataFrame(index=df.index)
    
    for source, factor in emission_factors.items():
        if source in df.columns:
            emissions[f'{source}_emissions'] = df[source] * factor
    
    # Calculate total emissions
    emissions['total_emissions'] = emissions.sum(axis=1)
    
    return emissions

def calculate_costs(generation_data, cost_factors):
    """
    Calculate costs from generation data.
    
    Args:
        generation_data (pd.DataFrame): DataFrame with generation data by source
        cost_factors (dict): Dictionary mapping source names to cost factors
        
    Returns:
        pd.DataFrame: DataFrame with costs by source
    """
    # Create a copy to avoid modifying the original
    df = generation_data.copy()
    
    # Calculate costs for each source
    costs = pd.DataFrame(index=df.index)
    
    for source, factor in cost_factors.items():
        if source in df.columns:
            costs[f'{source}_cost'] = df[source] * factor
    
    # Calculate total costs
    costs['total_cost'] = costs.sum(axis=1)
    
    return costs

def resample_data(data, freq='1H', agg_func='mean'):
    """
    Resample time series data to a different frequency.
    
    Args:
        data (pd.DataFrame): DataFrame with time series data
        freq (str): Frequency string (e.g., '1H', '1D', '1W')
        agg_func (str or dict): Aggregation function(s)
        
    Returns:
        pd.DataFrame: Resampled data
    """
    # Check if index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex for resampling")
    
    # Resample data
    if isinstance(agg_func, str):
        resampled = data.resample(freq).agg(agg_func)
    else:
        resampled = data.resample(freq).agg(agg_func)
    
    return resampled

def detect_anomalies(data, column, window=24, threshold=3.0):
    """
    Detect anomalies in time series data using rolling statistics.
    
    Args:
        data (pd.DataFrame): DataFrame with time series data
        column (str): Column to check for anomalies
        window (int): Window size for rolling statistics
        threshold (float): Threshold for anomaly detection (in standard deviations)
        
    Returns:
        pd.Series: Boolean series indicating anomalies
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = data[column].rolling(window=window).mean()
    rolling_std = data[column].rolling(window=window).std()
    
    # Calculate z-scores
    z_scores = (data[column] - rolling_mean) / rolling_std
    
    # Detect anomalies
    anomalies = (z_scores.abs() > threshold)
    
    return anomalies

def fill_missing_values(data, method='linear'):
    """
    Fill missing values in time series data.
    
    Args:
        data (pd.DataFrame): DataFrame with time series data
        method (str): Interpolation method ('linear', 'ffill', 'bfill', etc.)
        
    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Fill missing values
    df.interpolate(method=method, inplace=True)
    
    # Forward fill any remaining NaNs at the beginning
    df.fillna(method='ffill', inplace=True)
    
    # Backward fill any remaining NaNs at the end
    df.fillna(method='bfill', inplace=True)
    
    return df

def add_weather_features(data, weather_data, features=None):
    """
    Add weather features to demand data.
    
    Args:
        data (pd.DataFrame): DataFrame with demand data
        weather_data (pd.DataFrame): DataFrame with weather data
        features (list, optional): List of weather features to add
        
    Returns:
        pd.DataFrame: DataFrame with added weather features
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Check if indices are compatible
    if not isinstance(df.index, pd.DatetimeIndex) or not isinstance(weather_data.index, pd.DatetimeIndex):
        raise ValueError("Both dataframes must have DatetimeIndex")
    
    # Select features to add
    if features is None:
        weather_features = weather_data
    else:
        weather_features = weather_data[features]
    
    # Merge data
    merged = pd.merge(df, weather_features, left_index=True, right_index=True, how='left')
    
    # Fill missing values
    merged = fill_missing_values(merged)
    
    return merged

def export_data(data, file_path, format='csv'):
    """
    Export data to a file.
    
    Args:
        data (pd.DataFrame): DataFrame to export
        file_path (str): Path to save the file
        format (str): File format ('csv', 'excel', 'pickle', 'parquet')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Export data
    if format.lower() == 'csv':
        data.to_csv(file_path)
    elif format.lower() == 'excel':
        data.to_excel(file_path)
    elif format.lower() == 'pickle':
        data.to_pickle(file_path)
    elif format.lower() == 'parquet':
        data.to_parquet(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data exported to {file_path}")

def calculate_performance_metrics(episode_data):
    """
    Calculate performance metrics from episode data.
    
    Args:
        episode_data (dict): Dictionary containing episode data
            - costs: List of costs per episode
            - emissions: List of emissions per episode
            - imbalances: List of grid imbalances per episode
            - renewable_usage: List of renewable usage percentages per episode
            
    Returns:
        dict: Dictionary of performance metrics
    """
    metrics = {}
    
    # Calculate average metrics
    metrics['avg_cost'] = np.mean(episode_data['costs'])
    metrics['avg_emissions'] = np.mean(episode_data['emissions'])
    metrics['avg_imbalance'] = np.mean(episode_data['imbalances'])
    metrics['avg_renewable_usage'] = np.mean(episode_data['renewable_usage'])
    
    # Calculate standard deviations
    metrics['std_cost'] = np.std(episode_data['costs'])
    metrics['std_emissions'] = np.std(episode_data['emissions'])
    metrics['std_imbalance'] = np.std(episode_data['imbalances'])
    metrics['std_renewable_usage'] = np.std(episode_data['renewable_usage'])
    
    # Calculate min/max values
    metrics['min_cost'] = np.min(episode_data['costs'])
    metrics['max_cost'] = np.max(episode_data['costs'])
    metrics['min_emissions'] = np.min(episode_data['emissions'])
    metrics['max_emissions'] = np.max(episode_data['emissions'])
    metrics['min_imbalance'] = np.min(episode_data['imbalances'])
    metrics['max_imbalance'] = np.max(episode_data['imbalances'])
    metrics['min_renewable_usage'] = np.min(episode_data['renewable_usage'])
    metrics['max_renewable_usage'] = np.max(episode_data['renewable_usage'])
    
    return metrics

def create_results_directory(subdir=None):
    """
    Create a directory structure for storing results.
    
    Args:
        subdir (str, optional): Subdirectory name within results directory
        
    Returns:
        str: Path to the created directory
    """
    # Create main directories if they don't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create standard subdirectories
    training_dir = os.path.join(results_dir, "training")
    evaluation_dir = os.path.join(results_dir, "evaluation")
    hyperparameter_dir = os.path.join(results_dir, "hyperparameter_tuning")
    
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(hyperparameter_dir, exist_ok=True)
    
    # Create specific subdirectory if provided
    if subdir:
        subdir_path = os.path.join(results_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        return subdir_path
    
    return results_dir

def save_training_history(history, agent_type, output_dir):
    """
    Save training history to CSV files.
    
    Args:
        history (dict): Training history
        agent_type (str): Type of agent ('ppo' or 'sac')
        output_dir (str): Directory to save files
    """
    # Create dataframe for episode metrics
    episode_df = pd.DataFrame({
        'episode': range(1, len(history['episode_rewards']) + 1),
        'reward': history['episode_rewards'],
        'length': history['episode_lengths'],
        'cost': history['episode_costs'],
        'emissions': history['episode_emissions'],
        'imbalance': history['episode_imbalances'],
        'renewable_usage': history['episode_renewable_usage']
    })
    
    # Save episode metrics
    episode_df.to_csv(os.path.join(output_dir, f"{agent_type}_episode_metrics.csv"), index=False)
    
    # Save loss metrics if available
    if agent_type.lower() == 'ppo' and 'policy_losses' in history:
        loss_df = pd.DataFrame({
            'update': range(1, len(history['policy_losses']) + 1),
            'policy_loss': history['policy_losses'],
            'value_loss': history['value_losses']
        })
        loss_df.to_csv(os.path.join(output_dir, f"{agent_type}_loss_metrics.csv"), index=False)
    
    elif agent_type.lower() == 'sac' and 'actor_losses' in history:
        loss_df = pd.DataFrame({
            'update': range(1, len(history['actor_losses']) + 1),
            'actor_loss': history['actor_losses'],
            'critic_loss': history['critic_losses'],
            'alpha_loss': history.get('alpha_losses', [0] * len(history['actor_losses']))
        })
        loss_df.to_csv(os.path.join(output_dir, f"{agent_type}_loss_metrics.csv"), index=False)

def create_tensorboard_writer(log_dir=None, agent_type=None, comment=''):
    """
    Create a TensorBoard SummaryWriter.
    
    Args:
        log_dir (str, optional): Directory to save logs
        agent_type (str, optional): Type of agent ('ppo' or 'sac')
        comment (str, optional): Comment to add to log directory name
        
    Returns:
        torch.utils.tensorboard.SummaryWriter: TensorBoard writer
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise ImportError("TensorBoard not found. Install with: pip install tensorboard")
    
    # Create log directory if not provided
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if agent_type:
            log_dir = os.path.join("logs", "tensorboard", f"{agent_type}_{timestamp}_{comment}")
        else:
            log_dir = os.path.join("logs", "tensorboard", f"run_{timestamp}_{comment}")
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create writer
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"TensorBoard logs will be saved to {log_dir}")
    print("To view logs, run: tensorboard --logdir=logs/tensorboard")
    
    return writer

def log_metrics_to_tensorboard(writer, metrics, step, prefix=''):
    """
    Log metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        metrics (dict): Dictionary of metrics to log
        step (int): Current step
        prefix (str, optional): Prefix for metric names
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", value, step)
        elif isinstance(value, np.ndarray) and value.ndim == 1:
            writer.add_histogram(f"{prefix}/{key}", value, step)
    
    writer.flush()

def log_episode_metrics_to_tensorboard(writer, episode, reward, length, cost, emissions, imbalance, renewable_usage):
    """
    Log episode metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        episode (int): Episode number
        reward (float): Episode reward
        length (int): Episode length
        cost (float): Episode cost
        emissions (float): Episode emissions
        imbalance (float): Episode grid imbalance
        renewable_usage (float): Episode renewable usage percentage
    """
    metrics = {
        'reward': reward,
        'length': length,
        'cost': cost,
        'emissions': emissions,
        'imbalance': imbalance,
        'renewable_usage': renewable_usage
    }
    
    log_metrics_to_tensorboard(writer, metrics, episode, prefix='episode')

def log_agent_update_to_tensorboard(writer, update, losses, agent_type='ppo'):
    """
    Log agent update metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        update (int): Update step
        losses (dict): Dictionary of loss values
        agent_type (str): Type of agent ('ppo' or 'sac')
    """
    if agent_type.lower() == 'ppo':
        metrics = {
            'policy_loss': losses.get('policy_loss', 0),
            'value_loss': losses.get('value_loss', 0),
            'entropy': losses.get('entropy', 0)
        }
    elif agent_type.lower() == 'sac':
        metrics = {
            'actor_loss': losses.get('actor_loss', 0),
            'critic_loss': losses.get('critic_loss', 0),
            'alpha_loss': losses.get('alpha_loss', 0),
            'alpha': losses.get('alpha', 0)
        }
    else:
        metrics = losses
    
    log_metrics_to_tensorboard(writer, metrics, update, prefix='update')

def log_environment_to_tensorboard(writer, env, step):
    """
    Log environment state to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        env: Environment instance
        step (int): Current step
    """
    # Log environment-specific metrics if available
    if hasattr(env, 'get_metrics'):
        metrics = env.get_metrics()
        log_metrics_to_tensorboard(writer, metrics, step, prefix='environment')
    
    # Log supply mix if available
    if hasattr(env, 'supply_simulator') and hasattr(env.supply_simulator, 'get_supply_mix'):
        supply_mix = env.supply_simulator.get_supply_mix()
        for source, value in supply_mix.items():
            writer.add_scalar(f"environment/supply_mix/{source}", value, step)
    
    # Log demand if available
    if hasattr(env, 'current_demand'):
        writer.add_scalar('environment/demand', env.current_demand, step)
    
    writer.flush()

def log_hyperparameters_to_tensorboard(writer, hyperparams):
    """
    Log hyperparameters to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        hyperparams (dict): Dictionary of hyperparameters
    """
    # Convert hyperparameters to string for hparams logging
    hparams = {str(k): str(v) if isinstance(v, (list, dict)) else v 
               for k, v in hyperparams.items()}
    
    # Log hyperparameters
    writer.add_hparams(hparams, {'hparam/placeholder': 0})
    writer.flush()

def close_tensorboard_writer(writer):
    """
    Close TensorBoard writer.
    
    Args:
        writer: TensorBoard SummaryWriter
    """
    if writer is not None:
        writer.close()

def main():
    """Test data processing functions."""
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', periods=24*30, freq='H')
    demand = 1000 + 200 * np.sin(np.pi * dates.hour / 12) + 100 * np.random.randn(len(dates))
    
    # Create DataFrame
    data = pd.DataFrame({
        'demand': demand
    }, index=dates)
    
    # Preprocess data
    processed_data = preprocess_demand_data(
        data, normalize=True, add_time_features=True, add_lags=True, n_lags=24
    )
    
    print("Original data shape:", data.shape)
    print("Processed data shape:", processed_data.shape)
    print("Processed data columns:", processed_data.columns.tolist())
    
    # Create sequences
    X, y = create_sequences(processed_data, target_col='demand', seq_length=24, forecast_horizon=24)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Val shapes:", X_val.shape, y_val.shape)
    print("Test shapes:", X_test.shape, y_test.shape)
    
    # Export data
    os.makedirs('data', exist_ok=True)
    export_data(data, 'data/sample_demand.csv')

if __name__ == "__main__":
    main() 