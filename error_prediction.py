import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

# dir_name = 'ogi/Mehanicke primese'
# well_name = 'well_1_5-30-2250_2020-10-04'

# Function to downsample data by day, where each day has num_of_points_per_day points
def sort_by_day(df, num_of_points_per_day):
    # Convert date column to datetime if needed
    df['measure_date_only'] = pd.to_datetime(df['measure_date_only'])
    
    # Group by the day, but also keep measure_date_only column
    grouped = df.groupby('measure_date_only')
    
    # List to store downsampled data and corresponding days
    downsampled_data = []
    downsampled_dates = []
    
    # Process each day's data
    for day, group in grouped:
        # Select only numeric columns for downsampling
        numeric_group = group.select_dtypes(include=[np.number])
        group_size = len(numeric_group)
        
        # Check if there are enough points to perform downsampling
        if group_size < num_of_points_per_day:
            # If not enough points, just append the original group (or handle it as needed)
            downsampled_data.append(numeric_group.mean())
            downsampled_dates.append(day)  # Store the date
            continue
        
        # Calculate chunk size for downsampling
        chunk_size = group_size // num_of_points_per_day
        
        # Downsample by averaging chunks
        for i in range(num_of_points_per_day):
            # Define the start and end of the current chunk
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_of_points_per_day - 1 else group_size  # last chunk goes to the end
            
            # Average over the current chunk and append
            downsampled_chunk = numeric_group.iloc[start:end].mean()
            downsampled_data.append(downsampled_chunk)
            downsampled_dates.append(day)  # Store the date for each downsampled chunk

    # Combine all downsampled data into a DataFrame
    downsampled_df = pd.DataFrame(downsampled_data)
    
    # Add the dates as a new column
    downsampled_df['measure_date'] = downsampled_dates
    
    # Reset index to have a cleaner DataFrame
    downsampled_df.reset_index(drop=True, inplace=True)
    
    return downsampled_df

# Butterworth low-pass filter design
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply low-pass filter
def apply_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    filtered_data = filtfilt(b, a, data)  # Zero-phase filtering
    return filtered_data

# Normalize data between 0 and 1 using MinMaxScaler
def normalize_data(df, telemetry_cols):
    # Drop any rows with missing values
    df.dropna(inplace=True)
    scaler = MinMaxScaler()
    df[telemetry_cols] = scaler.fit_transform(df[telemetry_cols])
    return df

# Apply low-pass filter to telemetry columns
def low_pass(df, telemetry_cols, cutoff_frequency, sampling_rate):    
    # Apply low-pass filtering to each telemetry column
    for col in telemetry_cols:
        df[col] = apply_lowpass_filter(df[col], cutoff_frequency, sampling_rate)
    
    return df

def exponential_moving_average(df, telemetry_cols, span):
    for col in telemetry_cols:
        df[col] = df[col].ewm(span=span, adjust=False).mean()
    return df


# Calculates correlation between telemetry cols `koeficijent_kapaciteta` and `frekvencija`
# and adds it as a new feature `correlation_koeficijent_kapaciteta_frekvencija`
def correlation_feature(df, threshold=0.0002):
    # Compute the rolling correlation between the two columns
    df['correlation_koeficijent_kapaciteta_frekvencija'] = df['koeficijent_kapaciteta'].rolling(window=50).corr(df['frekvencija'])
    
    # Fill any NaN values that result from the rolling correlation at the start
    df['correlation_koeficijent_kapaciteta_frekvencija'].fillna(method='bfill', inplace=True)
    
    # Calculate the rolling differences
    rolling_diff_kapaciteta = df['koeficijent_kapaciteta'].diff().rolling(window=50).std()
    rolling_diff_frekvencija = df['frekvencija'].diff().rolling(window=50).std()
    
    # Detect oscillation based on the rolling standard deviations
    df['koeficijent_kapaciteta_frekvencija_oscillation_flag'] = np.where(
        (rolling_diff_kapaciteta > threshold) & (rolling_diff_frekvencija > threshold), 
        1, 
        0
    )

    df['correlation_pritisak_frekvencija'] = df['pritisak_na_prijemu_pumpe'].rolling(window=50).corr(df['frekvencija'])

    #Fill any NaN values that result from the rolling correlation at the start
    df['correlation_pritisak_frekvencija'].fillna(method='bfill', inplace=True)

    #Calculate the rolling differences
    rolling_diff_pritisak = df['pritisak_na_prijemu_pumpe'].diff().rolling(window=50).std()
    rolling_diff_frekvencija = df['frekvencija'].diff().rolling(window=50).std()

    #Detect oscillation based on the rolling standard deviations
    df['pritisak_frekvencija_oscillation_flag'] = np.where(
        (rolling_diff_pritisak > threshold) | (rolling_diff_frekvencija > threshold),
        1,
        0
    )

    df['pritisak_oscillation_flag'] = np.where(
        (rolling_diff_pritisak > threshold*1.5),
        1,
        0
    )
    
    return df


# Take the last window_size points and predict, then
# move the window by 50 points and repeat the process
# each prediction will assign one of the following labels: 'normal', 'mehanicke primese', 'nehermeticnost kompozicije tubinga' or 'talozenje kamenca' 
def make_prediction(df, dir_name, window_size=500, corr_koeficijent_frekvencija=-0.5, threshold_count=70, max_pritisak_threshold=30):
    predictions = []
    anomaly = 0
    prediction = 0  # Normal behavior

    # if dir_name is 'Talozenje kamenca', then label everything as 'normal', just last 1000 points as 'talozenje kamenca'
    if dir_name.endswith('Talozenje kamenca'):
        df['prediction'] = 0
        if len(df) > 1000:
            df['prediction'].iloc[-1000:] = 2
        else:
            df['prediction'] = 2
        return df

    # Iterate over the dataframe in windows
    for i in range(0, len(df) - window_size + 1, window_size):
        
        # Initialize counters for anomalies
        count_anomalies = 0
        max_pritisak_subsequence = 0
        current_subsequence = 0
        
        # Check each point in the window for anomalies
        for j in range(i, i + window_size):
            if anomaly:
                prediction = anomaly
                break
            # Condition 1: correlation < threshold and oscillation flag == 1
            if (df['correlation_koeficijent_kapaciteta_frekvencija'].iloc[j] < corr_koeficijent_frekvencija and 
                df['koeficijent_kapaciteta_frekvencija_oscillation_flag'].iloc[j] == 1):
                count_anomalies += 1

            # Condition 2: max subsequence of `pritisak_oscillation_flag == 1`
            if df['pritisak_oscillation_flag'].iloc[j] == 1:
                current_subsequence += 1
            else:
                # Update max subsequence
                max_pritisak_subsequence = max(max_pritisak_subsequence, current_subsequence)
                current_subsequence = 0
        
        # After window loop, ensure the subsequence is updated for the last batch
        max_pritisak_subsequence = max(max_pritisak_subsequence, current_subsequence)
        
        # Rule-based prediction assignment
        if count_anomalies > threshold_count or max_pritisak_subsequence > max_pritisak_threshold or anomaly:
            prediction = 1  if dir_name.endswith('Mehanicke primese') else 3 # 'mehanicke primese' or 'nehermeticnost kompozicije tubinga'
            anomaly = prediction
        else:
            prediction = 0  # Normal behavior
        
        # Apply the same prediction for the entire window
        predictions.extend([prediction] * window_size)

    # If there's any leftover (because the window might not perfectly cover the end), fill with the last prediction
    leftover = len(df) % window_size
    if leftover > 0:
        predictions.extend([prediction] * leftover)
    
    # Add the predictions as a new column in the dataframe
    df['prediction'] = predictions[:len(df)]  # Ensure that we match the length of df
    
    return df


# Helper function to split dataframe by continuous prediction segments
def split_by_prediction(df, prediction_col):
    # Identify where prediction changes
    df['segment'] = (df[prediction_col] != df[prediction_col].shift()).cumsum()
    segments = [segment_df for _, segment_df in df.groupby('segment')]
    
    return segments

# Plot telemetry data with different colors based on predictions
def plot_telemetry_data(df, telemetry_cols):
    num_features = len(telemetry_cols)
    
    # Create subplots: one for each telemetry feature
    fig, axs = plt.subplots(num_features, 1, figsize=(10, 5 * num_features), sharex=True)
    
    # Define color map and custom labels for predictions
    colors = {0: 'green', 1: 'purple', 2: 'orange', 3: 'red'}
    custom_labels = {0: 'Normalno', 1: 'mehanicke primese', 2: 'talozenje kamenca', 3: 'nehermeticnost kompozicije tubinga'}
    
    # Split dataframe into continuous segments by prediction
    segments = split_by_prediction(df, 'prediction')
    
    # Plot each telemetry column on its own subplot
    for i, col in enumerate(telemetry_cols):
        for segment in segments:
            prediction_value = segment['prediction'].iloc[0]  # Get the prediction value for this segment
            axs[i].plot(segment.index, segment[col], label=custom_labels[prediction_value], color=colors[prediction_value])
        
        axs[i].set_ylabel('Telemetry Values')
        axs[i].set_title(f'{col}')
        axs[i].legend(loc='upper right', title="Prediction/Error")
        axs[i].grid(True)

    # Set common x-label
    axs[-1].set_xlabel('Date')
    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    
    plt.show()

# Load well data, process it, apply filters, and plot
def load_well(dir_name, well_name, telemetry_cols, num_of_points_per_day=100, cutoff_frequency=0.7, sampling_rate=100):
    well_path = f"{dir_name}/{well_name}.csv"
    df = pd.read_csv(well_path)
        
    # Downsample data
    df = sort_by_day(df, num_of_points_per_day)
    
    # Normalize data
    df = normalize_data(df, telemetry_cols)

    # Apply low-pass filter
    df = low_pass(df, telemetry_cols, cutoff_frequency, sampling_rate)
    
    # Drop the first third of the data
    df = df.iloc[len(df)//3:]

    # Calculate correlation and add it as a new feature
    df = correlation_feature(df)
    telemetry_cols.append('correlation_koeficijent_kapaciteta_frekvencija')
    telemetry_cols.append('koeficijent_kapaciteta_frekvencija_oscillation_flag')
    telemetry_cols.append('pritisak_oscillation_flag')
    telemetry_cols.append('correlation_pritisak_frekvencija')
    telemetry_cols.append('pritisak_frekvencija_oscillation_flag')

    # assign label to the data (needs to be implemented)
    df = make_prediction(df, dir_name)

    # Plot the telemetry data
    plot_telemetry_data(df, telemetry_cols)
    
    return df

def loading_well_data(dir_name, well_name, telemetry_cols, num_of_points_per_day=100, cutoff_frequency=0.7, sampling_rate=100):
    well_path = f"{dir_name}/{well_name}.csv"
    df = pd.read_csv(well_path)
        
    # Downsample data
    df = sort_by_day(df, num_of_points_per_day)
    
    # Normalize data
    df = normalize_data(df, telemetry_cols)

    # Apply low-pass filter
    df = low_pass(df, telemetry_cols, cutoff_frequency, sampling_rate)
    
    # Drop the first third of the data
    df = df.iloc[len(df)//3:]

    # Calculate correlation and add it as a new feature
    df = correlation_feature(df)
    telemetry_cols.append('correlation_koeficijent_kapaciteta_frekvencija')
    telemetry_cols.append('koeficijent_kapaciteta_frekvencija_oscillation_flag')
    telemetry_cols.append('pritisak_oscillation_flag')
    telemetry_cols.append('correlation_pritisak_frekvencija')
    telemetry_cols.append('pritisak_frekvencija_oscillation_flag')

    # assign label to the data (needs to be implemented)
    df = make_prediction(df, dir_name)
    
    return df

def iterate_over_wells():

    df = pd.DataFrame()
    for dir_name in ['ogi/Mehanicke primese', 'ogi/Nehermeticnost kompozicije tubinga', 'ogi/Talozenje kamenca']:

        # Iterate through the all files in the directory and save all the data into a single DataFrame
        for file_name in os.listdir(dir_name):
            if file_name.endswith('.csv'):
                well_name = file_name[:-4]
                print(f'Processing well: {well_name}')
                new_df = loading_well_data(dir_name, well_name, 
                                       ['napon_ca', 'elektricna_struja_fazaa', 
           'koeficijent_kapaciteta', 'frekvencija', 'radno_opterecenje', 'pritisak_na_prijemu_pumpe', 'temperatura_motora', 
           'temperatura_u_busotini'])
                
                new_df['well_name'] = well_name
                df = pd.concat([df, new_df], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    df.to_csv(f'all_data.csv', index=False)

# iterate_over_wells()        

# dir_name = 'ogi/Talozenje kamenca'
# well_name = 'well_47_5-30-2100_2021-08-05'

# # # # Example of using the function
# load_well(dir_name, well_name, 
#           ['napon_ca', 'elektricna_struja_fazaa', 
#            'koeficijent_kapaciteta', 'frekvencija', 'radno_opterecenje', 'pritisak_na_prijemu_pumpe', 'temperatura_motora', 
#            'temperatura_u_busotini'], 
#           )

# Function to reshape the dataset
def reshape_dataset(filename='all_data', window_size=200):
    df = pd.read_csv(f'{filename}.csv')
    df.dropna(inplace=True)
    
    # Drop unnecessary columns, including 'prediction' for X
    df.drop(columns=['napon_ab', 'napon_bc', 'elektricna_struja_fazab', 
                     'aktivna_snaga', 'elektricna_struja_fazac', 
                     'well_name', 'measure_date'], inplace=True)
    
    # Convert dataframe to a NumPy array for faster processing
    data = df.drop(columns=['prediction']).values
    labels = df['prediction'].astype(int).values  # Cast labels to integers
    
    num_samples = data.shape[0] - window_size
    
    # Pre-allocate memory for X and y
    X = np.zeros((num_samples, window_size, data.shape[1]))
    y = np.zeros(num_samples)

    # Create the sliding windows more efficiently
    for i in range(num_samples):
        X[i] = data[i:i + window_size]
        y[i] = labels[i + window_size]

    # Flatten X
    X = X.reshape(X.shape[0], -1)
    
    return X, y

def model_training():

    # Load and reshape dataset
    print('Loading and reshaping dataset...')
    X, y = reshape_dataset()

    print("Unique values in y:", np.unique(y))

    # Split the dataset into train and test sets (stratified for balanced classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print('Creating model...')
    # Initialize XGBoost classifier
    xgb_model = XGBClassifier(n_estimators=10, max_depth=5, learning_rate=0.1, random_state=42, device='gpu')

    print('Training the model...')

    # Train the model
    xgb_model.fit(X_train, y_train, verbose=True)


    print('Evaluating the model...')
    # Make predictions on the test set
    y_pred = xgb_model.predict(X_test)

    # Output classification report
    print(classification_report(y_test, y_pred))

    # Get feature importances
    # importances = xgb_model.feature_importances_
    # plot_importance(xgb_model, importance_type='weight', max_num_features=10, title='Feature Importance', xlabel='F score', ylabel='Features')
    # plt.show()

    # Binarize the output for multiclass AUC
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3])  # Update classes according to your dataset
    y_prob = xgb_model.predict_proba(X_test)

    # Calculate AUC for each class and average
    auc_scores = []
    for i in range(y_test_binarized.shape[1]):
        auc = roc_auc_score(y_test_binarized[:, i], y_prob[:, i])
        auc_scores.append(auc)
        print(f'AUC for class {i}: {auc:.4f}')

    # Average AUC
    average_auc = sum(auc_scores) / len(auc_scores)
    print(f'Average AUC: {average_auc:.4f}')

    class_names = ['normalno', 'mehanicke primese', 'talozenje kamenca', 'nehermeticnost kompozicije tubinga']

    # Compute ROC curves for each class
    for i in range(y_test_binarized.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'ROC curve for class {class_names[i]} (AUC = {auc_scores[i]:.4f})')

    # Plotting the final ROC curve
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.show()

    # save the model to disk
    import pickle
    filename = 'finalized_model.sav'
    pickle.dump(xgb_model, open(filename, 'wb'))


model_training()