class my_class(object):
    pass
import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import h5py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

def load_datasets(folder_path):
    datasets = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith('.csv'):
            dataset = pd.read_csv(file_path, on_bad_lines='skip', dtype=str)
            print(f"Loaded CSV file: {filename}")
            datasets.append(dataset)
        
        elif filename.endswith('.npz'):
            data = np.load(file_path)
            # Assuming the .npz contains arrays named 'features' and 'labels'
            if 'features' in data and 'labels' in data:
                features = data['features']
                labels = data['labels']
                # Convert to DataFrame
                dataset = pd.DataFrame(features)
                dataset['label'] = labels
                print(f"Loaded NPZ file: {filename}")
                datasets.append(dataset)
            else:
                print(f"Warning: {filename} does not contain 'features' and 'labels'")
        else:
            print(f"Skipped file: {filename} (unsupported format)")
    
    if datasets:
        combined_data = pd.concat(datasets, ignore_index=True)
        # Ensure 'label' column is appropriately mapped
        combined_data['label'] = combined_data['label'].map({'human': 0, 'ai': 1}).astype(float)
        return combined_data
    else:
        print("No datasets loaded.")
        return None


# Specify the folder path for the datasets
folder_path = 'path_to_your_csv_files'

# Load the datasets once
data = load_datasets(folder_path)

if data is None:
    print("Could not load datasets properly. Exiting the script.")
else:
    # Preprocess the dataset
    # Fill or drop NaN values as appropriate
    data.fillna(method='fill', inplace=True)  # Fill NaN values with the forward fill method

    # Print the data for debugging
    print("Data after filling NaN values:")
    print(data.head())

    # Ensure label column is numeric
    data['label'] = data['label'].astype(float)

    # Check if the dataset is empty
    if data.empty:
        print("No data available after preprocessing. Please check the input files.")
    else:
        # Feature extraction (assuming features are already in the datasets)
        X = data.drop('label', axis=1).astype(float)
        y = data['label']

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'NeuralNetwork': Sequential([
                Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
        }

        for name, model in models.items():
            if name == 'NeuralNetwork':
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
                y_pred = (model.predict(X_test) > 0.5).astype(int)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            print(f"Evaluation for {name}:")
            print(classification_report(y_test, y_pred))

            # Save the model for later use
            if name == 'NeuralNetwork':
                model.save(f'{name}_model.h5')
            else:
                joblib.dump(model, f'{name}_model.pkl')

        # Use the specified file path for the TensorFlow model
        tf_model_path = "path_to_your_tensorflow_model.h5"
        loaded_model = None
        if tf_model_path:
            try:
                loaded_model = load_model(tf_model_path)  # Load your pre-trained TensorFlow model
                print("TensorFlow model loaded successfully!")
            except Exception as e:
                print(f"Error opening TensorFlow model file: {e}")

        # Only run prediction if TensorFlow model loaded successfully
        if loaded_model:
            # Use the specified file path for the audio file
            file_path = "path_to_your_audio_file.mp3"
            features = extract_features(file_path)
            if features is not None:
                features = np.array([features])
                prediction = (loaded_model.predict(features) > 0.5).astype(int)
                result = 'AI Generated' if prediction == 1 else 'Human Generated'
                print(f"Result: {result}")
        else:
            print("No TensorFlow model loaded, skipping to fallback models.")
            
            # Fall back to other models
            rf_model_path = 'RandomForest_model.pkl'
            gb_model_path = 'GradientBoosting_model.pkl'
            
            fallback_models = {}
            try:
                fallback_models['RandomForest'] = joblib.load(rf_model_path)
            except Exception as e:
                print(f"Error loading RandomForest model: {e}")
                
            try:
                fallback_models['GradientBoosting'] = joblib.load(gb_model_path)
            except Exception as e:
                print(f"Error loading GradientBoosting model: {e}")

            for name, model in fallback_models.items():
                if model:
                    features = extract_features(file_path)
                    if features is not None:
                        features = np.array([features])
                        prediction = model.predict(features)
                        result = 'AI Generated' if prediction ==



