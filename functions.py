import os
# audio
import librosa
import parselmouth

# etc
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import time
from copy import deepcopy

# inbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# xgboost
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# train on cuda
params = {
    "tree_method": "hist",  # Using 'hist' as the tree method
    "device": "cuda",  # Using CUDA for GPU computing instead of 'gpu_id'
    "verbosity": 2  # Higher verbosity for more detailed logging
}


EATD_AUDICO_DICT = {"negative": 'negative_out.wav', 
                "neutral": 'neutral_out.wav', 
                "positive": 'positive_out.wav'}

def save_audio_features_data(X_train, y_train, feature_path, label_path):
    np.save(feature_path, X_train)  # Saves the feature array to a binary file
    pd.DataFrame({'label': y_train}).to_csv(label_path, index=False)  # Saves labels to a CSV file

def _save_audio_features_data_best_paper(features_list, labels, feature_path, label_path):
    # Convert list of dictionaries to arrays
    aggregated_features = np.array([f['aggregated_features'] for f in features_list])
    mfcc_spectrums = np.array([f['mfcc_spectrum'] for f in features_list])
    
    # Save as a compressed .npz file to handle multiple arrays
    np.savez(feature_path, 
             aggregated_features=aggregated_features,
             mfcc_spectrums=mfcc_spectrums)
    
    # Save labels
    pd.DataFrame({'label': labels}).to_csv(label_path, index=False)

# To load later:
def load_mfcc_data(mfcc_path):
    data = np.load(mfcc_path)
    return [data[f'mfcc_{i}'] for i in range(len(data.files))]

def save_audio_features_data_best_paper(features_list, labels, feature_path, label_path):
    # Save aggregated features
    aggregated_features = np.array([f['aggregated_features'] for f in features_list])
    np.save(feature_path, aggregated_features)
    
    # Save MFCC spectrums separately
    mfcc_path = feature_path.replace('.npy', '_mfcc.npz')
    
    # Save each MFCC spectrum with its index
    mfcc_dict = {f'mfcc_{i}': f['mfcc_spectrum'] for i, f in enumerate(features_list)}
    np.savez(mfcc_path, **mfcc_dict)
    
    # Save labels
    pd.DataFrame({'label': labels}).to_csv(label_path, index=False)


def load_audio_features_data(feature_path, label_path):
    X_train = np.load(feature_path)  # Loads features from the binary file
    y_train = pd.read_csv(label_path)['label'].values  # Loads labels from the CSV file
    return X_train, y_train

def train_balanced_xgboost_grid_search(X_train, y_train, balance_strategy='combine'):
    """
    Train XGBoost with adaptive number of CV folds based on smallest class size.
    """
    # First, analyze class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    min_samples = np.min(counts)
    
    print("\nInitial class distribution:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples")
    
    # Determine number of folds based on smallest class size
    n_splits = min(3, min_samples)  # Use at most 3-fold CV, or fewer if necessary
    print(f"\nUsing {n_splits}-fold cross-validation due to class sizes")
    
    # Handle the balancing strategy
    if balance_strategy == 'smote':
        print("\nApplying SMOTE oversampling...")
        try:
            k_neighbors = min(min_samples - 1, 5)
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
        except ValueError as e:
            print(f"SMOTE failed: {e}")
            print("Falling back to weighted approach.")
            balance_strategy = 'weighted'
            X_train_balanced, y_train_balanced = X_train, y_train
            
    elif balance_strategy == 'undersample':
        print("\nApplying undersampling to majority class...")
        try:
            sampler = RandomUnderSampler(random_state=42)
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
        except ValueError as e:
            print(f"Undersampling failed: {e}")
            balance_strategy = 'weighted'
            X_train_balanced, y_train_balanced = X_train, y_train
            
    elif balance_strategy == 'combine':
        print("\nApplying combined SMOTE and undersampling...")
        try:
            k_neighbors = min(min_samples - 1, 5)
            over = SMOTE(k_neighbors=k_neighbors, random_state=42)
            under = RandomUnderSampler(random_state=42)
            pipeline = Pipeline([('over', over), ('under', under)])
            X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)
        except ValueError as e:
            print(f"Combined sampling failed: {e}")
            balance_strategy = 'weighted'
            X_train_balanced, y_train_balanced = X_train, y_train
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    print("\nClass distribution after balancing:")
    unique_balanced, counts_balanced = np.unique(y_train_balanced, return_counts=True)
    for cls, count in zip(unique_balanced, counts_balanced):
        print(f"Class {cls}: {count} samples")
    
    # Calculate class weights for weighted approach
    class_weights = {cls: len(y_train) / (len(unique) * count) 
                    for cls, count in zip(unique, counts)}
    
    # Define parameter grid for tuning
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # Create base XGBoost classifier
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique),
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Perform grid search with adjusted number of folds
    if balance_strategy == 'weighted':
        sample_weights = np.array([class_weights[y] for y in y_train])
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=n_splits,
            scoring='balanced_accuracy',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=n_splits,
            scoring='balanced_accuracy',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train_balanced, y_train_balanced)
    
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    return grid_search.best_estimator_

def __train_balanced_xgboost_grid_search(X_train, y_train, balance_strategy='combine'):
    """
    Train XGBoost with various balancing strategies and grid search for optimal parameters.
    
    Args:
        balance_strategy: 'smote', 'undersample', 'combine', or 'weighted'
    """
    # First, analyze class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    min_samples = np.min(counts)
    
    print("\nInitial class distribution:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples")
    
    # Determine appropriate k for SMOTE based on minimum class size
    k_neighbors = min(min_samples - 1, 5)  # k should be less than min class size
    
    # Handle the balancing strategy
    if k_neighbors < 2:
        print("\nWarning: Some classes have too few samples for SMOTE.")
        print("Falling back to weighted approach.")
        balance_strategy = 'weighted'
        X_train_balanced, y_train_balanced = X_train, y_train
    else:
        if balance_strategy == 'smote':
            print(f"\nApplying SMOTE oversampling with k_neighbors={k_neighbors}...")
            try:
                sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
            except ValueError as e:
                print(f"SMOTE failed: {e}")
                print("Falling back to weighted approach.")
                balance_strategy = 'weighted'
                X_train_balanced, y_train_balanced = X_train, y_train
                
        elif balance_strategy == 'undersample':
            print("\nApplying undersampling to majority class...")
            try:
                sampler = RandomUnderSampler(random_state=42)
                X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
            except ValueError as e:
                print(f"Undersampling failed: {e}")
                balance_strategy = 'weighted'
                X_train_balanced, y_train_balanced = X_train, y_train
                
        elif balance_strategy == 'combine':
            print("\nApplying combined SMOTE and undersampling...")
            try:
                over = SMOTE(k_neighbors=k_neighbors, random_state=42)
                under = RandomUnderSampler(random_state=42)
                pipeline = Pipeline([('over', over), ('under', under)])
                X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)
            except ValueError as e:
                print(f"Combined sampling failed: {e}")
                balance_strategy = 'weighted'
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
    
    # Calculate class weights for weighted approach
    class_weights = {cls: len(y_train) / (len(unique) * count) 
                    for cls, count in zip(unique, counts)}
    
    # Define parameter grid for tuning
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # Create base XGBoost classifier
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique),
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Perform grid search
    if balance_strategy == 'weighted':
        # Add sample weights to grid search
        sample_weights = np.array([class_weights[y] for y in y_train])
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='balanced_accuracy',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='balanced_accuracy',
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train_balanced, y_train_balanced)
    
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    return grid_search.best_estimator_


def feature_importance_xgboost(model, number_of_features=10):
    """
    Plot the top N most important features from an XGBoost model.
    
    Args:
        model: Trained XGBoost model
        number_of_features: Number of top features to display (default=10)
    """
    # Get feature importance
    importance_type = 'weight'  # can also use 'gain' or 'cover'
    feature_importance = model.get_booster().get_score(importance_type=importance_type)
    
    # Convert to list of tuples and sort by importance value
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Take only the top N features
    top_features = sorted_features[:number_of_features]
    
    # Separate features and values
    features, values = zip(*top_features)
    
    # Create position array for plotting
    pos = np.arange(len(features))
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.barh(pos, values)
    plt.yticks(pos, features)
    plt.xlabel(f'Feature Importance ({importance_type})')
    plt.title(f'Top {number_of_features} Most Important Features')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()
    
    # Print numerical values
    """
    
    print("\nTop Feature Importance Values:")
    for feature, value in top_features:
        print(f"{feature}: {value:.4f}")
    """

def train_xgboost_grid_search_simple(X_train, y_train, use_cuda=True):
    print("     TRAIN STARTS")
    """Train XGBoost model with hyperparameter tuning and evaluate it."""
    # measuring running time:
    start_time = time.time()
    
    # Define parameter grid for tuning
    NUM_CLASSES = len(np.unique(y_train))  

    if use_cuda:
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'tree_method': ['gpu_hist'],  # Use GPU optimized histogram algorithm
            'predictor': ['gpu_predictor']  # Use GPU for prediction
        }
    
        # Create XGBoost classifier
        xgb_clf = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=NUM_CLASSES,  # number of depression categories
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            tree_method='gpu_hist',  # Ensures the model uses the GPU histogram method
            predictor='gpu_predictor'  # Ensures predictions are on GPU
        )
    else:
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }

        # Create XGBoost classifier
        xgb_clf = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=NUM_CLASSES,  # number of depression categories
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    print("Training XGBoost model with grid search...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_

    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")

    end = time.time()
    print(f"Training time: {end - start_time:.2f} seconds")

    return best_model

def confusion_matrix_report(true_lables, pred_labels, labels):
    # Calculate and display metrics
    print("\nClassification Report:")
    print(classification_report(true_lables, pred_labels, 
                                target_names=labels))

    # Create confusion matrix
    cm = confusion_matrix(true_lables, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def extract_audio_features_mfcc(file_path):
    """Extract all audio features used in the paper"""
    # Load audio
    y, sr = librosa.load(file_path, duration=30)

    # MFCC features (they used MFC with multiple features)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # They used 40 MFCCs
    
    # Additional features mentioned in paper
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Calculate statistics
    features = {
        'mfcc_mean': np.mean(mfccs, axis=1),
        'mfcc_std': np.std(mfccs, axis=1),
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
        'pitch': librosa.piptrack(y=y, sr=sr)[1].mean(),
        'energy': np.sum(y**2),
        'speech_rate': len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y)/sr)
    }

    # Flatten
    feature_vector = []
    for value in features.values():
        if isinstance(value, np.ndarray):
            feature_vector.extend(value)
        else:
            feature_vector.append(value)
            
    return {
        'aggregated_features': np.array(feature_vector),
        'mfcc_spectrum': mfccs  # Full MFCC spectrum (40 x time_steps)
    }

def extract_audio_features_simple(file_path):
    """Extract relevant audio features using librosa."""
    # Load the audio file
    y, sr = librosa.load(file_path, duration=30)  # Load first 30 seconds
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Calculate statistics for each feature
    features = {
        'mfcc_mean': np.mean(mfccs, axis=1),
        'mfcc_std': np.std(mfccs, axis=1),
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
    }
    
    # Flatten the dictionary into a single feature vector
    feature_vector = []
    for value in features.values():
        if isinstance(value, np.ndarray):
            feature_vector.extend(value)
        else:
            feature_vector.append(value)
            
    return np.array(feature_vector)

def extract_audio_features_best(audio_file):
    # Load the audio file with parselmouth
    sound = parselmouth.Sound(audio_file)
    
    # Fundamental Frequency (F0)
    pitch = sound.to_pitch()
    f0 = pitch.selected_array['frequency']
    
    # Harmonic-to-Noise Ratio (HNR)
    hnr = sound.to_harmonicity()
    
    # Mel-frequency cepstral coefficients (MCEP)
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
    
    # Spectral Slope
    spectral_slope = librosa.feature.spectral_slope(y=y, sr=sr)
    
    # Extract other features using Praat through parselmouth or other libraries
    
    features = {
        'f0_mean': np.nanmean(f0),
        'f0_std': np.nanstd(f0),
        'hnr_mean': np.mean(hnr.values),
        'mfccs_mean': np.mean(mfccs, axis=1),
        'spectral_slope_mean': np.mean(spectral_slope)
        # Add other extracted features
    }
    return features



def extract_audio_features_complex(file_path):
    """
    Extract comprehensive audio features using librosa.
    Includes temporal, spectral, and rhythm features.
    """
    # Load the audio file
    y, sr = librosa.load(file_path, duration=30)  # Load first 30 seconds
    
    features = {}
    
    # 1. Basic Features
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Energy
    energy = np.sum(np.abs(y)**2, axis=0)
    features['energy'] = energy
    
    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # 2. Spectral Features
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spectral_contrast)
    features['spectral_contrast_std'] = np.std(spectral_contrast)
    
    # Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    features['spectral_flatness_std'] = np.std(spectral_flatness)
    
    # 3. MFCCs and Delta Features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Add each MFCC coefficient and its deltas
    for i in range(13):
        features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
        features[f'mfcc{i+1}_delta_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc{i+1}_delta_std'] = np.std(mfcc_delta[i])
        features[f'mfcc{i+1}_delta2_mean'] = np.mean(mfcc_delta2[i])
        features[f'mfcc{i+1}_delta2_std'] = np.std(mfcc_delta2[i])
    
    # 4. Rhythm Features
    # Tempo and Beat Features
    # tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # features['tempo'] = tempo
    # features['beats_mean'] = np.mean(np.diff(beats)) if len(beats) > 1 else 0
    # features['beats_std'] = np.std(np.diff(beats)) if len(beats) > 1 else 0
    
    # Onset Features
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features['onset_strength_mean'] = np.mean(onset_env)
    features['onset_strength_std'] = np.std(onset_env)
    
    # 5. Voice-specific Features
    # Fundamental Frequency (F0) Features
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                fmax=librosa.note_to_hz('C7'))
    features['f0_mean'] = np.mean(f0[~np.isnan(f0)]) if any(~np.isnan(f0)) else 0
    features['f0_std'] = np.std(f0[~np.isnan(f0)]) if any(~np.isnan(f0)) else 0
    features['voiced_fraction'] = np.mean(voiced_flag)
    
    # 6. Harmonic Features
    harmonic, percussive = librosa.effects.hpss(y)
    features['harmonic_mean'] = np.mean(harmonic)
    features['harmonic_std'] = np.std(harmonic)
    features['percussive_mean'] = np.mean(percussive)
    features['percussive_std'] = np.std(percussive)
    
    # Convert dictionary to feature vector
    feature_vector = []
    feature_names = []
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            feature_vector.extend(value)
            feature_names.extend([f"{key}_{i}" for i in range(len(value))])
        else:
            feature_vector.append(value)
            feature_names.append(key)
            
    return np.array(feature_vector), feature_names

def print_feature_descriptions():
    """Print descriptions of extracted features."""
    descriptions = {
        'Zero Crossing Rate': 'Rate of sign changes in the signal. Related to percussive sounds and voice characteristics.',
        'Energy': 'Overall energy/volume of the signal.',
        'RMS': 'Root Mean Square energy. Related to perceived volume.',
        'Spectral Centroid': 'Center of mass of the spectrum. Related to brightness of sound.',
        'Spectral Bandwidth': 'Width of the spectrum. Related to richness of sound.',
        'Spectral Rolloff': 'Frequency below which most of the energy is concentrated.',
        'Spectral Contrast': 'Difference between peaks and valleys in the spectrum.',
        'Spectral Flatness': 'How noise-like vs. tone-like the sound is.',
        'MFCCs': 'Mel-frequency cepstral coefficients. Represent the speech spectrum.',
        'Delta MFCCs': 'First derivative of MFCCs. Capture temporal changes.',
        'Delta2 MFCCs': 'Second derivative of MFCCs. Capture acceleration of changes.',
        'Tempo': 'Estimated beats per minute.',
        'Beat Features': 'Statistics about rhythm regularity.',
        'Onset Strength': 'Strength of musical events/note beginnings.',
        'Fundamental Frequency': 'Basic pitch of voice.',
        'Voiced Fraction': 'Proportion of voiced vs. unvoiced segments.',
        'Harmonic Features': 'Characteristics of tonal components.',
        'Percussive Features': 'Characteristics of percussive components.'
    }
    
    print("\nAudio Feature Descriptions:")
    for feature, description in descriptions.items():
        print(f"\n{feature}:")
        print(f"  {description}")

def process_audio_files(base_path, df, 
                        sentiments=['negative', 'neutral', 'positive'], 
                        method='simple', 
                        classification='binary',
                        dataset="EATD",
                        debug_mode=False):
    print(f"Processing inputs: base_path={base_path}, sentiments={sentiments}, method={method}, classification={classification}")
    """Process audio files for given dataframe entries, skipping problematic files.
    Parameters:
        base_path (str): Base path to audio files.
        df (DataFrame): DataFrame containing audio file information.
        sentiments (list): List of sentiments to process.
        method (str): Method to use for feature extraction.
        classification (str): Type of classification to perform. "binary" or "multiclass".
    Returns:
        features_list (list): List of feature vectors.
        labels (list): List of labels.
    """
    features_list = []
    labels = []
    processed_folders = []
    skipped_files = []


    if dataset  == "EATD":
        for _, row in df.iterrows():
            folder = row['folder']
            folder_path = f"{base_path}/{folder}"
            
            # Check if folder exists
            if not os.path.exists(folder_path):
                skipped_files.append((folder, "Folder not found"))
                continue
                
            # Process each type of audio file
            feature_set = []
            all_files_processed = True

            audio_files = [EATD_AUDICO_DICT[sentiment] for sentiment in sentiments]

            for audio_type in audio_files:
                file_path = f"{folder_path}/{audio_type}"
                
                if not os.path.exists(file_path):
                    skipped_files.append((folder, f"Missing file: {audio_type}"))
                    all_files_processed = False
                    continue
                    
                try:
                    if method == 'simple':
                        features = extract_audio_features_simple(file_path)
                        feature_set.extend(features)
                    elif method == 'complex':
                        # features = extract_audio_features_complex(file_path)
                        features = extract_audio_features_best(file_path)

                        feature_set.extend(features)
                    else:
                        raise ValueError(f"Invalid method: {method}")
        
                except Exception as e:
                    skipped_files.append((folder, f"Error processing {audio_type}: {str(e)}"))
                    all_files_processed = False
                    continue
            
            # Only add to dataset if all files were processed successfully
            if all_files_processed:
                features_list.append(feature_set)
                if classification == 'binary':
                    labels.append(row['depression_category_int_binary'])
                elif classification == 'multiclass':
                    labels.append(row['depression_category_int'])
                
                processed_folders.append(folder)
    
    elif dataset == "DAIC":
        feature_set = []
        for _, row in df.iterrows():
            patient = row['Participant_ID']
            depression_category = row['PHQ8_Binary']
            # labels.append(row['PHQ8_Binary'])
            # print(patient)
            # list patient related files in the folder. wav file name starts with the patiuent id
            patient_files = [f for f in os.listdir(base_path) if f.startswith(str(patient)) and f.endswith('.wav')]
            # print(patient_files)
            for f in patient_files:

                file_path = f"{base_path}/{f}"
                if debug_mode:
                    print(f"Processing file: {file_path}")
                try:
                    if method == 'simple':
                        features = extract_audio_features_simple(file_path)
                        # feature_set.extend(features)
                        features_list.append(features)
                        labels.append(depression_category)
                    elif method == 'complex':
                        features = extract_audio_features_complex(file_path)
                        # feature_set.extend(features)
                        features_list.append(features)
                        labels.append(depression_category)
                    else:
                        raise ValueError(f"Invalid method: {method}")
        
                except Exception as e:
                    skipped_files.append((file_path, f"Error processing {audio_type}: {str(e)}"))
                    continue
        
    # Print summary of processing
    print(f"\nProcessing Summary:")
    print(f"Successfully processed: {len(processed_folders)} folders")
    print(f"Skipped: {len(skipped_files)} files")
    
    if skipped_files:
        print("\nSkipped Files Details:")
        for folder, reason in skipped_files:
            print(f"Folder {folder}: {reason}")
    
    if not features_list:
        print("No files were successfully processed!")
        return None, None
    
    return np.array(features_list), np.array(labels)


def feature_extraction_and_save(base_path, df, 
                        sentiments=['negative', 'neutral', 'positive'], 
                        method='simple', 
                        classification='binary',
                        dataset="EATD",
                        debug_mode=False,
                        save_features_per_patient=True,
                        filename_prefix=""):
    print(f"Processing inputs: base_path={base_path}, sentiments={sentiments}, method={method}, classification={classification}")
    """Process audio files for given dataframe entries, skipping problematic files.
    Parameters:
        base_path (str): Base path to audio files.
        df (DataFrame): DataFrame containing audio file information.
        sentiments (list): List of sentiments to process.
        method (str): Method to use for feature extraction.
        classification (str): Type of classification to perform. "binary" or "multiclass".
    Returns:
        features_list (list): List of feature vectors.
        labels (list): List of labels.
    """
    features_list = []
    labels = []
    processed_folders = []
    skipped_files = []


    if dataset  == "EATD":
        for _, row in df.iterrows():
            folder = row['folder']
            folder_path = f"{base_path}/{folder}"
            
            # Check if folder exists
            if not os.path.exists(folder_path):
                skipped_files.append((folder, "Folder not found"))
                continue
                
            # Process each type of audio file
            feature_set = []
            all_files_processed = True

            audio_files = [EATD_AUDICO_DICT[sentiment] for sentiment in sentiments]

            for audio_type in audio_files:
                file_path = f"{folder_path}/{audio_type}"
                
                if not os.path.exists(file_path):
                    skipped_files.append((folder, f"Missing file: {audio_type}"))
                    all_files_processed = False
                    continue
                    
                try:
                    if method == 'simple':
                        features = extract_audio_features_simple(file_path)
                        feature_set.extend(features)
                    elif method == 'complex':
                        # features = extract_audio_features_complex(file_path)
                        features = extract_audio_features_best(file_path)

                        feature_set.extend(features)
                    else:
                        raise ValueError(f"Invalid method: {method}")
        
                except Exception as e:
                    skipped_files.append((folder, f"Error processing {audio_type}: {str(e)}"))
                    all_files_processed = False
                    continue
            
            # Only add to dataset if all files were processed successfully
            if all_files_processed:
                features_list.append(feature_set)
                if classification == 'binary':
                    labels.append(row['depression_category_int_binary'])
                elif classification == 'multiclass':
                    labels.append(row['depression_category_int'])
                
                processed_folders.append(folder)
    
    elif dataset == "DAIC":
        feature_set = []
        prev_patient = ""
        for _, row in df.iterrows():
            patient = row['Participant_ID']
            if prev_patient == "":
                prev_patient = deepcopy(patient)
            else:
                None
            if save_features_per_patient:
                if patient != prev_patient:
                    
                    feature_path = f"{base_path}/{filename_prefix}_features_{method}_{prev_patient}.npy"
                    #print(feature_path)
                    #print(features_list)
                    label_path = f"{base_path}/{filename_prefix}_label_{method}_{prev_patient}.csv"
                    #print(label_path)
                    print(prev_patient)
                    print(labels)
                    if filename_prefix == "best_paper":
                        print(" POIIIIINGGG ")
                        save_audio_features_data_best_paper(features_list, np.array(labels), feature_path, label_path)
                    else:
                        save_audio_features_data(np.array(features_list), np.array(labels), feature_path, label_path)
                    features_list = []
                    labels = []
                    prev_patient = deepcopy(patient)
                    # break
                else:
                    None
            else:
                None
            if classification == 'binary':
                depression_category = row['PHQ8_Binary']
            elif classification == 'multiclass':
                depression_category = row['PHQ8_Score']
            else:
                raise ValueError(f"Invalid classification: {classification}")
            
            # labels.append(row['PHQ8_Binary'])
            # print(patient)
            # list patient related files in the folder. wav file name starts with the patiuent id
            patient_files = [f for f in os.listdir(base_path) if f.startswith(str(patient)) and f.endswith('.wav')]
            # print(patient_files)
            for f in patient_files:

                file_path = f"{base_path}/{f}"
                if debug_mode:
                    print(f"Processing file: {file_path}")
                try:
                    if method == 'simple':
                        features = extract_audio_features_simple(file_path)
                        # feature_set.extend(features)
                        features_list.append(features)
                        labels.append(depression_category)
                    elif method == 'complex':
                        features = extract_audio_features_complex(file_path)
                        # feature_set.extend(features)
                        features_list.append(features)
                        labels.append(depression_category)
                    elif method == 'only_mfcc':
                        features = extract_audio_features_mfcc(file_path)
                        features_list.append(features)
                        labels.append(depression_category)
                    else:
                        raise ValueError(f"Invalid method: {method}")
        
                except Exception as e:
                    skipped_files.append((file_path, f"Error processing: {str(e)}"))
                    continue
        
    # Print summary of processing
    print(f"\nProcessing Summary:")
    print(f"Successfully processed: {len(processed_folders)} folders")
    print(f"Skipped: {len(skipped_files)} files")
    
    if skipped_files:
        print("\nSkipped Files Details:")
        for folder, reason in skipped_files:
            print(f"Folder {folder}: {reason}")
    
    if not features_list:
        print("No files were successfully processed!")
        return None, None
    
    return np.array(features_list), np.array(labels)


