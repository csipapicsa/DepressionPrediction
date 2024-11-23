import numpy as np
import pandas as pd
from scipy.stats import mode  # Option 1: Using scipy


def phq8_to_multiclass(phq_scores):
    """
    Convert PHQ-8 scores to multiclass labels.
    
    Parameters:
    - phq_scores (np.array): An array of PHQ-8 scores.
    
    Returns:
    - np.array: An array of multiclass labels corresponding to depression severity.
    """
    labels = np.zeros_like(phq_scores, dtype=int)  # Default to 'Normal'
    
    labels[(phq_scores >= 5) & (phq_scores <= 9)] = 1  # Mild
    labels[(phq_scores >= 10) & (phq_scores <= 14)] = 2  # Moderate
    labels[(phq_scores >= 15) & (phq_scores <= 19)] = 3  # Moderately severe
    labels[(phq_scores >= 20)] = 4  # Severe
    
    return labels

def phq8_to_binary(phq_scores):
    """
    Convert PHQ-8 scores to binary labels.

    Parameters:
    - phq_scores (np.array): An array of PHQ-8 scores.

    Returns:
    - np.array: An array of binary labels (0 or 1).
    """
    labels = np.zeros_like(phq_scores, dtype=int)  # Default to 'Normal'

    labels[(phq_scores >= 10)] = 1  # Mild or higher

    return labels


def load_and_organize_features(feature_path):
    """
    For one patient all features
    Load features from .npy file and organize them back into their original structure.
    The features were saved in a specific order in the feature vector.
    """
    # Load the saved features
    features_array = np.load(feature_path)
    
    # Create a list to hold features for each audio chunk
    organized_features = []
    
    # For each chunk's feature vector
    for feature_vector in features_array:
        # Initialize index for slicing the feature vector
        idx = 0
        
        # Reconstruct the features dictionary
        # MFCC means and stds (40 coefficients each)
        mfcc_mean = feature_vector[idx:idx+40]
        idx += 40
        mfcc_std = feature_vector[idx:idx+40]
        idx += 40
        
        # Single value features
        features_dict = {
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'spectral_centroid_mean': feature_vector[idx],
            'spectral_bandwidth_mean': feature_vector[idx + 1],
            'spectral_rolloff_mean': feature_vector[idx + 2],
            'zero_crossing_rate_mean': feature_vector[idx + 3],
            'pitch': feature_vector[idx + 4],
            'energy': feature_vector[idx + 5],
            'speech_rate': feature_vector[idx + 6]
        }
        
        organized_features.append(features_dict)
    
    return organized_features


def create_feature_summary_df(organized_features):
    """
    Create a DataFrame with min, max, and avg values for all features across chunks.
    """
    # First collect all values for each feature
    feature_values = {
        'mfcc_mean': np.array([chunk['mfcc_mean'] for chunk in organized_features]),
        'mfcc_std': np.array([chunk['mfcc_std'] for chunk in organized_features]),
        'spectral_centroid': np.array([chunk['spectral_centroid_mean'] for chunk in organized_features]),
        'spectral_bandwidth': np.array([chunk['spectral_bandwidth_mean'] for chunk in organized_features]),
        'spectral_rolloff': np.array([chunk['spectral_rolloff_mean'] for chunk in organized_features]),
        'zero_crossing_rate': np.array([chunk['zero_crossing_rate_mean'] for chunk in organized_features]),
        'pitch': np.array([chunk['pitch'] for chunk in organized_features]),
        'energy': np.array([chunk['energy'] for chunk in organized_features]),
        'speech_rate': np.array([chunk['speech_rate'] for chunk in organized_features])
    }
    
    # Create a dictionary to store all features
    all_features = {}
    
    # Handle MFCC coefficients (40 of them)
    for i in range(40):
        # For MFCC means
        all_features[f'mfcc_mean_{i}_min'] = np.min(feature_values['mfcc_mean'][:, i])
        all_features[f'mfcc_mean_{i}_max'] = np.max(feature_values['mfcc_mean'][:, i])
        all_features[f'mfcc_mean_{i}_avg'] = np.mean(feature_values['mfcc_mean'][:, i])
        
        # For MFCC standard deviations
        all_features[f'mfcc_std_{i}_min'] = np.min(feature_values['mfcc_std'][:, i])
        all_features[f'mfcc_std_{i}_max'] = np.max(feature_values['mfcc_std'][:, i])
        all_features[f'mfcc_std_{i}_avg'] = np.mean(feature_values['mfcc_std'][:, i])
    
    # Handle other features
    for feature in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                   'zero_crossing_rate', 'pitch', 'energy', 'speech_rate']:
        all_features[f'{feature}_min'] = np.min(feature_values[feature])
        all_features[f'{feature}_max'] = np.max(feature_values[feature])
        all_features[f'{feature}_avg'] = np.mean(feature_values[feature])
    
    # Create DataFrame with a single row
    df = pd.DataFrame([all_features])
    
    return df

def aggregate_patient_chunks(features_list, labels, chunk_size=10):
    """
    Aggregate features from multiple chunks for each patient.
    
    Args:
        features_list: List of feature dictionaries
        labels: Corresponding labels
        chunk_size: Number of chunks to aggregate
    """
    n_chunks = len(features_list)
    n_groups = n_chunks // chunk_size
    
    aggregated_features = []
    aggregated_labels = []
    
    for i in range(n_groups):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_group = features_list[start_idx:end_idx]
        
        # Aggregate MFCC features
        mfcc_means = np.array([chunk['mfcc_mean'] for chunk in chunk_group])
        mfcc_stds = np.array([chunk['mfcc_std'] for chunk in chunk_group])
        
        # Calculate statistics over the chunks
        group_features = {
            # Mean and std of MFCC means across chunks
            'mfcc_mean_mean': np.mean(mfcc_means, axis=0),
            'mfcc_mean_std': np.std(mfcc_means, axis=0),
            # Mean and std of MFCC stds across chunks
            'mfcc_std_mean': np.mean(mfcc_stds, axis=0),
            'mfcc_std_std': np.std(mfcc_stds, axis=0),
            # Other features averaged across chunks
            'spectral_centroid_mean': np.mean([chunk['spectral_centroid_mean'] for chunk in chunk_group]),
            'spectral_bandwidth_mean': np.mean([chunk['spectral_bandwidth_mean'] for chunk in chunk_group]),
            'spectral_rolloff_mean': np.mean([chunk['spectral_rolloff_mean'] for chunk in chunk_group]),
            'zero_crossing_rate_mean': np.mean([chunk['zero_crossing_rate_mean'] for chunk in chunk_group]),
            'pitch_mean': np.mean([chunk['pitch'] for chunk in chunk_group]),
            'energy_mean': np.mean([chunk['energy'] for chunk in chunk_group]),
            'speech_rate_mean': np.mean([chunk['speech_rate'] for chunk in chunk_group])
        }
        
        # Flatten the features into a single vector
        feature_vector = np.concatenate([
            group_features['mfcc_mean_mean'],
            group_features['mfcc_mean_std'],
            group_features['mfcc_std_mean'],
            group_features['mfcc_std_std'],
            [group_features['spectral_centroid_mean']],
            [group_features['spectral_bandwidth_mean']],
            [group_features['spectral_rolloff_mean']],
            [group_features['zero_crossing_rate_mean']],
            [group_features['pitch_mean']],
            [group_features['energy_mean']],
            [group_features['speech_rate_mean']]
        ])
        
        aggregated_features.append(feature_vector)
        # Take the most common label in the chunk group
        aggregated_labels.append(mode(labels[start_idx:end_idx]))
    
    return np.array(aggregated_features), np.array(aggregated_labels)


