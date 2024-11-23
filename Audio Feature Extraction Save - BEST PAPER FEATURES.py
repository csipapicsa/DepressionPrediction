import pandas as pd
#dev
import importlib as imp
import functions
from functions import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_PATH = "datasets/DAIC-WOZ/"
files = os.listdir(BASE_PATH+"/cuts")
df_train = pd.read_csv(BASE_PATH + "train_split_Depression_AVEC2017.csv")
df_dev = pd.read_csv(BASE_PATH + "dev_split_Depression_AVEC2017.csv")

df_dev = df_dev[['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score']]
df_train = df_train[['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score']]

# DEV
# df_dev = df_dev[0:3]
# TRAIN
# df_train = df_train[0:3]
imp.reload(functions)
from functions import *
sentiments=['negative', 'neutral', 'positive']

BASE_PATH

feature_extraction_and_save(BASE_PATH+"cuts", df_train, [], 
                                               method="only_mfcc", classification="multiclass", dataset="DAIC", 
                                               debug_mode=False, save_features_per_patient=True, filename_prefix="best_paper")

feature_extraction_and_save(BASE_PATH+"cuts", df_dev, [], 
                                               method="only_mfcc", classification="multiclass", dataset="DAIC", 
                                               debug_mode=False, save_features_per_patient=True, filename_prefix="best_paper")
