from input_target import input_target
from data_loader import load_raw_data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os




def preprocess():
    df= load_raw_data()
    
    train_inputs, train_target, val_inputs, val_target, test_inputs, test_target = input_target()
    
    numeric_cols = train_inputs.select_dtypes(include=['number']).columns
    categorical_cols = train_inputs.select_dtypes(include=['object', 'category']).columns
    
    # Fill missing values for numeric columns with median{imputing}
    
    imputer = SimpleImputer(strategy='median').fit(df[numeric_cols])
    
    train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
    test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])
    
    #scaling numeric cols
    
    scaler = MinMaxScaler().fit(df[numeric_cols])
    
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols]= scaler.transform(val_inputs[numeric_cols])
    test_inputs[numeric_cols]= scaler.transform(test_inputs[numeric_cols])
    
    # dealing with categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df[categorical_cols])
    
    # Get encoded column names
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

# Transform categorical columns and create DataFrames
    train_encoded = pd.DataFrame(encoder.transform(train_inputs[categorical_cols]),
                             columns=encoded_cols, index=train_inputs.index)
    val_encoded   = pd.DataFrame(encoder.transform(val_inputs[categorical_cols]),
                             columns=encoded_cols, index=val_inputs.index)
    test_encoded  = pd.DataFrame(encoder.transform(test_inputs[categorical_cols]),
                             columns=encoded_cols, index=test_inputs.index)

# Drop original categorical columns
    train_inputs = train_inputs.drop(columns=categorical_cols)
    val_inputs   = val_inputs.drop(columns=categorical_cols)
    test_inputs  = test_inputs.drop(columns=categorical_cols)

# Concatenate encoded columns
    train_inputs = pd.concat([train_inputs, train_encoded], axis=1)
    val_inputs   = pd.concat([val_inputs, val_encoded], axis=1)
    test_inputs  = pd.concat([test_inputs, test_encoded], axis=1)

    return train_inputs, val_inputs, test_inputs, imputer, scaler, encoder, numeric_cols, categorical_cols,encoded_cols



