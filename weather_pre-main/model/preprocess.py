import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from input_target import input_target

def get_preprocessing_pipeline(numeric_cols, categorical_cols):
    """
    Creates a professional Scikit-Learn pipeline for preprocessing.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def preprocess():
    train_inputs, train_target, val_inputs, val_target, test_inputs, test_target = input_target()
    
    numeric_cols = train_inputs.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = get_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # Fit ONLY on training data to prevent data leakage
    train_inputs_processed = preprocessor.fit_transform(train_inputs)
    val_inputs_processed = preprocessor.transform(val_inputs)
    test_inputs_processed = preprocessor.transform(test_inputs)
    
    # Get feature names for the encoded columns
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
    all_feature_names = numeric_cols + list(cat_feature_names)

    # Convert back to DataFrames for easier handling
    train_inputs = pd.DataFrame(train_inputs_processed, columns=all_feature_names)
    val_inputs = pd.DataFrame(val_inputs_processed, columns=all_feature_names)
    test_inputs = pd.DataFrame(test_inputs_processed, columns=all_feature_names)

    return train_inputs, train_target, val_inputs, val_target, test_inputs, test_target


