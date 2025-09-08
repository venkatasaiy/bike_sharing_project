import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the bike-sharing dataset.
    
    Parameters:
    - file_path (str): Path to the CSV dataset.
    
    Returns:
    - X_train_preprocessed (array): Preprocessed training features.
    - X_test_preprocessed (array): Preprocessed testing features.
    - y_train (Series): Training target values.
    - y_test (Series): Testing target values.
    - preprocessor (ColumnTransformer): Fitted preprocessor for reuse.
    """
    # Load dataset from CSV file
    df = pd.read_csv(file_path)
    
    # Parse datetime column and extract time-related features
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Define feature categories for preprocessing
    categorical_cols = ['season', 'weather', 'hour', 'dayofweek', 'month']
    numerical_cols = ['temp', 'humidity', 'windspeed']
    binary_cols = ['holiday', 'workingday']
    
    # Prepare features (X) and target (y)
    X = df.drop(['count', 'datetime'], axis=1)
    y = df['count']
    
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define preprocessing pipeline for different feature types
    preprocessor = ColumnTransformer(
        transformers=[
            # One-hot encode categorical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            # Scale numerical features
            ('num', StandardScaler(), numerical_cols),
            # Pass binary features unchanged
            ('bin', 'passthrough', binary_cols)
        ]
    )
    
    # Fit preprocessor on training data and transform both training and test data
    X_train_preprocessed = preprocessor.fit_transform(X_train).toarray()
    X_test_preprocessed = preprocessor.transform(X_test).toarray()
    
    # Return preprocessed data and fitted preprocessor
    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor

if __name__ == "__main__":
    # Test the module with sample data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("data/bike_sharing.csv")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")