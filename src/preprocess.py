import pandas as pd


def preprocess_data(train_path,test_path):

    print(f"Loading data from {train_path} and {test_path}...")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")

    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
    target = 'SalePrice'

    x_train = train[features]
    y_train = train[target]
    x_test = test[features]

    x_train = x_train.fillna(x_train.median())
    x_test = x_test.fillna(x_test.median())

    return x_train,y_train,x_test


