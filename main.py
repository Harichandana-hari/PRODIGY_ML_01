from src.preprocess import preprocess_data
from src.train_model import train_model
from src.predict import make_predictions



def main():
    
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    output_path = "output/predictions.csv"

    x_train, y_train, x_test = preprocess_data(train_path, test_path)
    print("processed")

    model = train_model(x_train, y_train)
    print("Model training completed")

    make_predictions(model, x_test, output_path)
    print("Pipeline complete")

if __name__ == "__main__":
    main()