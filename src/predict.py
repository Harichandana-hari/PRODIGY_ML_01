import pandas as pd
import pickle

def make_predictions(model, x_test, output_path):

    
    with open('output/model.pkl', 'rb') as file:
        model = pickle.load(file)
    predictions = model.predict(x_test)
    output = pd.DataFrame({'Id' : x_test.index, 'SalePrice' : predictions})
    output.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
