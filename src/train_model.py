import pickle
from sklearn.linear_model import LinearRegression

def train_model(x,y):
    
    model = LinearRegression()
    model.fit(x,y)
    with open('output/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model
