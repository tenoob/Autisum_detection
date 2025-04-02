from sklearn.ensemble import RandomForestClassifier
import pickle

def load_rf():
    rf = 'model/model.pkl'
    with open(rf, 'rb') as file:
        rf_model = pickle.load(file)

    print("done")
    return rf_model