from flask import Flask, render_template, request
import pickle
import gzip
import numpy as np

app = Flask(__name__)

# Load the model and scaler
def load_model():
    with gzip.open('model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()


scaler = pickle.load(open('Scaler .pkl', 'rb'))

# Encoding dictionaries
mszoning_dict = {'RL': 0, 'RM': 1, 'FV': 2, 'RH': 3, 'C (all)': 4}
lotconfig_dict = {'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4}
bldgtype_dict = {'1Fam': 0, '2fmCon': 1, 'Duplex': 2, 'TwnhsE': 3, 'Twnhs': 4}
exterior1st_dict = {
    'AsphShn': 0, 'AsbShng': 1, 'BrkComm': 2, 'BrkFace': 3, 'CBlock': 4,
    'CemntBd': 5, 'HdBoard': 6, 'ImStucc': 7, 'MetalSd': 8, 'Plywood': 9,
    'Stone': 10, 'Stucco': 11, 'VinylSd': 12, 'Wd Sdng': 13, 'WdShing': 14
}

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # Retrieve form data
    data = [
        int(request.form['MSSubClass']),
        mszoning_dict[request.form['MSZoning']],
        float(request.form['LotArea']),
        lotconfig_dict[request.form['LotConfig']],
        bldgtype_dict[request.form['BldgType']],
        int(request.form['OverallCond']),
        int(request.form['YearBuilt']),
        int(request.form['YearRemodAdd']),
        exterior1st_dict[request.form['Exterior1st']],
        float(request.form['BsmtFinSF2']),
        float(request.form['TotalBsmtSF'])
    ]
    
    # Convert to numpy array and scale the data
    input_data = np.array([data])
    scaled_data = scaler.transform(input_data)
    
    # Predict using the pre-trained model
    prediction = model.predict(scaled_data)
    
    # Return the result to result.html
    return render_template('result.html', prediction=round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)
