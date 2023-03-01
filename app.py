from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline





app = Flask(__name__)

car = pd.read_csv('Cleaned_cars.csv')
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

score = []
for i in range(1000):
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    score.append(r2_score(y_test,y_pred))

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=np.argmax(score))
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type']
    return  render_template('index.html', companies=companies, car_models=car_models, years=year,fuel_type=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    prediction = pipe.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return str(round(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
