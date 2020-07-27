from app import app, model
import pytest
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 


def test_home_page():
  response = app.test_client().get('/')
  assert response.status_code == 200

def test_prediction():

    # Making prediction on saved model
    pred = [1,0,0,0.835,1,40.92]

    prediction = model.predict([pred])[0]
    assert prediction == 0
