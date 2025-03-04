import json
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import census_app

# test Fast API root


def test_api_locally_get_root():
    """ Test Fast API root route"""

    with TestClient(census_app) as client:
        r = client.get("/")
    # bs = BeautifulSoup(r.content, "html.parser")
    # res = bs.find("h1").get_text()

    assert r.status_code == 200
    assert r.json() == {"hello": "word"}


def test_api_locally_get_predictions_inf1():
    """ Test Fast API predict route with a '<=50K' salary prediction result """

    expected_res = "Predicts ['<=50K']"
    test_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    headers = {"Content-Type": "application/json"}

    with TestClient(census_app) as client:
        r = client.post("/predict", data=json.dumps(test_data),
                        headers=headers)
        assert r.status_code == 200
        assert (r.json()["predict"][: len(expected_res)]) == expected_res


def test_api_locally_get_predictions_inf2():
    """ Test Fast API predict route with a '>50K' salary prediction result """

    expected_res = "Predicts ['>50K']"
    test_data = {
        "age": 40,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 20000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    headers = {"Content-Type": "application/json"}

    with TestClient(census_app) as client:
        r = client.post("/predict", data=json.dumps(test_data),
                        headers=headers)
        assert r.status_code == 200
        assert (r.json()["predict"][: len(expected_res)]) == expected_res
