import pandas as pd
import numpy as np
import pytest

# import pdb
from sklearn.compose import ColumnTransformer
from .model import generate_feature_encoding

@pytest.fixture
def data():
    """ Retrieve Cleaned Dataset """
    train_file = "starter/data/census_cleaned.csv"
    df = pd.read_csv(train_file)
    df = df.iloc[:, :-1]  # exclude label
    return df

def test_data_char_cleaned(data):
    """ Check that there are no ? characters in the categorical features """
    cat_col = data.select_dtypes(include=[object]).columns
    for col in cat_col:
        filt = data[col] == "?"
        assert filt.sum() == 0, f"Found ? character in feature {col}"


def test_data_column_name_cleaned(data):
    """ Check that there are no spaces in the column names """
    col_names = data.columns
    for col in col_names:
        assert " " not in col, f"Found space character in feature {col}"


def test_one_generate_feature_encoding(data):
    """ Check that the feature encoding column transformer object is created """
    num_vars = data.select_dtypes(include=np.number).columns
    cat_vars = data.select_dtypes(include=[object]).columns

    ct = generate_feature_encoding(data, cat_vars=cat_vars, num_vars=num_vars)

    assert isinstance(
        ct, ColumnTransformer
    ), "generate_feature_encoding returned wrong type!"
