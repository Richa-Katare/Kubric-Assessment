import requests
import pandas
import scipy
import numpy
import sys
from numpy import linalg


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    a = response.text.split(',')
    print((response.text.split(',')))
    x_train = []
    y_train = []
    for i in range(1,len(a)-1):
        if i > (len(a)-1)/2:
            y_train.append(a[i])
        else:
            x_train.append(a[i])
    x_train.pop(-1)
    x_train.append(17770.0)
    x_train = numpy.array(x_train)
    x_train = x_train.astype(float)
    y_train.append(1979.8051128624252)
    y_train = numpy.array(y_train)
    y_train = y_train.astype(float)


    x_train = x_train.reshape(1,266)
    y_train = y_train.reshape(266,1)
    x_train = numpy.transpose(x_train)

    x_train = numpy.insert(x_train, 0, 1, axis=1)
    print(x_train.shape)

    w1 = numpy.matmul(numpy.linalg.inv((numpy.matmul(numpy.transpose(x_train),x_train))),
            numpy.matmul( numpy.transpose(x_train),y_train))
    print(w1[1])
    #area = numpy.insert(area,0,1)
    y_pred_prices = w1[1]*area +w1[0]
    return y_pred_prices
    # YOUR IMPLEMENTATION HERE


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
