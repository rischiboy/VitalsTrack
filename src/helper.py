from typing import List

import numpy as np
import pandas as pd

from scipy.stats import linregress
from scipy.interpolate import CubicSpline, interp1d
from sklearn.preprocessing import StandardScaler

# Stats parameters
WEIGHT = 1  # Weight for the mean
WEIGHT_DT = 0.8  # Weight for the derivative
NUM_CATEGORIES = 4  # Number of age categories
BTM_PTILE = 0.5
TOP_PTILE = 0.5


def list_is_nan(label_list):
    for x in label_list:
        if not np.isnan(x):
            return False
    return True


"""
Categorize patients into age groups in order to impute the missing values by using statistics of the corresponding age group
"""


def categorize(label_list, stat_data):
    age = label_list[0]
    min_age = stat_data["Age_min"].values
    max_age = stat_data["Age_max"].values

    for i in range(len(min_age)):
        if age >= min_age[i] and age <= max_age[i]:
            return [i] + label_list[1:]
        else:
            continue
    return None


"""
Determine the trend of a time series by computing the slope using linear regression
"""


def trend(values: pd.Series) -> float:
    values = values.to_list()

    # No trend if no value exists
    if list_is_nan(values):
        return 0

    valid_values, valid_times = remove_hours_with_nan(values)
    assert len(valid_times) == len(valid_values)

    # No trend if only one value is present
    if len(valid_values) == 1:
        return 0

    # Compute the slope of the linear regression
    slope, _, _, _, _ = linregress(valid_times, valid_values)
    return slope


"""
Remove all nan values from the time series and return the list of valid values and their corresponding measurement hours
"""


def remove_hours_with_nan(values: List) -> tuple:

    valid_hours = []
    valid_values = []

    for i in range(len(values)):
        if np.isnan(values[i]):
            continue

        valid_hours.append(i)
        valid_values.append(values[i])

    return (valid_values, valid_hours)


"""
Fill up nan values with interpolation between existing values and return an weighted average over the list
"""


def interpolated_val(label_list, start_idx, mean, min_val, max_val):
    num_idx = []
    num_vals = []
    derivative = []
    curr_elem = 0

    for i in range(len(label_list)):
        if label_list[i] is np.nan:
            continue
        else:
            num_idx.append(i + start_idx)
            num_vals.append(label_list[i])
            if curr_elem != 0:
                derivative.append(num_vals[curr_elem] - num_vals[curr_elem - 1])
            curr_elem += 1

    max_idx = len(label_list) - 1 + start_idx

    if len(num_idx) <= 3:
        result = WEIGHT * np.mean(num_vals) + (1 - WEIGHT) * mean
        return result
    elif (max_idx - num_idx[-1]) > 2:
        # max_deviation = (np.max(num_vals) - np.min(num_vals))
        mean_vals = np.mean(num_vals)

        estimated_val = num_vals[-1] + (
            WEIGHT_DT * derivative[-1] + (1 - WEIGHT_DT) * np.mean(derivative[0:-1])
        )
        estimated_val = (
            max(min_val, estimated_val)
            if (estimated_val < min_val)
            else min(max_val, estimated_val)
        )

        # estimated_val =  min(max_val,random.uniform(0,max_deviation) + mean_vals) if (mean_vals < mean) else max(min_val,random.uniform(-max_deviation,0) + mean_vals)
        num_idx.append(max_idx)
        num_vals.append(estimated_val)

        interpolate_func = interp1d(
            num_idx, num_vals, kind="linear", fill_value="extrapolate"
        )
        # interpolate_func = CubicSpline(num_idx,num_vals)
        reg_x = np.linspace(start=start_idx, stop=max_idx, num=len(label_list))
        reg_y = [interpolate_func(x) for x in reg_x]

        """ regression = Ridge(alpha=reg_lambda).fit(reg_x.reshape((-1,1)),reg_y)
        idx = np.array([max_idx+1])
        predicted_val = regression.predict(idx.reshape((-1,1)))[0]                        #Predict for the 13th hour
        #predicted_val = interpolate_func(max_idx+1)
        """
        predicted_val = np.median(reg_y)
        if predicted_val < min_val:
            predicted_val = WEIGHT * min_val + (1 - WEIGHT) * mean_vals
        if predicted_val > max_val:
            predicted_val = WEIGHT * max_val + (1 - WEIGHT) * mean_vals
        return predicted_val

    else:
        interpolate_func = interp1d(
            num_idx, num_vals, kind="linear", fill_value="extrapolate"
        )
        # interpolate_func = CubicSpline(num_idx,num_vals)
        reg_x = np.linspace(start=start_idx, stop=max_idx, num=len(label_list))
        reg_y = [interpolate_func(x) for x in reg_x]

        """predicted_val = interpolate_func(max_idx+1)
        regression = Ridge(alpha=reg_lambda).fit(reg_x.reshape((-1,1)),reg_y)
        idx = np.array([max_idx+1])
        predicted_val = regression.predict(idx.reshape((-1,1)))[0]  
        """
        mean_vals = np.mean(num_vals)

        predicted_val = np.median(reg_y)

        if predicted_val < min_val:
            predicted_val = WEIGHT * min_val + (1 - WEIGHT) * mean_vals
        if predicted_val > max_val:
            predicted_val = WEIGHT * max_val + (1 - WEIGHT) * mean_vals

        return predicted_val


def standardize_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame = None, drop_cols: list = None
) -> pd.DataFrame:
    normalizer = StandardScaler()

    if drop_cols:
        train_data = train_data.drop(drop_cols, axis=1)
        if test_data is not None:
            test_data = test_data.drop(drop_cols, axis=1)

    norm_train_data = normalizer.fit_transform(train_data)

    if test_data is not None:
        norm_test_data = normalizer.transform(test_data)
        return norm_train_data, norm_test_data
    else:
        return norm_train_data, None
