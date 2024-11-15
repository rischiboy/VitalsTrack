"""
Group the time data of one patient together and save it in a separate csv file
where each entry is a list of values of the corresponding column belonging to a certain patient
"""

import ast
import math
import os
import random
from typing import List

import numpy as np
import pandas as pd
import yaml

from scipy.stats import linregress
from scipy.interpolate import CubicSpline, interp1d
from sklearn.preprocessing import StandardScaler

from helper import *

"""
Compute the mean and standard deviation of all columns grouped by their age group
"""


# def generate_age_group_stats(input_data, file):
#     min_age, max_age = input_data["Age"].min(), input_data["Age"].max()
#     size_of_cat = math.floor((max_age - min_age) / NUM_CATEGORIES) - 1
#     cat_name_prefix = "cat_"
#     cat_names = []
#     cat_min_boundries = []
#     cat_max_boundries = []

#     for i in range(NUM_CATEGORIES):
#         cat_name_i = cat_name_prefix + str(i + 1)
#         cat_names.append(cat_name_i)
#         if not cat_max_boundries:
#             cat_min_boundries.append(min_age)
#             cat_max_boundries.append(min_age + size_of_cat)
#         else:
#             temp = cat_max_boundries[-1] + 1
#             cat_min_boundries.append(temp)
#             cat_max_boundries.append(temp + size_of_cat)

#     cat_max_boundries[-1] = max_age
#     col_labels = input_data.columns.values.tolist()[3:]
#     cat_statistics = pd.DataFrame(
#         data={
#             "Category": cat_names,
#             "Age_min": cat_min_boundries,
#             "Age_max": cat_max_boundries,
#         }
#     )

#     for label in col_labels:
#         mean_label = "mean_" + label
#         std_label = "std_" + label
#         avg_values = []
#         std_values = []
#         for i in range(NUM_CATEGORIES):
#             cat_i = input_data[
#                 (input_data["Age"] >= cat_min_boundries[i])
#                 & (input_data["Age"] <= cat_max_boundries[i])
#             ]
#             mean_cat_i = cat_i[label].mean()
#             std_cat_i = cat_i[label].std()
#             avg_values.append(mean_cat_i)
#             std_values.append(std_cat_i)
#         cat_statistics[mean_label] = avg_values
#         cat_statistics[std_label] = std_values

#     cat_statistics.to_csv(file, index=False, float_format="%.3f")
#     return


def generate_age_group_stats(input_data: pd.DataFrame, outfile: str):
    cat_labels = [f"cat_{i+1}" for i in range(len(AGE_BINS) - 1)]
    vitals = input_data.columns.values.tolist()[3:]

    # Extract Age_min and Age_max from bins
    age_ranges = pd.DataFrame(
        {
            "Category": cat_labels,
            "Age_min": AGE_BINS[:-1],  # All but the last bin edge
            "Age_max": AGE_BINS[1:],  # All but the first bin edge
        }
    )

    # Segment the data into age groups
    input_data["Category"] = pd.cut(
        input_data["Age"], bins=AGE_BINS, labels=cat_labels, right=True
    )

    # Aggregate the data by age group to obtain age relevant statistics
    vital_stats = {}

    for vital in vitals:
        for agg in AGG_FUNCTIONS:
            vital_stats[f"{agg}_{vital}"] = (vital, agg)

    # Perform aggregation
    stats = input_data.groupby("Category").agg(**vital_stats).reset_index()

    # Merge the age ranges with the statistics
    stats = pd.merge(age_ranges, stats, on="Category")

    stats.to_csv(outfile, index=False, float_format="%.3f")

    return


"""
Represent series of patient data as a single value using weighted average.
Missing values are imputed using age group statistics.
"""


def impute_data(features: pd.DataFrame, stat_data: pd.DataFrame) -> pd.DataFrame:
    print("-----Imputing missing data-----")

    cat_labels = stat_data["Category"].values
    ft_names = features.columns.values.tolist()[2:]
    ft_names = [ft for ft in ft_names if "trend" not in ft]

    features["Category"] = pd.cut(features["Age"], bins=AGE_BINS, labels=cat_labels)

    # Merge the features with the statistics on Category
    merged_features = pd.merge(features, stat_data, on="Category", how="left")

    for ft in ft_names:
        merged_features[ft] = merged_features[ft + "_x"].fillna(
            merged_features[ft + "_y"]
        )

    # Drop the redundant columns
    redundant_cols = (
        [col + "_x" for col in ft_names]
        + [col + "_y" for col in ft_names]
        + ["Category", "Age_min", "Age_max"]
    )
    merged_features.drop(columns=redundant_cols, inplace=True)

    print("-----Finished imputing data-----")

    return merged_features


# def min_max_impute(group_data, stat_data, out_file, quantile_range):
#     print("-----Min-Max Imputation of Data-----")
#     col_labels = group_data.columns.values.tolist()
#     imputed_df = pd.DataFrame()
#     imputed_df["pid"] = group_data["pid"]
#     imputed_df["Start_time"] = group_data["Time"].apply(lambda x: x[0])
#     imputed_df["Age"] = group_data["Age"].apply(lambda x: x[0])

#     new_cols = {}

#     for label in col_labels[3:]:
#         min_label = "Min_" + label
#         max_label = "Max_" + label
#         trend_label = "Trend_" + label

#         min_value, max_value = quantile_range[label]

#         min_col = group_data[label].apply(
#             lambda l: (
#                 min_value
#                 if (list_is_nan(l))
#                 else min([x for x in l if not math.isnan(x)])
#             )
#         )
#         max_col = group_data[label].apply(
#             lambda l: (
#                 max_value
#                 if (list_is_nan(l))
#                 else max([x for x in l if not math.isnan(x)])
#             )
#         )
#         trend_col = group_data[label].apply(
#             lambda l: np.nan if (list_is_nan(l)) else trend(l)
#         )

#         new_cols.update(
#             {min_label: min_col, max_label: max_col, trend_label: trend_col}
#         )

#         print("Imputed: " + label)

#     imputed_df = pd.concat([imputed_df, pd.DataFrame(new_cols)], axis=1)

#     # Fill imputed trend values with mean
#     imputed_df = imputed_df.apply(pd.to_numeric, errors="coerce")
#     imputed_df.fillna(imputed_df.mean(), inplace=True)

#     imputed_df.to_csv(out_file, index=False, float_format="%.3f")

#     return None


# Data Loading and File Creation

"""
Group the time data of one patient together and save it in a separate csv file
"""


def group_series_data(features: pd.DataFrame, output_file: str) -> None:
    patient_data = pd.read_csv(features)
    column_attr = patient_data.columns.values.tolist()

    column_attr.pop(0)  # Remove PID
    first = True
    grouped_frame = pd.DataFrame
    for label in column_attr:
        if first:
            grouped_frame = (
                patient_data.groupby(["pid"], sort=False)[label]
                .apply(list)
                .reset_index(name=label)
            )
            first = False
        else:
            temp_frame = (
                patient_data.groupby(["pid"], sort=False)[label]
                .apply(list)
                .reset_index(name=label)
            )
            grouped_frame = grouped_frame.join(temp_frame.set_index("pid"), on="pid")

    grouped_frame.to_csv(output_file, index=False)

    return


"""
Load the grouped data as a DataFrame from the csv file
"""


def load_group_data(group_data_file: str, row_limit: int = None) -> pd.DataFrame:
    def parse_func(x):
        return list(map(lambda y: np.nan if (y == "nan") else ast.literal_eval(y), x))

    if row_limit:
        df = pd.read_csv(group_data_file, nrows=row_limit)
    else:
        df = pd.read_csv(group_data_file)

    col_labels = df.columns.values.tolist()
    col_labels.pop(0)

    for col in col_labels:
        df[col] = df[col].apply(
            lambda x: parse_func(x.replace("[", "").replace("]", "").split(", "))
        )

    return df


"""
Extract meaningful features from the time series data.
Impute missing values using the statistics of the corresponding age group.
"""


def build_features(
    data: pd.DataFrame, stat_data: pd.DataFrame, out_file: str
) -> pd.DataFrame:
    col_labels = data.columns.values.tolist()
    grouped_data = data.groupby(["pid"], dropna=False, sort=False)

    features = {}

    for label in col_labels[2:]:
        if label == "Age":
            features[label] = (label, "mean")
        else:
            features["min_" + label] = (label, "min")
            features["max_" + label] = (label, "max")

            features["mean_" + label] = (label, "mean")
            features["std_" + label] = (label, "std")

            features["trend_" + label] = (label, trend)

    grouped_data = grouped_data.agg(**features).reset_index()

    imputed_features = impute_data(grouped_data, stat_data)
    imputed_features.to_csv(out_file, index=False)

    return


def generate_files_from_data(
    features_file: str,
    group_data_file: str,
    stat_file: str,
    imputed_file: str,
    min_max_imputed_file: str,
    final_features_file: str,
) -> None:

    # if not os.path.isfile(group_data_file):
    #     group_series_data(features_file, group_data_file)

    if not os.path.isfile(stat_file):
        df = pd.read_csv(features_file)
        generate_age_group_stats(df, stat_file)

    # if not os.path.isfile(imputed_file) or not os.path.isfile(min_max_imputed_file):
    #     min_col_vals = {}
    #     max_col_vals = {}
    #     quantile_range = {}

    #     df_training = pd.read_csv(features_file)
    #     col_labels = df_training.columns.values.tolist()[3:]

    #     # df_training = df_training.head(100)  # For testing purposes

    #     for label in col_labels:
    #         min_col_vals[label] = df_training[label].min()
    #         max_col_vals[label] = df_training[label].max()
    #         bottom = df_training[label].quantile(BTM_PTILE)
    #         top = df_training[label].quantile(TOP_PTILE)
    #         quantile_range[label] = (bottom, top)

    #     parsed_group_data = load_group_data(group_data_file)
    #     # parsed_group_data = load_group_data(
    #     #     group_data_file, row_limit=100
    #     # )  # For testing purposes
    #     stat_data = pd.read_csv(stat_file)

    #     if not os.path.isfile(imputed_file):
    #         impute_data(
    #             parsed_group_data,
    #             stat_data,
    #             imputed_file,
    #             min_col_vals,
    #             max_col_vals,
    #             quantile_range,
    #         )

    #     if not os.path.isfile(min_max_imputed_file):
    #         min_max_impute(
    #             parsed_group_data, stat_data, min_max_imputed_file, quantile_range
    #         )

    features = pd.read_csv(features_file)
    stat_data = pd.read_csv(stat_file)

    build_features(data=features, stat_data=stat_data, out_file=final_features_file)

    return


def main():

    # File paths
    with open("config/paths.yaml", "r") as file:
        paths = yaml.safe_load(file)

    # print(paths["train"])

    # Generate files for training data
    train_args = {
        "features_file": paths["train"]["train_features"],
        "group_data_file": paths["train"]["group_data_file"],
        "stat_file": paths["train"]["stat_file"],
        "imputed_file": paths["train"]["imputed_file"],
        "min_max_imputed_file": paths["train"]["min_max_imputed_file"],
        "final_features_file": paths["train"]["final_train_file"],
    }

    generate_files_from_data(**train_args)

    # Generate files for test data
    test_args = {
        "features_file": paths["test"]["test_features"],
        "group_data_file": paths["test"]["test_group_data_file"],
        "stat_file": paths["test"]["test_stat_file"],
        "imputed_file": paths["test"]["test_imputed_file"],
        "min_max_imputed_file": paths["test"]["test_min_max_imputed_file"],
        "final_features_file": paths["test"]["final_test_file"],
    }

    generate_files_from_data(**test_args)


if __name__ == "__main__":

    # Stats parameters
    NUM_CATEGORIES = 4  # Number of age categories
    BTM_PTILE = 0.5
    TOP_PTILE = 0.5

    AGE_BINS = [14, 30, 50, 70, 100]
    AGG_FUNCTIONS = ["min", "max", "mean", "std"]

    main()
