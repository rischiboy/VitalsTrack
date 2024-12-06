# Medical Data Prediction: Classification and Regression

## Task Description

This task involves analyzing medical data collected from hospitalized patients, specifically focusing on predicting both binary pathology labels and continuous health indicators. The dataset consists of time-series data for 12 vital measurements recorded hourly over the first 12 hours of hospitalization. These measurements include metrics such as EtCO2 (end-tidal carbon dioxide), BaseExcess, and PTT (partial thromboplastin time), among others.

In addition to the time-series data, we are provided with:

- **Binary Labels**: Denoting whether a patient developed a certain pathology within the following 12 hours (pathology indicators).
- **Health Indicators**: These include various metrics, such as average blood pressure, which need to be predicted through regression models.

### Problem Setting

The core challenge of this task lies in the **sparse nature** of the data. Most medical tests are not performed every hour, and not every patient receives all measurements during the first 12 hours of hospitalization. This results in a dataset with significant **missing values**, which complicates model training.

Additionally, the dataset is **highly imbalanced**, with some patients having only a few measurements or data points for certain tests. We need to predict labels and health indicator values specifically for the **13th hour**, based on the data from the preceding 12 hours.

### Goal

The objective is to:

1. **Train classification models** to predict whether a patient will be affected by specific pathologies within the following 12 hours, based on the measurements taken during the first 12 hours.

2. **Train regression models** to predict the values of key health indicators, such as average blood pressure, at the 13th hour.

The challenge is to effectively handle the missing values and generate meaningful features that can optimize the performance of the models for both classification and regression tasks.

## Approach

### Handling Missing Data

To address the challenge of missing values, we applied the following strategies:

1. **Summary Statistics**: For every patient, we calculated a collection of summary statistics from the available measurements, such as the empirical mean, standard deviation, and trends (e.g., rate of change) over the first 12 hours. This helps generate robust features for patients with incomplete data.

2. **Imputation**: For patients with sparse data (i.e., measurements taken only once or twice), we used **age group statistics** for imputation. Specifically, we categorized patients into **4 age groups** (k=4) and computed the mean, standard deviation, minimum, and maximum values for each label within each age group. These imputed values were then used to fill in missing data points.

3. **Feature Engineering**: We designed a set of meaningful features based on the available data, such as:
   - Trend analysis of vital measurements over the 12-hour period.
   - Statistical measures (mean, stddev) for each vital measurement.
   - Imputed values for sparse data based on age group statistics.

### Feature Selection

To improve model performance and reduce overfitting, we applied feature selection techniques:

- **For Classification**: We used feature importance scores derived from **Random Forests** to evaluate and select the most significant features. This helped in reducing the number of features and focusing on the most informative ones for predicting the pathology labels.
  
- **For Regression**: We used **Mutual Information** as a metric to evaluate the relevance of each feature to the target health indicators. This allowed us to identify and retain the most informative features, improving model accuracy and efficiency.

### Model Selection

After preprocessing and feature engineering, we tested various machine learning models through cross-validation to evaluate their performance:

- **Classification Models (for Pathology Labels)**: We applied **Random Forests** to predict the binary pathology labels. We also compared Random Forests with **Support Vector Machines (SVM)** for classification and found that Random Forests performed better, especially with larger datasets. Additionally, Random Forests are more efficient and scalable when dealing with a high number of features.

- **Regression Models (for Health Indicators)**: For predicting continuous health indicators, we used **Kernel Ridge Regression** and found that it performed well, particularly when the relationships between the features and the target variables are non-linear.

## Results

The combination of Random Forest for classification and Kernel Ridge Regression for regression yielded strong performance, even with the challenges posed by missing data and sparse measurements.

## Conclusion

This task demonstrated how to handle missing and sparse medical data effectively and develop robust predictive models for both classification and regression tasks. By using summary statistics, trend analysis, and imputation based on age groups, we were able to create meaningful features that improved model performance.
