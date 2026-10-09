
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD TRAINING DATA


df = pd.read_csv("train.csv")

print("Train shape:", df.shape)
print(df.head())
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())


# 2. DATE FEATURES


df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

print("\nDate range:")
print("Start:", df["date"].min())
print("End:", df["date"].max())



# 3. BASIC SALES ANALYSIS


print("\nSales statistics:")
print(df["sales"].describe())

print("\nZero sales:", (df["sales"] == 0).sum())
print("On promotion:", (df["onpromotion"] > 0).sum())

print("\nNumber of product families:", df["family"].nunique())
print("Number of stores:", df["store_nbr"].nunique())



# 4. LOAD STORE INFORMATION


stores = pd.read_csv("stores.csv")

print("\nStores shape:", stores.shape)
print(stores.head())

df = df.merge(
    stores,
    on="store_nbr",
    how="left"
)

print("\nTrain shape after store merge:", df.shape)



# 5. LOAD HOLIDAY DATA


holidays = pd.read_csv("holidays_events.csv")

holidays["date"] = pd.to_datetime(holidays["date"])

holiday_data = holidays[
    holidays["type"] == "Holiday"
].copy()


national_holidays = holiday_data[
    holiday_data["locale"] == "National"
]["date"].drop_duplicates()


df["is_national_holiday"] = (
    df["date"].isin(national_holidays)
).astype(int)

print("\nNational holiday counts:")
print(df["is_national_holiday"].value_counts())



# 6. LOAD OIL DATA


oil = pd.read_csv("oil.csv")

oil["date"] = pd.to_datetime(oil["date"])


oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
oil["dcoilwtico"] = oil["dcoilwtico"].bfill()


df = df.merge(
    oil,
    on="date",
    how="left"
)


df["dcoilwtico"] = df["dcoilwtico"].ffill()
df["dcoilwtico"] = df["dcoilwtico"].bfill()

print("\nMissing oil prices in train:")
print(df["dcoilwtico"].isnull().sum())



# 7. LOAD TEST DATA


test = pd.read_csv("test.csv")

test["date"] = pd.to_datetime(test["date"])

# Date features
test["year"] = test["date"].dt.year
test["month"] = test["date"].dt.month
test["day"] = test["date"].dt.day
test["day_of_week"] = test["date"].dt.dayofweek

print("\nTest shape:", test.shape)
print("Test start:", test["date"].min())
print("Test end:", test["date"].max())


# 8. MERGE STORES INTO TEST


test = test.merge(
    stores,
    on="store_nbr",
    how="left"
)


print("\nTest data after store merge:")
print(test.head())
print("Test shape after store merge:", test.shape)

# 9. MERGE OIL INTO TEST

test = test.merge(
    oil,
    on="date",
    how="left"
)

test["dcoilwtico"] = test["dcoilwtico"].ffill()
test["dcoilwtico"] = test["dcoilwtico"].bfill()

print("\nMissing oil prices in test:")
print(test["dcoilwtico"].isnull().sum())


# 10. NATIONAL HOLIDAY FEATURE FOR TEST

test["is_national_holiday"] = (
    test["date"].isin(national_holidays)
).astype(int)

print("\nNational holiday counts in test:")
print(test["is_national_holiday"].value_counts())

# 11. SORT TRAINING DATA

df = df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

print("\nData after sorting:")
print(df[["store_nbr", "family", "date", "sales"]].head(10))

# 12. CREATE LAG FEATURES

df["sales_lag_1"] = df.groupby(
    ["store_nbr", "family"]
)["sales"].shift(1)

df["sales_lag_7"] = df.groupby(
    ["store_nbr", "family"]
)["sales"].shift(7)



df = df.dropna(
    subset=["sales_lag_1", "sales_lag_7"]
).reset_index(drop=True)

print("\nTrain shape after lag features:", df.shape)


# 13. CREATE FEATURES AND TARGET

X = df.drop(
    columns=["sales", "id", "date"]
)

y = df["sales"]



X = pd.get_dummies(
    X,
    columns=["family", "city", "state", "type"],
    dtype=int
)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# 14. TIME-BASED TRAIN / VALIDATION SPLIT

validation_date = (
    df["date"].max() - pd.Timedelta(days=30)
)

train_mask = df["date"] < validation_date
validation_mask = df["date"] >= validation_date

X_train = X.loc[train_mask]
X_val = X.loc[validation_mask]

y_train = y.loc[train_mask]
y_val = y.loc[validation_mask]

print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_val.shape)

print("Validation starts:", validation_date)


# 15. LINEAR REGRESSION BASELINE

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_val
)

linear_mae = mean_absolute_error(
    y_val,
    linear_pred
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        linear_pred
    )
)

linear_r2 = r2_score(
    y_val,
    linear_pred
)

print("\nLinear Regression Results")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R2:", linear_r2)


# 16. RANDOM FOREST
sample_size = 300000

X_train_rf = X_train.sample(
    n=min(sample_size, len(X_train)),
    random_state=42
)

y_train_rf = y_train.loc[X_train_rf.index]

rf_model = RandomForestRegressor(
    n_estimators=30,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_rf, y_train_rf)

rf_pred = rf_model.predict(X_val)

rf_mae = mean_absolute_error(
    y_val,
    rf_pred
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        rf_pred
    )
)

rf_r2 = r2_score(
    y_val,
    rf_pred
)

print("\nRandom Forest Results")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)


# 17. MODEL COMPARISON

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2
    ]
})

print("\nModel Comparison:")
print(results)


# 18. ACTUAL VS PREDICTED

comparison = pd.DataFrame({
    "Actual Sales": y_val.values[:10],
    "Predicted Sales": rf_pred[:10]
})

print("\nActual vs Predicted:")
print(comparison)


# 19. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(12, 5))

plt.plot(
    y_val.values[:100],
    label="Actual Sales"
)

plt.plot(
    rf_pred[:100],
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Sample")
plt.ylabel("Sales")
plt.legend()

plt.show()


# 20. FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))


# 21. FEATURE IMPORTANCE GRAPH

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.gca().invert_yaxis()

plt.show()


# 22. PREPARE FINAL TRAINING DATA

X_final = X
y_final = y

print("\nFinal training shape:", X_final.shape)


# 23. PREPARE TEST DATA FOR RECURSIVE FORECASTING


historical_sales = df[
    ["date", "store_nbr", "family", "sales"]
].copy()

print("\nHistorical sales:")
print(historical_sales.head())
print("Historical sales shape:", historical_sales.shape)


# 24. TRAIN FINAL MODEL

final_model = rf_model

print("\nFinal model ready!")


# 25. RECURSIVE TEST FORECAST

test_dates = sorted(
    test["date"].unique()
)


sales_history_dict = {
    (row.store_nbr, row.family, row.date): row.sales
    for row in df.itertuples()
}

predictions = []


for current_date in test_dates:

    current_rows = test[
        test["date"] == current_date
    ].copy()

    lag_1_values = []
    lag_7_values = []

    for row in current_rows.itertuples():

        key_lag_1 = (
            row.store_nbr,
            row.family,
            current_date - pd.Timedelta(days=1)
        )

        key_lag_7 = (
            row.store_nbr,
            row.family,
            current_date - pd.Timedelta(days=7)
        )

        lag_1_values.append(
            sales_history_dict.get(
                key_lag_1,
                0
            )
        )

        lag_7_values.append(
            sales_history_dict.get(
                key_lag_7,
                0
            )
        )

    current_rows["sales_lag_1"] = lag_1_values
    current_rows["sales_lag_7"] = lag_7_values

    current_X = current_rows.drop(
        columns=["id", "date"]
    )

    current_X = pd.get_dummies(
        current_X,
        columns=["family", "city", "state", "type"],
        dtype=int
    )


    current_X = current_X.reindex(
        columns=X_final.columns,
        fill_value=0
    )

    
    current_predictions = final_model.predict(
        current_X
    )

    current_predictions = np.maximum(
        current_predictions,
        0
    )

    predictions.extend(
        current_predictions
    )

    for row, prediction in zip(
        current_rows.itertuples(),
        current_predictions
    ):

        key = (
            row.store_nbr,
            row.family,
            current_date
        )

        sales_history_dict[key] = prediction


# 26. CREATE SUBMISSION DATAFRAME

submission = pd.DataFrame({
    "id": test["id"].values,
    "sales": predictions
})


# 27. FINAL CHECKS

print("\nSubmission preview:")
print(submission.head(10))

print("\nSubmission shape:")
print(submission.shape)

print("\nMissing values:")
print(submission.isnull().sum())

print("\nPrediction statistics:")
print(submission["sales"].describe())



# 28. SAVE SUBMISSION

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nsubmission.csv created successfully!")

# FINAL PROJECT RESULTS


accuracy = rf_r2 * 100

print("\n" + "=" * 50)
print("FINAL PROJECT RESULTS")
print("=" * 50)

print("Best Model: Random Forest")
print(f"MAE: {rf_mae:.2f}")
print(f"RMSE: {rf_rmse:.2f}")
print(f"R2 Score: {rf_r2:.4f}")
print(f"Accuracy (R2 Percentage): {accuracy:.2f}%")
print(f"Test Predictions: {len(submission)}")
print("Submission File: submission.csv")