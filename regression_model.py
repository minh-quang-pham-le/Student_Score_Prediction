# Step 1: Data Collection
import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor
from ydata_profiling import ProfileReport
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Step 2: Statistics
data = pd.read_csv("StudentScore.xls")
# profile = ProfileReport(data, title = "Student Score", explorative = True)
# profile.to_file("student_score_report.html")

target = "writing score"
x = data.drop(columns=target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
# clf = LazyRegressor(verbose = 0, ignore_warnings=True, custom_metric=None, random_state=42)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# print(models)


# Step 3: Data preprocessing (Pretending there are some missing values)
bool_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", OrdinalEncoder())
])
x_train[["gender", "lunch", "test preparation course"]] = bool_transformer.fit_transform(x_train[["gender", "lunch", "test preparation course"]])
x_test[["gender", "lunch", "test preparation course"]] = bool_transformer.transform(x_test[["gender", "lunch", "test preparation course"]])

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
x_train[["math score", "reading score"]] = num_transformer.fit_transform(x_train[["math score", "reading score"]])
x_test[["math score", "reading score"]] = num_transformer.transform(x_test[["math score", "reading score"]])

educational_levels = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
ordinal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", OrdinalEncoder(categories=[educational_levels]))
])

x_train[["parental level of education"]]= ordinal_transformer.fit_transform(x_train[["parental level of education"]])
x_test[["parental level of education"]]= ordinal_transformer.transform(x_test[["parental level of education"]])

ohe = OneHotEncoder(drop="first", sparse_output=False)
# Concatenate training set
race_train = ohe.fit_transform(x_train[["race/ethnicity"]])
race_train_df = pd.DataFrame(race_train, index=x_train.index, columns=ohe.get_feature_names_out(["race/ethnicity"]))
x_train = x_train.drop(columns=["race/ethnicity"])
x_train = pd.concat([x_train, race_train_df], axis=1)
# Concatenate test set
race_test = ohe.transform(x_test[["race/ethnicity"]])
race_test_df = pd.DataFrame(race_test, index=x_test.index, columns=ohe.get_feature_names_out(["race/ethnicity"]))
x_test = x_test.drop(columns=["race/ethnicity"])
x_test = pd.concat([x_test, race_test_df], axis=1)

# Step 4: Model building
params = {
    "penalty": ['l1', 'l2', 'elasticnet', 'none'],
    'alpha': [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 5, 10]
}

print("SGDRegressor GridSearchCV...")
grid = GridSearchCV(SGDRegressor(random_state=42), params, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=4)
grid.fit(x_train, y_train)

# Step 5: Model evaluation
print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {-grid.best_score_}")
print(f"Done...")

print("\nScore on test set:")

best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"MSE : {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAE : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R²  : {r2_score(y_true, y_pred):.4f}\n")

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("SGDRegressor", y_test, y_pred)

# Step 6: Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(lr, file)