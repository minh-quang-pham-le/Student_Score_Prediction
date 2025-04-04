# Step 1: Data Collection
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from ydata_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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


# Step 3: Data preprocessing
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
# Huấn luyện và biến đổi tập huấn luyện
race_train = ohe.fit_transform(x_train[["race/ethnicity"]])
race_train_df = pd.DataFrame(race_train, index=x_train.index, columns=ohe.get_feature_names_out(["race/ethnicity"]))
x_train = x_train.drop(columns=["race/ethnicity"])
x_train = pd.concat([x_train, race_train_df], axis=1)
# Biến đổi tập kiểm tra
race_test = ohe.transform(x_test[["race/ethnicity"]])
race_test_df = pd.DataFrame(race_test, index=x_test.index, columns=ohe.get_feature_names_out(["race/ethnicity"]))
x_test = x_test.drop(columns=["race/ethnicity"])
x_test = pd.concat([x_test, race_test_df], axis=1)

# Step 4: Model building
params = {
    "penalty": ['l1', 'l2', None],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 3, 5, 10]
}