# Step 1: Data Collection
import pandas as pd
from lazypredict.Supervised import numeric_transformer
from ydata_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
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

# Step 3: Data preprocessing
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

educational_levels = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
ordinal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", OrdinalEncoder(categories=[educational_levels]))
])

output = ordinal_transformer.fit_transform(x_train[["parental level of education"]])