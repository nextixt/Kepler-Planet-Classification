import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt # for future

df = pd.read_csv('cumulative.csv')

features = ['koi_pdisposition', 'koi_score', 'koi_period', 'koi_depth', 'koi_teq', 'koi_prad', 'koi_steff', 'koi_srad'] # for future, not using now



second_feat = ['koi_period', 'koi_depth', 'koi_disposition'] # now using this features

# defining label encoder and imputer
le = LabelEncoder()
imputer = SimpleImputer()

# transforming and splitting X and y
X = pd.DataFrame(imputer.fit_transform(df[['koi_period', 'koi_depth']]))
y = pd.DataFrame(le.fit_transform(df['koi_disposition']))
X.columns = df[['koi_period', 'koi_depth']].columns
y.columns = df[['koi_disposition']].columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# defing model, fitting and making predictions
model = XGBClassifier(n_jobs = -1, n_estimators = 100)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
report = classification_report(y_test, prediction)
print(f'Accuracy after learning: {accuracy}\n Classification report: {report}')
