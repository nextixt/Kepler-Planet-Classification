import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import shap

df = pd.read_csv('cumulative.csv')

# defining num columns and categorial columns
num_cols = ['koi_score', 'koi_period', 'koi_depth', 'koi_teq', 'koi_prad', 'koi_steff', 'koi_srad', 'koi_impact']
cat_cols  = ['koi_pdisposition']


# defining and splitting X and y
X = df[num_cols + cat_cols]
y = df['koi_disposition']
le = LabelEncoder()
y_encoded = le.fit_transform(y) 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.3)

# defining model
model = XGBClassifier(n_jobs = -1, n_estimators = 100)

# defining Column transformer for categorial column and num columns and then defining pipeline
num_transformer = SimpleImputer()

cat_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

column_transformer = ColumnTransformer(transformers = [
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
]
)

kepler_pipeline = Pipeline(steps = [
    ('transformer', column_transformer),
    ('model', model)
])

# fitting pipeline and making predictions 
kepler_pipeline.fit(X_train, y_train)
prediction = kepler_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
report = classification_report(y_test, prediction)

# Class matching for y
print("Class matching for y:")
for i, class_name in enumerate(le.classes_):
    print(f"{i} -> {class_name}")

# match for koi_pdisposition in X
cat_transformer.fit(X_train[cat_cols]) 
encoder = cat_transformer.named_steps['encoder']
print("Match for koi_pdisposition in X:")
for i, category in enumerate(encoder.categories_[0]):
    print(f"{i} -> {category}")

# printing report
print(f'Accuracy after learning: {accuracy}\n Classification report: {report}')

# shap visualization
X_train_processed = kepler_pipeline.named_steps['transformer'].transform(X_train)
model_for_shap = kepler_pipeline.named_steps['model']
explainer = shap.TreeExplainer(model_for_shap)
shap_values = explainer.shap_values(X_train_processed)
shap.summary_plot(shap_values, X_train_processed, feature_names=num_cols + cat_cols, plot_type="bar")
plt.show()
