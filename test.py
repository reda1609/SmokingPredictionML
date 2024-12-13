import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load the new dataset
data = pd.read_csv('our_data.csv')

# Display basic information about the dataset
print("Dataset Info:")
data.info()

print("\nSummary Statistics:")
print(data.describe())

# Univariate Analysis
plt.figure(figsize=(12, 6))
sns.countplot(x='smoking', data=data)
plt.title('Smoking Status Distribution')
plt.xlabel('Smoking (1 = Smoker, 0 = Non-Smoker)')
plt.ylabel('Count')
plt.show()

# Histograms for continuous features
continuous_features = ['fasting blood sugar', 'Cholesterol', 'relaxation', 'serum creatinine', 'dental caries', 'height(cm)', 'waist(cm)', 'AST', 'age', 'Gtp']
for feature in continuous_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Checking missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
data[continuous_features] = imputer.fit_transform(data[continuous_features])

# Bivariate Analysis
# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

# Smoking status vs Continuous Features
for feature in continuous_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='smoking', y=feature, data=data)
    plt.title(f'{feature} by Smoking Status')
    plt.xlabel('Smoking (1 = Smoker, 0 = Non-Smoker)')
    plt.ylabel(feature)
    plt.show()

# Multivariate Analysis
# Standardize the data for PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[continuous_features])

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Add PCA results to the dataset
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]

# Scatter plot of the PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='smoking', data=data, palette='coolwarm')
plt.title('PCA Result - Smoking Status')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Smoking')
plt.show()

# Feature Engineering and Preprocessing
# Normalize continuous features using Min-Max Scaling
scaler_minmax = MinMaxScaler()
data[continuous_features] = scaler_minmax.fit_transform(data[continuous_features])

print("\nNormalized Features:")
print(data[continuous_features].head())

# Ensemble Methods
# Prepare data for modeling
X = data[continuous_features]
y = data['smoking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bagging
bagging_model = BaggingClassifier(n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_pred = bagging_model.predict(X_test)
print("\nBagging Classifier:")
print("Accuracy:", accuracy_score(y_test, bagging_pred))
print(classification_report(y_test, bagging_pred))

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rfc_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
gs_rf = GridSearchCV(rf_model, rfc_params, cv=5, scoring='accuracy')
gs_rf.fit(X_train, y_train)
print("\nBest Parameters for Random Forest:", gs_rf.best_params_)
rf_best = gs_rf.best_estimator_
rf_pred = rf_best.predict(X_test)
print("\nRandom Forest Classifier (Tuned):")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Boosting
boosting_model = AdaBoostClassifier(random_state=42)
abc_params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1]}
gs_abc = GridSearchCV(boosting_model, abc_params, cv=5, scoring='accuracy')
gs_abc.fit(X_train, y_train)
print("\nBest Parameters for AdaBoost:", gs_abc.best_params_)
abc_best = gs_abc.best_estimator_
boosting_pred = abc_best.predict(X_test)
print("\nAdaBoost Classifier (Tuned):")
print("Accuracy:", accuracy_score(y_test, boosting_pred))
print(classification_report(y_test, boosting_pred))
