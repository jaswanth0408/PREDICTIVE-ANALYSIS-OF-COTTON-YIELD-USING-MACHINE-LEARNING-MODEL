# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from flaml.ml import sklearn_metric_loss_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  
from flaml import AutoML

# Load Datasets
train = pd.read_csv(r"D:\jaswanth\2.project\DATA SET\cleaned_final_dataset.csv")
test = pd.read_csv(r"D:\jaswanth\2.project\DATA SET\test_dataset.csv")

# Reduce Memory Usage
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage before optimization: {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

# Apply Memory Reduction
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# Encode Categorical Variables
categorical_columns = ["State Name", "Dist Name", "SOIL TYPE PERCENT (Percent)"]
try:
    encoders = joblib.load("encoders.pkl")
    print("Encoders loaded successfully")
except FileNotFoundError:
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])
        encoders[col] = le
    joblib.dump(encoders, "encoders.pkl")
    print("Encoders trained and saved")
for col in categorical_columns:
    test[col] = encoders[col].transform(test[col])

# Replace Infinite and Missing Values
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)
num_cols = train.select_dtypes(include=[np.number]).columns
test[num_cols] = test[num_cols].fillna(test[num_cols].median())
train[num_cols] = train[num_cols].fillna(train[num_cols].median())

# Feature Scaling
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

# Separate Target Variable
y = train.pop('COTTON YIELD (Kg per ha)')
X = train

# Train/Validation Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Train AutoML Model
automl = AutoML()
automl.fit(X_train, y_train, task="regression", metric='rmse', time_budget=3600)

# Display Best Model and Hyperparameters
print('Best ML learner:', automl.best_estimator)
print('Best hyperparameter config:', automl.best_config)
print('Best RMSE on validation data: {0:.4g}'.format(automl.best_loss))
print('Training duration: {0:.4g} s'.format(automl.best_config_train_time))

# Additional Evaluation Metrics
def evaluate_model(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    return mae, rmse, r2, mape

train_pred = automl.predict(X_train)
test_pred = automl.predict(X_test)

train_metrics = evaluate_model(y_train, train_pred)
test_metrics = evaluate_model(y_test, test_pred)

print("\nTrain Metrics:")
print(f"MAE: {train_metrics[0]:.4f}, RMSE: {train_metrics[1]:.4f}, R²: {train_metrics[2]:.4f}, MAPE: {train_metrics[3]:.4f}")

print("\nValidation/Test Metrics:")
print(f"MAE: {test_metrics[0]:.4f}, RMSE: {test_metrics[1]:.4f}, R²: {test_metrics[2]:.4f}, MAPE: {test_metrics[3]:.4f}")

# Predict on Final Test Dataset
y_pred = automl.predict(test)
sol = pd.DataFrame(y_pred, columns=['COTTON YIELD (Kg per ha)'])
print("\nTest Predictions Sample:")
print(sol.head())

# Save Model
joblib.dump(automl.model, 'trainedmodel.pkl')
print("\nModel saved as automl_model.pkl")

# Feature Importance Plot
if hasattr(automl.model, 'feature_importances_'):
    feature_importances = automl.model.feature_importances_
    features = X.columns
    sns.barplot(x=feature_importances, y=features)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=test_pred)
plt.title('Actual vs Predicted Cotton Yield')
plt.xlabel('Actual Cotton Yield')
plt.ylabel('Predicted Cotton Yield')
plt.axline([0, 0], slope=1, color='r', linestyle='--', label='Perfect Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




