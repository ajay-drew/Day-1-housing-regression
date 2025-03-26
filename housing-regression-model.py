import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/housing.csv")

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X['total_bedrooms'].fillna(X['total_bedrooms'].median(), inplace=True)
X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)

X['rooms_per_household'] = X['total_rooms'] / X['households']
X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

param_grid = {
    'ridge__alpha': [0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best CV R² Score:", grid_search.best_score_)

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training R²: {train_r2:.3f}")
print(f"Testing R²: {test_r2:.3f}")

feature_names = X.columns
coefficients = best_model.named_steps['ridge'].coef_
feat_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(coefficients)
}).sort_values('Importance', ascending=False)

sns.set_style("whitegrid")
sns.set_palette("husl")

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance, palette="viridis")
plt.title('Feature Importance', fontsize=14, weight='bold', pad=10)
plt.xlabel('Absolute Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, cmap='Spectral', c=y_test)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Prediction')
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Predicted vs Actual Values', fontsize=14, weight='bold', pad=10)
plt.legend()
plt.tight_layout()
plt.show()