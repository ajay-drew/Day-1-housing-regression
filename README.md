# California Housing Prices Regression

An intermediate-level linear regression implementation using scikit-learn to forecast home prices from the California Housing Prices dataset, which is accessible on Kaggle. A pipeline, feature engineering, hyperparameter tuning using GridSearchCV, and simple visualizations are all included in the source.

## Features
- **Pipeline**: Combines preprocessing (StandardScaler) and modeling (Ridge regression).
- **Hyperparameter Tuning**: Uses GridSearchCV to optimize the Ridge `alpha` parameter.
- **Feature Engineering**: Adds derived features like rooms per household and bedrooms per room.
- **Evaluation**: Reports RMSE and R² scores for training and testing sets.
- **Visualizations**: Feature importance, predicted vs actual values, and residual plots.

## Docker Setup

You can run this project in a Docker container for a consistent environment.

### Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your system
- `housing.csv` from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) placed in the `data/` folder

### Build the Docker Image
1. Navigate to the project directory:
   ```bash
   cd california-housing-regression
   ```