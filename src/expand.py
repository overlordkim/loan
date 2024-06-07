import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

# Reading public and internet tables
public_csv_path = '../assets/train_made_work_year.csv'
internet_csv_path = '../assets/train_internet_made_work_year.csv'

public_df = pd.read_csv(public_csv_path)
internet_df = pd.read_csv(internet_csv_path)

# Update column names
internet_df.rename(columns={'is_default': 'isDefault'}, inplace=True)

# Define the target column
target_col = 'isDefault'

# Fill missing columns in the internet dataframe from the public dataframe
missing_columns = set(public_df.columns) - set(internet_df.columns)
for col in missing_columns:
    internet_df[col] = public_df[col].mean()

# Remove extra columns from the internet dataframe
extra_columns = set(internet_df.columns) - set(public_df.columns)
internet_df.drop(columns=extra_columns, inplace=True)

# Fill other missing values in the internet dataframe with the mean of the public dataframe
for col in internet_df.columns:
    if internet_df[col].isnull().any():
        internet_df[col].fillna(public_df[col].mean(), inplace=True)

# Splitting features and the target variable
X_public = public_df.drop(columns=[target_col])
y_public = public_df[target_col]
X_internet = internet_df.drop(columns=[target_col])
y_internet = internet_df[target_col]



# Ensure prediction set and training set have the same feature columns
columns = [col for col in public_df.columns if col != target_col]
X_internet = X_internet[columns]

# Training the CatBoost model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.08,
    depth=5,
    eval_metric='AUC',
    verbose=200,
    random_seed=42
)

# Fit model
model.fit(X_public, y_public, cat_features = [col for col in X_public.columns if 'class_' in col or
                'employer_type_' in col or 'industry_' in col or 'work_year_' in col]
)

# Predicting the default probability
predictions_proba = model.predict_proba(X_internet)[:, 1]
print(predictions_proba)

# Print the number of samples meeting the condition under different thresholds
thresholds = np.arange(0.01, 0.04, 0.005)
print(internet_df[target_col].head(10))
for threshold in thresholds:
    selected_indices = internet_df[((predictions_proba < threshold) & (internet_df[target_col] == 0)) |
                                   ((predictions_proba >= 1 - threshold) & (internet_df[target_col] == 1))].index
    count = len(selected_indices)
    print(f"Threshold: {threshold:.3f}, Number of non-default samples: {count}")

    # Save selected rows and combine with the original public table
    selected_df = internet_df.loc[selected_indices]
    combined_df = pd.concat([public_df, selected_df])
    output_csv_path = f'../assets/combined_{threshold:.3f}.csv'
    combined_df.to_csv(output_csv_path, index=False)
    print(f'Saved to {output_csv_path}')
