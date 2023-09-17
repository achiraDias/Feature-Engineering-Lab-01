# %% [markdown]
# Import libraries and modules

# %%
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# Import training, validation and testing datasets

# %%
train_data = pd.read_csv('train.csv')
valid_data = pd.read_csv('valid.csv')
test_data = pd.read_csv('test.csv')

# %%
train_data.head()

# %% [markdown]
# Process the data

# %% [markdown]
# Drop the columns where there are null values for the lables in the training dataset

# %%
# Check for null values in train dataset
train_null_counts = train_data.isnull().sum()
print("train null counts : \n {}".format(train_null_counts))

# Drop rows with null values in the final four columns for train dataset
train_data = train_data.dropna(subset=train_data.columns[-4:], how='any')

# %% [markdown]
# Replace the null values in the features with their means in the train, valid and test datasets.

# %%
train_data = train_data.fillna(train_data.mean())
valid_data = valid_data.fillna(valid_data.mean())
test_data = test_data.fillna(test_data.mean())

# %% [markdown]
# Processed training data

# %%
train_data.head()

# %% [markdown]
# Separate features and labels in the train, valid and test datasets

# %%
# Train dataset
train_features = train_data.iloc[:, :-4]
train_labels = train_data.iloc[:, -4:]

# Valid dataset
valid_features = valid_data.iloc[:, :-4]
valid_labels = valid_data.iloc[:, -4:]

# Test dataset
test_features = test_data.iloc[:, :-4]
test_labels = test_data.iloc[:, -4:]

# %% [markdown]
# Extract the first label in the train, valid and test datasets

# %%
train_label2 = train_labels.iloc[:,1]
valid_label2 = valid_labels.iloc[:,1]
test_label2 = test_labels.iloc[:,1]

# %% [markdown]
# # Predicting Label 2 without Feature Engineering

# %% [markdown]
# Make copies of the features and labels of the datasets to be used in the models without feature engineering

# %%
# Train dataset
train_features_copy = train_features.copy()
train_labels_copy = train_labels.copy()

# Valid dataset
valid_features_copy = valid_features.copy()
valid_labels_copy = valid_labels.copy()

# Test dataset
test_features_copy = test_features.copy()
test_labels_copy = test_labels.copy()

# %% [markdown]
# Make copies of the label 2 of the datasets to be used in the models without feature engineering

# %%
train_label2_copy = train_label2.copy()
valid_label2_copy = valid_label2.copy()
test_label2_copy = test_label2.copy()

# %%
scaler = StandardScaler()
train_features_copy = scaler.fit_transform(train_features_copy)
valid_features_copy = scaler.transform(valid_features_copy)
test_features_copy = scaler.transform(test_features_copy)

# %% [markdown]
# Use the raw scaled features to train the best model which is KNN Regressor

# %%
from sklearn.neighbors import KNeighborsRegressor
kNeighborsRegressor = KNeighborsRegressor()

kNeighborsRegressor.fit(train_features_copy, train_label2_copy)

# %% [markdown]
# Used the trained model on all features to predict the valid and get metrics

# %%
# Predict on the train data
y_pred_base_train = kNeighborsRegressor.predict(train_features_copy)

# Metrics for classification evaluation
meanSquareError = mean_squared_error(train_label2_copy, y_pred_base_train)
r2Score = r2_score(train_label2_copy, y_pred_base_train)

print("KNRegressor on train data:")
print('Mean Square Error: ', meanSquareError)
print('R2 Score: ',r2Score)
print('\n')

# Predict on the validation data
y_pred_base_valid = kNeighborsRegressor.predict(valid_features_copy)

# Metrics for classification evaluation on validation data
meanSquareError = mean_squared_error(valid_label2_copy, y_pred_base_valid)
r2Score = r2_score(valid_label2_copy, y_pred_base_valid)

print("KNRegressor on valid data:")
print('Mean Square Error: ', meanSquareError)
print('R2 Score: ',r2Score)

# %% [markdown]
# Predict the label 2 on test data

# %%
# Predict on the test data
y_pred_base_test = kNeighborsRegressor.predict(test_features_copy)

# %% [markdown]
# # Predicting Label 2 with Feature Engineering

# %% [markdown]
# Use feature selection based on correlation matrix and feature extraction based on PCA

# %%
# Plot distribution of train_label2
labels, counts = np.unique(train_label2, return_counts=True)

plt.figure(figsize=(10, 4))
plt.xticks(labels)
plt.bar(labels, counts)
plt.xlabel('Target Label 2')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 2')
plt.show()

# %%
#Calculate the correlation matrix
correlation_matrix = train_features.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()

# %% [markdown]
# Identifying the features that are highly correlated with each other in the traning dataset

# %%
correlation_threshold = 0.9

highly_correlated = set()

# Find highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)

# %%
# Remove highly correlated features
train_features = train_features.drop(columns=highly_correlated)
valid_features = valid_features.drop(columns=highly_correlated)
test_features = test_features.drop(columns=highly_correlated)

# %%
# Display the filtered train feature count
print("Filtered train features: {}".format(train_features.shape))

# Display the filtered valid feature count
print("Filtered valid features: {}".format(valid_features.shape))

# Display the filtered test feature count
print("Filtered test features: {}".format(test_features.shape))

# %% [markdown]
# Identify the features that are highly correlated with the label using the traning dataset

# %%
# Calculate the correlation matrix between features and train_label2
correlation_with_target = train_features.corrwith(train_label2)

# Correlation threshold
correlation_threshold = 0.05

# Select features that meet the correlation threshold
highly_correlated_features = correlation_with_target[correlation_with_target.abs() > correlation_threshold]

print(highly_correlated_features)

# %% [markdown]
# Extract the features that are only highly correlated with the label2

# %%
train_features = train_features[highly_correlated_features.index]
valid_features = valid_features[highly_correlated_features.index]
test_features = test_features[highly_correlated_features.index]

# %% [markdown]
# Display the resulting feature shapes of the datasets

# %%
# Display the filtered train feature count
print("Filtered train features: {}".format(train_features.shape))

# Display the filtered valid feature count
print("Filtered valid features: {}".format(valid_features.shape))

# Display the filtered test feature count
print("Filtered test features: {}".format(test_features.shape))

# %% [markdown]
# Standardize the features of all datasets

# %%
scaler = StandardScaler()
standardized_train_features = scaler.fit_transform(train_features)
standardized_valid_features = scaler.transform(valid_features)
standardized_test_features = scaler.transform(test_features)

# %% [markdown]
# ### Feature Extraction

# %%
variance_threshold = 0.999

# Apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_result = pca.fit_transform(standardized_train_features)
pca_valid_result = pca.transform(standardized_valid_features)
pca_test_result = pca.transform(standardized_test_features)

# Explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_
print("Explained Variance Ratio after Dimensionality Reduction:", explained_variance_ratio_reduced)

# Plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_result.shape[1] + 1), explained_variance_ratio_reduced)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# Display the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_result.shape))
# Display the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_result.shape))
# Display the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_result.shape))

# %% [markdown]
# K Neighbors Regressor is selected based on Mean Square Error and R2 score

# %%
# Number of features used in PCA
num_features = pca_train_result.shape[1]
print(f"Number of features: {num_features}\n")

model = KNeighborsRegressor()

# Train the model on the training data
model.fit(pca_train_result, train_label2)

# Predict on the train data
y_pred_train = model.predict(pca_train_result)

# Calculate metrics for classification evaluation
meanSquareError = mean_squared_error(train_label2, y_pred_train)
r2Score = r2_score(train_label2, y_pred_train)

print("KNRegressor on train data:")
print('Mean Square Error: ', meanSquareError)
print('R2 Score: ',r2Score)
print("\n")

# Predict on the validation data
y_pred_valid = model.predict(pca_valid_result)

# Calculate metrics for classification evaluation on validation data
meanSquareError = mean_squared_error(valid_label2, y_pred_valid)
r2Score = r2_score(valid_label2, y_pred_valid)

print("KNRegressor on valid data:")
print('Mean Square Error: ', meanSquareError)
print('R2 Score: ',r2Score)
print("\n")

# Predict on the test data
y_pred_test = model.predict(pca_test_result)


# %% [markdown]
# # Generate Output CSV

# %%
feature_count = pca_test_result.shape[1]

header_row = [f"new_feature_{i}" for i in range(1,feature_count+1)]

df = pd.DataFrame(pca_test_result, columns  = header_row)

df.insert(loc=0, column='Predicted labels before feature engineering', value=y_pred_base_test)
df.insert(loc=1, column='Predicted labels after feature engineering', value=y_pred_test)
df.insert(loc=2, column='No of new features', value=np.repeat(feature_count, pca_test_result.shape[0]))

df.to_csv('190137J_label_2.csv', index=False)


