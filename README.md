# Data-orch-Project
Classification of Heart Arrhythmia using vector database and similarity search.

import pandas as pd
# Load the dataset from the specified path
mitbih_train = pd.read_csv('/content/mitbih_train.csv', header=None)
# Split data into features (x_data) and labels (y_data)
x_data = mitbih_train.iloc[:, :-1] # All columns except the last one
y_data = mitbih_train.iloc[:, -1] # The last column is the label
# Get the unique classes
classes = y_data.unique()
# Loop through each class and count the samples
for i in classes:
class_count = (y_data == i).sum() # Count samples for each class
print(f"Class {int(i)} has {class_count} samples.")
import pandas as pd
import matplotlib.pyplot as plt
# Assuming equilibre is already defined
# Example: equilibre = y_data.value_counts()
# Sample data for equilibre (replace this with your actual equilibre)
equilibre = pd.Series([72471, 6431, 5788, 2223, 641], index=[0, 4, 2, 3, 1])
# Mapping labels based on the provided counts
labels = ['n', 'q', 'v', 's', 'f'] # These are your class labels in order
# Plotting the donut chart
plt.figure(figsize=(20, 11)) # Square figure for a better donut appearance
my_circle = plt.Circle((0, 0), 0.7, color='white') # Create the donut effect
plt.pie(equilibre, labels=labels, colors=['red', 'green', 'blue', 'skyblue', 'orange'],
autopct='%1.1f%%', startangle=90) # Start angle for better orientation
p = plt.gcf() # Get the current figure
p.gca().add_artist(my_circle) # Add the circle in the center
plt.title('Distribution of Classes in the Dataset') # Optional title
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
57
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
file_path = '/content/mitbih_train.csv'
data = pd.read_csv(file_path)
# Clean up column names
cleaned_columns = [str(col).replace('e+00.', '').replace('1.000000000000000000', '1') for col in
data.columns]
data.columns = cleaned_columns
# Step 1: Check for missing values
missing_values = data.isnull().sum().sum()
print(f'Missing values: {missing_values}')
# Step 2: Normalize the data
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
# Step 3: Visualize the correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = normalized_data.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()
# Step 4: Visualize
if missing_values > 0:
normalized_data.fillna(normalized_data.mean(), inplace=True)
normalized_data.to_csv('normalized_ptbdb_abnormal.csv', index=False)
plt.figure(figsize=(12, 6))
for i in range(5):
plt.plot(normalized_data.iloc[:, i], label=f'Feature {i+1}')
plt.title('Sample ECG Signals (First 5 Features)')
plt.xlabel('Time')
plt.ylabel('Normalized Value')
plt.legend()
plt.show()
import wfdb
58
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# Load MIT-BIH Arrhythmia dataset
record = wfdb.rdrecord('/content/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database1.0.0/100')
annotation = wfdb.rdann('/content/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database1.0.0/100', 'atr')
# Extract signal and annotations
signal = record.p_signal
annotations = annotation.symbol
# Convert to pandas DataFrame
data = pd.DataFrame(signal, columns=['Channel 1', 'Channel 2'])
# Add annotations and ensure shapes are consistent
data['Annotations'] = np.nan
# Iterate only over the valid index range to avoid mismatches
for i, sample in enumerate(annotation.sample):
if sample < len(data): # Check if the sample index is within the data bounds
data.loc[sample, 'Annotations'] = annotations[i]
# Filter the data based on valid classes *before* extracting X and y
from collections import Counter
class_counts = Counter(data['Annotations'].dropna()) # Count only non-NaN values
valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
data = data[data['Annotations'].isin(valid_classes)]
# Features and labels
X = data[['Channel 1', 'Channel 2']].values
y = data['Annotations'].values
# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, 
stratify=y)
59
# Train the SVM model
svm_classifier = SVC(kernel='rbf', C=1, gamma='auto') # Using RBF kernel
svm_classifier.fit(X_train, y_train)
# Make predictions
y_pred = svm_classifier.predict(X_test)
# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
# Optional: Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
