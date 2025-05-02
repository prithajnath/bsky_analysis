from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import numpy as np

nb_df = pd.read_csv("nb_features.csv")
sl_df = pd.read_csv("df_features.csv")
pn_df = pd.read_csv("prithaj_features.csv")
labels_df = pd.read_csv("kmeans_cluster_labels.csv")

nb_df.drop(columns=['created_at'], inplace=True)
sl_df.drop(columns=['handle', 'bio', 'created_at', 'first_post_created_at', 'first_post_body'], inplace=True)

merged = pn_df.merge(labels_df, left_on='author_did', right_on='author_did')
merged = merged.merge(sl_df, left_on='author_did', right_on='did').drop(columns=['did'])
merged = merged.merge(nb_df, left_on='author_did', right_on='did').drop(columns=['did'])

X = merged.drop(columns=['author_did', 'kmeans_cluster']).replace({False: 0, True: 1}).to_numpy()
y = merged['kmeans_cluster'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

forest = RandomForestClassifier(random_state=42)
balanced_bagger = BalancedBaggingClassifier(forest, sampling_strategy='auto', replacement=False, random_state=42)
balanced_bagger.fit(X_train, y_train)
y_pred = balanced_bagger.predict(X_test)

print("----- Balanced Bagging Classifier -----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))