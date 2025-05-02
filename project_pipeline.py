import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import os
os.environ['LDFLAGS'] = '-L/opt/homebrew/opt/libomp/lib'
os.environ['CPPFLAGS'] = '-I/opt/homebrew/opt/libomp/include'

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

merged_df = pd.read_csv("merged_df.csv")

# Fill NA for has_mentions_in_bio and has_url_in_bio
merged_df['has_mentions_in_bio'] = merged_df['has_mentions_in_bio'].fillna(0)
merged_df['has_url_in_bio'] = merged_df['has_url_in_bio'].fillna(0)

# Drop rows with any missing values
merged_df_clean = merged_df.dropna()
merged_df_clean

# select features and target
X = merged_df_clean.drop(columns=['kmeans_cluster', 'did', 'handle', 'first_post_body', 'bio', 'created_at', 'first_post_created_at'])
y = merged_df_clean['kmeans_cluster']


# train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# class imbalance ratios
scale = y_train.value_counts()[0] / y_train.value_counts()[1]
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# defined baseline models
# These models treat all classes equally, regardless of how imbalanced the dataset is

# more balanced models based on class weight
# These versions automatically adjusts the importance of each class based on their frequency in the training data.

# logistic regression
log_reg = LogisticRegression(
    max_iter=1000
)

log_reg_bal = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

log_reg_lasso = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    penalty='l1', # L1
    solver='liblinear')

# svc
svc = SVC(
    probability=True
)

svc_bal = SVC(
    probability=True,
    class_weight='balanced'
)


svc_lasso = LinearSVC(
    penalty='l1', # lasso
    dual=False,
    class_weight='balanced',
    max_iter=10000)

# random forest
rf = RandomForestClassifier()
rf_bal = RandomForestClassifier(
    class_weight='balanced'
)

# xgboost
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='aucpr'
)
xgb_bal = XGBClassifier(
    use_label_encoder=False,
    eval_metric='aucpr',
    scale_pos_weight=scale)
xgb_lasso = XGBClassifier(
    use_label_encoder=False,
    eval_metric='aucpr',
    scale_pos_weight=scale,
    reg_alpha=1.0  # Lasso
)

# iterative xgboost
xgb_iter = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale,
    n_estimators=500
)
xgb_iter.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=True
)

# Histogram Based Gradient Boosting
hgb = HistGradientBoostingClassifier()
hgb.fit(X_train_scaled, y_train)

hgb_bal = HistGradientBoostingClassifier()
hgb_bal.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# Combine into ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', log_reg),
        ('lr_bal', log_reg_bal),
        ('lr_lasso', log_reg_lasso),
        ('svc', svc),
        ('svc_bal', svc_bal),
        ('rf', rf),
        ('rf_bal', rf_bal),
        ('xgb', xgb),
        ('xgb_bal', xgb_bal),
        ('xgb_lasso', xgb_lasso),
        ('xgb_iter', xgb_iter),
        ('hgb', hgb),
        ('hgb_bal', hgb_bal),
    ],
    voting='soft' # soft uses predict_proba - hard uses voting
)

# Ensemble with Histogram-based gradient boosting and Logistic Balanced
hgb_and_logistic_bal = VotingClassifier(
    estimators=[
        ('lr_bal', log_reg_bal),
        ('hgb_bal', hgb_bal),
    ],
    voting='soft' # soft uses predict_proba - hard uses voting
)

# define classifiers
models = {
    "Logistic Regression": log_reg,
    "Logistic Regression (Balanced)": log_reg_bal,
    "Logistic Regression (Lasso)": log_reg_lasso,
    "SVC": svc,
    "SVC (Balanced)": svc_bal,
    "SVC (Lasso)": svc_lasso,
    "Random Forest": rf,
    "Random Forest (Balanced)": rf_bal,
    "XGBoost": xgb,
    "XGBoost (Balanced)": xgb_bal,
    "XGBoost (Lasso)": xgb_lasso,
    "XGBoost with Iterations": xgb_iter,
    "HGB Gradient Boosting Classifier": hgb,
    "HGB Classifier (Balanced)": hgb_bal,
    "Ensemble (Voting Classifier)": ensemble,
    "HGB and Logistic(Bal) Ensemble": hgb_and_logistic_bal
}
print(y.value_counts(normalize=True))


# Collect metrics
metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}


# train, predict, evaluate
for name, model in models.items():
    print(f"\n🔍 {name}")

    # train
    model.fit(X_train_scaled, y_train)

    # predict
    y_pred = model.predict(X_val_scaled)

    # evaluate
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Precision:", precision_score(y_val, y_pred, average='binary', zero_division=0))
    print("Recall:", recall_score(y_val, y_pred, average='binary', zero_division=0))
    print("F1 Score:", f1_score(y_val, y_pred, average='binary', zero_division=0))

    # full classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


    metrics['Model'].append(name)
    metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
    metrics['Precision'].append(precision_score(y_val, y_pred, average='binary', zero_division=0))
    metrics['Recall'].append(recall_score(y_val, y_pred, average='binary', zero_division=0))
    metrics['F1 Score'].append(f1_score(y_val, y_pred, average='binary', zero_division=0))

# convert to dataframe
metrics_df = pd.DataFrame(metrics)

# plot
ax = metrics_df.set_index('Model').plot(kind='bar', figsize=(12, 6))
plt.title("Classifier Comparison on Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=90)
plt.grid(axis='y')
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=1)
plt.tight_layout()
plt.show()