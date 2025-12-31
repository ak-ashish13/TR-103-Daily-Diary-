import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

class CreditCardFraudDetector:
    def __init__(self):
        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self, train_path, test_path):
        """Load separate train and test data"""
        st.write("### Loading Data...")
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)

        # Handle missing values
        if self.df_train.isnull().sum().any():
            self.df_train.fillna(method='ffill', inplace=True)
        if self.df_test.isnull().sum().any():
            self.df_test.fillna(method='ffill', inplace=True)

        # Separate features & target
        self.X_train = self.df_train.drop(['Class', 'Time'], axis=1, errors='ignore')
        self.y_train = self.df_train['Class']
        self.X_test = self.df_test.drop(['Class', 'Time'], axis=1, errors='ignore')
        self.y_test = self.df_test['Class']

        # Scale amount column
        if 'Amount' in self.X_train.columns:
            self.X_train['Amount'] = self.scaler.fit_transform(self.X_train[['Amount']])
            self.X_test['Amount'] = self.scaler.transform(self.X_test[['Amount']])

        st.success("Data loaded successfully!")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def explore_data(self):
        """Show class distribution and transaction stats"""
        st.subheader("Class Distribution")
        class_dist = self.y_train.value_counts()
        class_dist_pct = self.y_train.value_counts(normalize=True).mul(100).round(4)
        st.write(f"**Normal Transactions:** {class_dist[0]:,} ({class_dist_pct[0]:.2f}%)")
        st.write(f"**Fraudulent Transactions:** {class_dist[1]:,} ({class_dist_pct[1]:.2f}%)")

        st.subheader("Transaction Amount Statistics")
        fraud = self.df_train[self.df_train['Class'] == 1]['Amount']
        normal = self.df_train[self.df_train['Class'] == 0]['Amount']

        stats = pd.DataFrame({
            "Normal": normal.describe(),
            "Fraud": fraud.describe()
        })
        st.dataframe(stats)

    def train_model(self, use_smote=True):
        """Train logistic regression model with optional SMOTE"""
        st.write("### Training Model...")
        if use_smote:
            st.info("Using SMOTE to handle class imbalance")
            model = Pipeline([
                ('sampling', SMOTE(random_state=42)),
                ('classification', LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                ))
            ])
        else:
            model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )

        model.fit(self.X_train, self.y_train)
        self.model = model

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1')
        st.write(f"**Cross-Validation F1 Scores:** {cv_scores}")
        st.write(f"**Mean CV F1 Score:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return model

    def evaluate_model(self, threshold=0.5):
        """Evaluate model on test data"""
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)

        st.subheader("Confusion Matrix")
        st.table(pd.DataFrame(cm,
            index=["Actual Normal", "Actual Fraud"],
            columns=["Predicted Normal", "Predicted Fraud"]
        ))

        st.subheader("Performance Metrics")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1-Score:** {f1:.4f}")
        st.write(f"**ROC-AUC:** {roc_auc:.4f}")
        st.write(f"**Average Precision:** {avg_precision:.4f}")

        st.subheader("Classification Report")
        st.text(classification_report(self.y_test, y_pred, target_names=["Normal", "Fraud"], digits=4))

def main():
    st.title("💳 Credit Card Fraud Detection App")
    st.write("Upload **training** and **testing** datasets to train and evaluate the model.")

    train_file = st.file_uploader("Upload Training CSV", type=["csv"])
    test_file = st.file_uploader("Upload Testing CSV", type=["csv"])

    if train_file and test_file:
        detector = CreditCardFraudDetector()
        X_train, X_test, y_train, y_test = detector.load_data(train_file, test_file)
        detector.explore_data()

        if st.button("Train Model"):
            model = detector.train_model(use_smote=True)
            detector.evaluate_model()

if __name__ == "__main__":
    main()