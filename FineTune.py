import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load Data
df = pd.read_csv('Dataset.csv', dtype={'Patient_ID': 'int32', 'SepsisLabel': 'int8'})
df.columns = df.columns.str.strip()

# 2. Drop unnecessary columns
cols_to_drop = ['Unit1', 'Unit2', 'HospAdmTime', 'EtCO2', 'Bilirubin_direct', 'TroponinI', 'Fibrinogen']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Fill missing values
df = df.fillna(0)

# 3. Scaling
scaler = StandardScaler()
feature_cols = [c for c in df.columns if c not in ['SepsisLabel', 'Patient_ID', 'Hour']]
df[feature_cols] = scaler.fit_transform(df[feature_cols].astype('float32'))

# 4. Sequence creation
def create_fast_sequences(df, seq_length=6):
    data_array = df[feature_cols].values
    labels_array = df['SepsisLabel'].values
    ids_array = df['Patient_ID'].values

    X, y = [], []

    for i in range(seq_length, len(df)):
        if ids_array[i] == ids_array[i - seq_length]:
            X.append(data_array[i - seq_length:i])
            y.append(labels_array[i])

    return np.array(X, dtype='float32'), np.array(y, dtype='int8')

X, y = create_fast_sequences(df, seq_length=6)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# 6. Class Weights (BOOST recall)
class_weights = {0: 1, 1: 6}
print("Using Class Weights:", class_weights)

# 7. Model
model = Sequential([
    Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])),

    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32),
    Dropout(0.3),

    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision')
    ]
)

# 8. Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 9. Training
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=1024,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# 10. Predictions
y_probs = model.predict(X_test, batch_size=2048).flatten()

# =========================
# 🔥 FINAL THRESHOLD TUNING
# =========================
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Prioritize recall more than precision
score = recall * 0.8 + precision * 0.2
best_index = np.argmax(score)

best_threshold = thresholds[best_index]

# Slightly lower threshold to boost recall
sbest_threshold = max(best_threshold - 0.05, 0.2)


print("\nFinal Threshold:", best_threshold)
print("Precision:", precision[best_index])
print("Recall:", recall[best_index])

# Apply threshold
y_preds = (y_probs > best_threshold).astype(int)

# 11. Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_preds))

# 12. ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# 13. Confusion Matrix
cm = confusion_matrix(y_test, y_preds)

# =========================
# 📊 Visualization
# =========================
plt.figure(figsize=(20, 10))

# Accuracy
plt.subplot(2, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

# ROC Curve
plt.subplot(2, 3, 2)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend()
plt.title('ROC Curve')

# Confusion Matrix
plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')

# Prediction Distribution
plt.subplot(2, 3, 4)
plt.hist(y_probs[y_test == 0], bins=50, alpha=0.5, label='Actual Negative')
plt.hist(y_probs[y_test == 1], bins=50, alpha=0.5, label='Actual Positive')
plt.legend()
plt.title('Prediction Distribution')

# Precision-Recall Curve
plt.subplot(2, 3, 5)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.tight_layout()
plt.show()

model.save('sepsis_model.keras')
print("Model saved successfully!")


joblib.dump(feature_cols, 'features.pkl')
print("Features saved successfully!")

mean_values = df[feature_cols].mean()
joblib.dump(mean_values, 'mean_values.pkl')
print("Mean values saved successfully!")

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved!")

with open('threshold.txt', 'w') as f:
    f.write(str(best_threshold))

print("Threshold saved!")