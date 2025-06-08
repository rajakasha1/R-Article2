import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train symbolic models
dt_clf = DecisionTreeClassifier(random_state=42).fit(X_scaled, y)
lr_clf = LogisticRegression(max_iter=1000).fit(X_scaled, y)
rf_clf = RandomForestClassifier(random_state=42).fit(X_scaled, y)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

input_dim = X.shape[1]
nn_model = SimpleNN(input_dim)

# Train the neural network
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.01)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = nn_model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# Neural network prediction function
def nn_predict_proba(x_input):
    nn_model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_input, dtype=torch.float32)
        logits = nn_model(x_t)
        probs = torch.softmax(logits, dim=1).numpy()
    return probs

# Streamlit UI
st.title("Breast Cancer Classification Interactive Demo")

st.write("Input patient features to get predictions from different models.")

# User input
input_features = []
for feature in feature_names:
    val = st.slider(
        f"{feature}",
        float(np.min(X[:, list(feature_names).index(feature)])), 
        float(np.max(X[:, list(feature_names).index(feature)])), 
        float(np.mean(X[:, list(feature_names).index(feature)]))
    )
    input_features.append(val)

input_array = np.array(input_features).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Model selection
model_option = st.selectbox("Choose Model", [
    "Decision Tree",
    "Logistic Regression",
    "Random Forest",
    "Neural Network",
    "Hybrid: Decision Tree + Neural Network",
    "Hybrid: Logistic Regression + Neural Network"
])

# Prediction logic
if st.button("Predict"):
    if model_option == "Decision Tree":
        pred = dt_clf.predict(input_scaled)[0]
        prob = dt_clf.predict_proba(input_scaled)[0]
    elif model_option == "Logistic Regression":
        pred = lr_clf.predict(input_scaled)[0]
        prob = lr_clf.predict_proba(input_scaled)[0]
    elif model_option == "Random Forest":
        pred = rf_clf.predict(input_scaled)[0]
        prob = rf_clf.predict_proba(input_scaled)[0]
    elif model_option == "Neural Network":
        prob = nn_predict_proba(input_scaled)[0]
        pred = np.argmax(prob)
    elif model_option == "Hybrid: Decision Tree + Neural Network":
        prob_dt = dt_clf.predict_proba(input_scaled)[0]
        prob_nn = nn_predict_proba(input_scaled)[0]
        prob = (prob_dt + prob_nn) / 2
        pred = np.argmax(prob)
    elif model_option == "Hybrid: Logistic Regression + Neural Network":
        prob_lr = lr_clf.predict_proba(input_scaled)[0]
        prob_nn = nn_predict_proba(input_scaled)[0]
        prob = (prob_lr + prob_nn) / 2
        pred = np.argmax(prob)

    st.write(f"**Prediction:** {'Malignant' if pred == 0 else 'Benign'}")
    st.write(f"**Probability (Malignant, Benign):** {prob}")