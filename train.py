import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib

# 1️⃣ Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# 2️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Feature scaling (CRITICAL)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4️⃣ Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 5️⃣ Define model (MUST MATCH app.py)
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# 6️⃣ Loss & optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7️⃣ Training loop
epochs = 100

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 8️⃣ Evaluation (optional but good)
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)

print("Final Test Loss:", test_loss.item())

# 9️⃣ Save model
torch.save(model.state_dict(), "model.pth")

# 🔥 10️⃣ Save scaler (VERY IMPORTANT)
joblib.dump(scaler, "scaler.pkl")

print("✅ Model saved as model.pth")
print("✅ Scaler saved as scaler.pkl")