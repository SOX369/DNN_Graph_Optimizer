import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE


def train_model(model, train_loader, epochs=10):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    print(f"Training on {DEVICE}...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")


def evaluate_performance(model, test_loader):
    model = model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0

    # 测量推理延迟
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()

    accuracy = 100 * correct / total
    avg_latency = (end_time - start_time) / len(test_loader) * 1000  # ms per batch

    return accuracy, avg_latency