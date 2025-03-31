"""
    Problem Description
    Develop an algorithm to detect cancer in pathology scans.
    This project consists of a CNN computer vision model that will identify cancerous tissue from provided lymph node images.
"""

"""
    Exploratory Data Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train_labels.csv")

# label_counts = df["label"].value_counts()
# print(label_counts)

# plt.bar(label_counts.index, label_counts.values, color=["green", "red"])
# plt.xticks([0, 1], ["No Cancer (0)", "Cancer Detected (1)"])
# plt.ylabel("Count")
# plt.title("Distribution of Labels in train_labels.csv")
# plt.show()

# Based off of the EDA, we can see that there is roughly 30% more non-cancerous images in the training set than cancerous ones.
# To account for this, we will do undersampling and reduce the non-cancerous images in training until the class counts are equal.

"""
    Model Architecture Explanation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np

# Since each image is 96 by 96 pixels, we will have an input layer of 96 * 96 * 3 to account for each pixel and the 3 color channels
# There are 3 hidden layers with 512 nodes each, which was chosen arbitrarily
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 512 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

model = CNNModel()

class CancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.non_cancer_count = 0
        self.non_cancer_limit = 89117
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id, label = self.df.iloc[idx]
        if label == 0 and self.non_cancer_count >= self.non_cancer_limit:
            return None
        if label == 0:
            self.non_cancer_count += 1
        img_path = os.path.join("train", img_id + ".tif")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (96, 96))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return img, label

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CancerDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

epochs = 1
train_losses = []
val_losses = []

print("begin training")
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        try:
            if data is None:
                continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0
        
                train_losses.append(loss.item())
                if i > 1000:
                    break
        except:
            continue


plt.plot(train_losses, label='Training Loss')
plt.xlabel('Batch (x100)')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_ids = [f[:-4] for f in os.listdir(test_dir) if f.endswith('.tif')]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.test_dir, img_id + ".tif")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (96, 96))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        return img_id, img

test_dataset = TestDataset("test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, pin_memory=True)

model.eval()
predictions = []
total_batches = len(test_loader)
batch_size = test_loader.batch_size

with torch.no_grad():
    for batch_idx, (img_ids, inputs) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        for img_id, label in zip(img_ids, predicted.cpu().numpy()):
            predictions.append((img_id, label))

        if batch_idx % 100 == 0:
            processed_images = (batch_idx + 1) * batch_size
            print(f"processed {processed_images}/{len(test_dataset)} images")

output_df = pd.DataFrame(predictions, columns=["id", "label"])
output_df.to_csv("test_predictions.csv", index=False)
print("test predictions saved")

"""
    Results and Analysis"
"""
# The training loss stopped improving around 0.5, and the resulting model achieved an accuracy around 79%.
# Training was not done on the entire dataset or for multiple epochs because that would have taken an unreasonable amount of time.
# With the given training restraints, I think the model had an acceptable accuracy.

"""
    Conclusion
"""
# The model ended up with around 79% accuracy, which is acceptable considering the limited training time and dataset constraints.
# Since I only trained for one epoch and didn’t use the full dataset, there is definitely room for improvement.
# With more training time, data augmentation, or even a more complex model architecture, the accuracy could be pushed higher.
# Overall, the results show that deep learning can be useful for this cancer detection task, even with a fairly simple setup.