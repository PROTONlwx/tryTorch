import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])

validation_test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = validation_test_transforms)
test_data = datasets.ImageFolder(test_dir, transform = validation_test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

device = ("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)
for param in model.parameters():
    param.requiers_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 512)),
    ("relu1", nn.ReLU()),
    ("dp1", nn.Dropout(0.2)),
    ("fc2", nn.Linear(512, 102)),
    ("output", nn.LogSoftmax(dim=1))
]))
model.classifier = classifier
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
epochs = 1
steps = 0
running_loss = 0
print_every = 40
for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(valid_loader):.3f}.. "
                  f"Test accuracy: {accuracy / len(valid_loader):.3f}")
            running_loss = 0
            model.train()

print(f"finished training")

test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Test loss: {test_loss / len(test_loader):.3f}.. "
          f"Test accuracy: {accuracy / len(test_loader):.3f}")

#save trained model
model.class_to_idx = train_data.class_to_idx
checkpoint = {'arch': 'densenet121',
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'epochs': epochs,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}
torch.save(checkpoint,'checkpointtest.pth')