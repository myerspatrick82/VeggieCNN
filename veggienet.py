import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyperparameters
    num_epoch = 3
    batch_size = 128
    learning_rate = 0.001

    # transform for data

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                                    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])])

    path = r".\Vegetables\Augmented"

    dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    indices = list(range(len(dataset)))
    labels = [dataset.targets[i] for i in indices]
    classes = dataset.classes

    train_indices, test_indices = train_test_split(indices, test_size=0.1, stratify=labels, random_state=42)

    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    # Verify distribution
    train_labels = [dataset.targets[i] for i in train_indices]
    test_labels = [dataset.targets[i] for i in test_indices]
    print("Train distribution:", Counter(train_labels))
    print("Test distribution:", Counter(test_labels))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # image, label = next(iter(dataloader))
    # plt.imshow(image[0][0])
    # print(label)
    # plt.show()

    # img = Image.open(path)
    # transform1 = transforms.Resize((224, 224))
    # resized_img = transform1(img)
    # plt.imshow(resized_img)
    # plt.show()

    print(len(train_loader))

    """
    Dimensionality reduction issue here
    To figure out how to not reduce dimensionality use this formula:
    P = (K - 1) / 2 --- ex. P = (5 - 1) / 2 = 4 / 2 = 2 = P
    """

    class VeggieNet(nn.Module):
        def __init__(self):
            super(VeggieNet, self).__init__() # 224 x 224 x 3
            self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding=2) # 224 x 224 x 16 (16 channels now)
            self.pool = nn.MaxPool2d(2, 2) # 112 x 112 x 16
            self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2) # 112 x 112 x 32 (32 channels)
            self.fc1 = nn.Linear(56*56*32, 128) # after applying another pooling layer this would become 56 x 56 x 32
            self.fc2 = nn.Linear(128, 6) # for 6 classes

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 56*56*32)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
            
    model = VeggieNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # training loop

    # Image.MAX_IMAGE_PIXELS = None
    print(torch.cuda.is_available())
    print('Started Training')
    total_steps = len(train_loader)
    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)   

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0 or i == num_epoch:
                print(f'epoch {epoch+1} / {num_epoch}, Step {i+1}/{total_steps}, loss: {loss.item():.4f} ')

    print('Finished Training')

    print('Beginning Testing')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(classes))]
        n_class_samples = [0 for i in range(len(classes))]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # print(labels)
            outputs = model(images)
            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            # print(predicted)
            # print(labels)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(len(classes)):
            if n_class_samples[i] > 0:  
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classes[i]}: {acc:.2f} %')
            else:
                print(f'Accuracy of {classes[i]}: No samples in test set')

        # save the model
        # torch.save(model, 'veggieCNN.pth')


if __name__ == "__main__":
    main()