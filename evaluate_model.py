import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.main(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data_set = torchvision.datasets.CIFAR10(
    root='dataset',
    train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    download=True
)

test_data_load = DataLoader(dataset=test_data_set, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def test_model():
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_data_load:
            images = images.to(device)
            labels = labels.to(device)
            outputs = mynet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    print(f'\nOverall Accuracy: {100 * correct / total:.2f}%')

    acc_per_class = []
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            acc_per_class.append(acc)
            print(f'{classes[i]}: {acc:.2f}%')
        else:
            acc_per_class.append(0)
            print(f'{classes[i]}: N/A (no samples)')

    plt.figure(figsize=(10, 6))
    plt.bar(classes, acc_per_class, color='skyblue')
    plt.ylim(0, 100)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy on CIFAR-10')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('accuracy_bar_chart.png')
    plt.show()

if __name__ == '__main__':
    model_path = input("Please input model file name: ").strip()
    mynet = Net().to(device)
    state_dict = torch.load(model_path, weights_only=True)
    mynet.load_state_dict(state_dict)
    mynet.eval()
    test_model()
