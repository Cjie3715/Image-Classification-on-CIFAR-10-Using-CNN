import torch.optim.adam
from torch.utils.data import DataLoader
import torchvision
from torch import nn
import torch
import matplotlib.pyplot as plt  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_set = torchvision.datasets.CIFAR10(root='dataset',train=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32,32),padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
]),download=True)

test_data_set = torchvision.datasets.CIFAR10(root='dataset',train=False,transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32,32),padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
]),download=True)

data_size = len(data_set)
test_data_size = len(test_data_set)

data_load = DataLoader(dataset=data_set,batch_size=64,shuffle=True,drop_last=True)
test_data_load = DataLoader(dataset=test_data_set,batch_size=64,shuffle=True,drop_last=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(128)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128*4*4,1024),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256,10)
        )

    def forward(self,x):
        return self.fc(self.main(x))
    
mynet = Net()
mynet = mynet.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 1e-3
optim = torch.optim.Adam(mynet.parameters(),lr=learning_rate)

train_step = 0
epoch = 20 

train_loss_history = []
test_loss_history = []
steps_history = []

if __name__ == '__main__':
    for i in range(epoch):
        print(f'第{i+1}輪訓練')
        mynet.train()
        epoch_train_loss = 0.0
        
        for j,(imgs,targets) in enumerate(data_load):
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = mynet(imgs)
            loss = loss_fn(outputs,targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_train_loss += loss.item()
            train_step += 1
            
            if train_step % 100 == 0:
                print(f'Num of train:{train_step},loss={loss.item()}')
                train_loss_history.append(loss.item())
                steps_history.append(train_step)
        
        avg_train_loss = epoch_train_loss / len(data_load)
        
        mynet.eval()
        accuracy_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for j,(imgs,targets) in enumerate(test_data_load):
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = mynet(imgs)
                loss = loss_fn(outputs, targets)
                test_loss += loss.item()

                accuracy = (outputs.argmax(axis=1) == targets).sum()
                accuracy_total += accuracy

            avg_test_loss = test_loss / len(test_data_load)
            test_loss_history.append(avg_test_loss)
            
            print(f'epoch:{i+1}, acc:{accuracy_total/test_data_size}, train_loss:{avg_train_loss:.4f}, test_loss:{avg_test_loss:.4f}')
            torch.save(mynet.state_dict(),f'CIAFR_10_{i+1}_acc_{accuracy_total/test_data_size}.pth')
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps_history, train_loss_history, label='Train Loss (per 100 steps)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Training Steps')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch+1), [sum(train_loss_history[i*len(data_load)//100:(i+1)*len(data_load)//100])/(len(data_load)//100) for i in range(epoch)], 
             label='Train Loss (per epoch)')
    plt.plot(range(1, epoch+1), test_loss_history, label='Test Loss (per epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Test Loss vs. Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_curves.png')  
    plt.show()