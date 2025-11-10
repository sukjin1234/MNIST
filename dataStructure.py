import torchvision.transforms as transforms

# 데이터 전처리 규칙 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

from torchvision.datasets import MNIST

# 학습용 데이터셋 불러오기
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

# 테스트용 데이터셋 불러오기
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
print(len(train_dataset))
print(len(test_dataset))

from torch.utils.data import DataLoader

# 학습용 DataLoader 생성 (데이터를 섞음)
train_loader = DataLoader(dataset=train_dataset, batch_size = 32, shuffle=True)

# 테스트용 DataLoader 생성 (데이터를 섞지 않음)
test_loader = DataLoader(dataset=test_dataset, batch_size = 32, shuffle=False)

for (X_train, y_train) in train_loader:
  print('X_train:', X_train.size(), 'type:', X_train.type())
  print('y_train:', y_train.size(), 'type:', y_train.type())
  break

import matplotlib.pyplot as plt
import numpy
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))

for i in range(10):
  plt.subplot(1,10,i+1)
  plt.axis('off')
  plt.imshow(numpy.reshape(X_train.numpy()[i], (28,28)), cmap='gray')
  plt.title('Class: ' + str(y_train.numpy()[i]))
plt.show()