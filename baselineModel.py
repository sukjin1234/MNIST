import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 데이터 전처리 규칙 정의
transform = transforms.Compose([
    #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])


# 학습용 데이터셋 불러오기
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

# 테스트용 데이터셋 불러오기
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

# 학습용 DataLoader 생성 (데이터를 섞음)
train_loader = DataLoader(dataset=train_dataset, batch_size = 32, shuffle=True)

# 테스트용 DataLoader 생성 (데이터를 섞지 않음)
test_loader = DataLoader(dataset=test_dataset, batch_size = 32, shuffle=False)


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPBaseline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}%")

    # 시각화를 위한 정/오분류 예시 수집
    correct_examples = []  # (image_tensor_cpu, pred, true)
    wrong_examples = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            for i in range(inputs.size(0)):
                if preds[i] == targets[i] and len(correct_examples) < 10:
                    correct_examples.append((inputs[i].cpu(), int(preds[i].item()), int(targets[i].item())))
                elif preds[i] != targets[i] and len(wrong_examples) < 10:
                    wrong_examples.append((inputs[i].cpu(), int(preds[i].item()), int(targets[i].item())))

                if len(correct_examples) >= 10 and len(wrong_examples) >= 10:
                    break
            if len(correct_examples) >= 10 and len(wrong_examples) >= 10:
                break

    def show_grid(examples, title):
        plt.figure(figsize=(12, 2.5))
        plt.suptitle(title)
        for idx, (img, pred, true) in enumerate(examples):
            plt.subplot(1, 10, idx + 1)
            plt.axis('off')
            # img shape: [1, 28, 28]
            plt.imshow(img.squeeze(0), cmap='gray')
            plt.title(f"p:{pred}\nt:{true}", fontsize=9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if correct_examples:
        show_grid(correct_examples, "Correctly Classified (10 examples)")
    if wrong_examples:
        show_grid(wrong_examples, "Misclassified (10 examples)")
    plt.show()


if __name__ == "__main__":
    main()

