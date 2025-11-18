import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import math

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

DROPOUT_FEATURE = 0.1
DROPOUT_CLASSIFIER = 0.3


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(DROPOUT_FEATURE)
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = out + identity
        out = self.relu2(out)
        return out


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, num_blocks: int = 3, 
                 base_channels: int = 32, width_scale: float = 2.0,
                 channels: List[int] = None):
        """
        Args:
            num_classes: 출력 클래스 개수
            num_blocks: ResidualBlock의 개수 (깊이)
            base_channels: 첫 번째 블록의 출력 채널 수 (너비 기본값)
            width_scale: 각 블록마다 채널 수를 늘리는 배율
            channels: 명시적으로 채널 수 리스트 지정 (None이면 base_channels와 width_scale로 계산)
        """
        super().__init__()
        
        # 채널 수 계산
        if channels is None:
            # base_channels부터 시작해서 width_scale만큼 늘려가며 계산
            channels = [int(base_channels * (width_scale ** i)) for i in range(num_blocks)]
        else:
            # 명시적으로 채널 수가 주어진 경우
            num_blocks = len(channels)
        
        # 입력 채널: MNIST는 1채널
        in_channels = 1
        
        # ResidualBlock들을 동적으로 생성
        self.blocks = nn.ModuleList()
        for i, out_channels in enumerate(channels):
            # 모든 블록에 stride=2 적용 (다운샘플링)
            # 입력: 28x28 -> 블록1: 14x14 -> 블록2: 7x7 -> 블록3: 4x4 (또는 3x3)
            stride = 2
            self.blocks.append(ResidualBlock(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        
        # AdaptiveAvgPool2d로 고정 크기로 변환
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
        
        # 최종 채널 수 계산 (마지막 블록의 출력 채널)
        final_channels = channels[-1]
        feature_dim = final_channels * 3 * 3  # AdaptiveAvgPool2d(3,3) 이후 크기
        
        # Classifier: 특징 차원을 적절히 조절
        classifier_hidden = max(128, final_channels // 2)  # 최소 128, 또는 채널의 절반
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT_CLASSIFIER),
            nn.Linear(feature_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_CLASSIFIER),
            nn.Linear(classifier_hidden, num_classes),
        )
        
        # 디버깅용 정보 저장
        self.num_blocks = num_blocks
        self.channels = channels
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResidualBlock들을 순차적으로 통과
        for block in self.blocks:
            x = block(x)
        # 고정 크기로 변환
        x = self.adaptive_pool(x)
        # Classifier 통과
        x = self.classifier(x)
        return x


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


class LearningRateFinder:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def find_lr(self, dataloader: DataLoader, start_lr: float = 1e-7, end_lr: float = 10, num_iter: int = 100) -> Tuple[List[float], List[float]]:
        lrs = []
        losses = []
        
        # 원래 학습률 저장
        original_lr = self.optimizer.param_groups[0]['lr']
        
        # 학습률 범위 설정
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        
        # 모델을 학습 모드로 설정
        self.model.train()
        
        # 데이터로더에서 배치 가져오기
        data_iter = iter(dataloader)
        
        for i in range(num_iter):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 현재 학습률 계산
            current_lr = start_lr * (lr_mult ** i)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # 순전파
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            lrs.append(current_lr)
            losses.append(loss.item())
            
            # 손실이 너무 커지면 중단
            if loss.item() > 4 * min(losses):
                break
        
        # 원래 학습률 복원
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = original_lr
            
        return lrs, losses


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        self.best_weights = model.state_dict().copy()


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


def plot_lr_finder(lrs: List[float], losses: List[float]) -> None:
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    
    # 최적 학습률 추정 (손실이 가장 급격히 감소하는 지점)
    if len(losses) > 10:
        smoothed_losses = np.convolve(losses, np.ones(10)/10, mode='valid')
        smoothed_lrs = lrs[9:]
        if len(smoothed_losses) > 0:
            min_idx = np.argmin(smoothed_losses)
            optimal_lr = smoothed_lrs[min_idx]
            plt.axvline(x=optimal_lr, color='red', linestyle='--', label=f'Optimal LR: {optimal_lr:.2e}')
            plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss (log scale)')
    plt.title('Learning Rate Finder (Log Scale)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses: List[float], train_accs: List[float], 
                         val_losses: List[float], val_accs: List[float]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 사용 예시:
# ============================================================================
# 
# # 작은 모델 (얕고 좁음): 2개 블록, 기본 채널 16
# small_model = SimpleCNN(num_blocks=2, base_channels=16, width_scale=2.0)
# 
# # 기본 모델 (기존 설정과 동일): 3개 블록, 기본 채널 32
# default_model = SimpleCNN()  # 또는 SimpleCNN(num_blocks=3, base_channels=32, width_scale=2.0)
# 
# # 큰 모델 (깊고 넓음): 5개 블록, 기본 채널 64
# large_model = SimpleCNN(num_blocks=5, base_channels=64, width_scale=1.5)
# 
# # 매우 큰 모델 (매우 깊고 매우 넓음): 6개 블록, 기본 채널 128
# huge_model = SimpleCNN(num_blocks=6, base_channels=128, width_scale=1.5)
# 
# # 명시적으로 채널 수 지정
# custom_model = SimpleCNN(channels=[32, 48, 64, 96, 128])  # num_blocks는 자동으로 5로 설정됨
# 
# ============================================================================


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 기본 모델 생성 (기존과 동일한 설정)
    # 다른 설정을 테스트하려면 아래 예시를 참고:
    # - 작은 모델: SimpleCNN(num_blocks=2, base_channels=16)
    # - 큰 모델: SimpleCNN(num_blocks=5, base_channels=64, width_scale=1.5)
    # - 사용자 정의: SimpleCNN(channels=[32, 64, 128, 256])
    model = SimpleCNN().to(device)
    
    # 모델 구조 정보 출력
    print(f"\n모델 구조:")
    print(f"  - 블록 개수 (깊이): {model.num_blocks}")
    print(f"  - 채널 수 (너비): {model.channels}")
    print(f"  - 특징 벡터 차원: {model.feature_dim}")
    
    # 모델 파라미터 개수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 총 파라미터 수: {total_params:,}")
    print(f"  - 학습 가능 파라미터: {trainable_params:,}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 학습률 탐색
    print("학습률 탐색 중...")
    lr_finder = LearningRateFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.find_lr(train_loader, start_lr=1e-7, end_lr=1, num_iter=200)
    
    # 최적 학습률 추정
    if len(losses) > 10:
        smoothed_losses = np.convolve(losses, np.ones(10)/10, mode='valid')
        smoothed_lrs = lrs[9:]
        if len(smoothed_losses) > 0:
            min_idx = np.argmin(smoothed_losses)
            optimal_lr = smoothed_lrs[min_idx]
            print(f"추정된 최적 학습률: {optimal_lr:.2e}")
            
            # 최적 학습률로 설정
            for param_group in optimizer.param_groups:
                param_group['lr'] = optimal_lr
    
    plot_lr_finder(lrs, losses)
    
    # 스케줄러 설정 (ReduceLROnPlateau 사용)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Early Stopping 설정
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    # 학습 기록 저장
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    epochs = 50
    best_val_acc = 0.0
    
    print(f"\n학습 시작 (최대 {epochs} 에포크)")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # 기록 저장
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 최고 성능 업데이트
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # 스케줄러 업데이트 (검증 손실 기반)
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc*100:.2f}% | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}% | "
              f"lr: {current_lr:.2e}")
        
        # Early Stopping 체크
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\n최고 검증 정확도: {best_val_acc*100:.2f}%")
    
    # 학습 과정 시각화
    plot_training_history(train_losses, train_accs, val_losses, val_accs)

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

