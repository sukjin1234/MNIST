import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple
from CNNModel import SimpleCNN


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    저장된 모델을 로드합니다.
    
    Args:
        model_path: 모델 파일 경로
        device: 사용할 디바이스 (cuda 또는 cpu)
    
    Returns:
        로드된 모델
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 설정 가져오기
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model = SimpleCNN(
            num_classes=config.get('num_classes', 10),
            channels=config.get('channels', None),
            num_blocks=config.get('num_blocks', None),
            base_channels=config.get('base_channels', 32),
            width_scale=config.get('width_scale', 2.0)
        )
    else:
        # 기본 설정으로 모델 생성
        model = SimpleCNN(num_classes=10)
    
    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def load_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    이미지를 로드하고 전처리합니다.
    
    Args:
        image_path: 이미지 파일 경로
        transform: 전처리 변환
    
    Returns:
        전처리된 이미지 텐서
    """
    try:
        # 이미지 로드 (RGBA나 RGB를 그레이스케일로 변환)
        image = Image.open(image_path)
        
        # RGBA나 RGB를 그레이스케일로 변환
        if image.mode != 'L':
            image = image.convert('L')
        
        # MNIST 형식에 맞게 리사이즈 (28x28)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 색상 반전 처리 (MNIST는 검은 배경에 흰 숫자)
        # myData 이미지가 흰 배경에 검은 숫자라면 반전 필요
        img_array = np.array(image)
        
        # 픽셀 값의 평균을 확인하여 배경이 밝은지 어두운지 판단
        # 평균이 128보다 크면 밝은 배경(반전 필요), 작으면 어두운 배경(반전 불필요)
        mean_value = img_array.mean()
        if mean_value > 128:
            # 밝은 배경이면 반전 (255 - 픽셀값)
            image = Image.fromarray(255 - img_array)
        
        # 전처리 적용
        image_tensor = transform(image)
        
        # 배치 차원 추가 [1, C, H, W]
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        print(f"이미지 로드 실패 ({image_path}): {e}")
        return None


def predict_image(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> Tuple[int, torch.Tensor]:
    """
    이미지에 대한 예측을 수행합니다.
    
    Args:
        model: 학습된 모델
        image_tensor: 전처리된 이미지 텐서
        device: 사용할 디바이스
    
    Returns:
        (예측 클래스, 확률 분포)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        
    return predicted_class, probabilities[0]


def process_myData_folder(data_folder: str, model: nn.Module, transform: transforms.Compose, device: torch.device) -> List[Tuple[str, int, int, float]]:
    """
    myData 폴더의 모든 이미지를 처리하고 예측합니다.
    
    Args:
        data_folder: myData 폴더 경로
        model: 학습된 모델
        transform: 이미지 전처리 변환
        device: 사용할 디바이스
    
    Returns:
        [(이미지 경로, 정답, 예측값, 확률), ...] 리스트
    """
    results = []
    data_path = Path(data_folder)
    
    # 각 숫자 폴더(0-9) 처리
    for label_folder in sorted(data_path.iterdir()):
        if not label_folder.is_dir():
            continue
        
        try:
            # 폴더명이 정답 레이블
            true_label = int(label_folder.name)
        except ValueError:
            print(f"경고: '{label_folder.name}' 폴더는 숫자가 아닙니다. 건너뜁니다.")
            continue
        
        # 폴더 내의 모든 이미지 파일 처리
        image_files = sorted([f for f in label_folder.iterdir() 
                             if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
        
        for image_file in image_files:
            # 이미지 로드 및 전처리
            image_tensor = load_image(str(image_file), transform)
            
            if image_tensor is None:
                continue
            
            # 예측 수행
            predicted_label, probabilities = predict_image(model, image_tensor, device)
            confidence = probabilities[predicted_label].item()
            
            results.append((str(image_file), true_label, predicted_label, confidence))
    
    return results


def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}\n")
    
    # 모델 경로
    model_path = "mnist_cnn_model_base.pth"
    
    # 모델 로드
    print(f"모델 로드 중: {model_path}")
    try:
        model = load_model(model_path, device)
        print("모델 로드 완료\n")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 전처리 설정 (학습 시 사용한 것과 동일)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # myData 폴더 처리
    data_folder = "myData"
    print(f"데이터 폴더 처리 중: {data_folder}\n")
    
    results = process_myData_folder(data_folder, model, transform, device)
    
    # 결과 출력
    print("=" * 80)
    print(f"{'이미지 파일':<50} {'정답':<8} {'예측':<8} {'확률':<10} {'정확도':<8}")
    print("=" * 80)
    
    correct_count = 0
    total_count = len(results)
    
    for image_path, true_label, predicted_label, confidence in results:
        is_correct = "✓" if true_label == predicted_label else "✗"
        if true_label == predicted_label:
            correct_count += 1
        
        # 파일명만 표시 (전체 경로가 너무 길 경우)
        filename = os.path.basename(image_path)
        print(f"{filename:<50} {true_label:<8} {predicted_label:<8} {confidence*100:.2f}%{'':<6} {is_correct:<8}")
    
    print("=" * 80)
    print(f"\n총 이미지 수: {total_count}")
    print(f"정확히 예측한 수: {correct_count}")
    print(f"정확도: {correct_count/total_count*100:.2f}%")


if __name__ == "__main__":
    main()

