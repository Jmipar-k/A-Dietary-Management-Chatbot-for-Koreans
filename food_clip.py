from medical import lora_clip
import clip
from labels import food_labels
import torch
from PIL import Image

def predict_top_class(clip_model, image_tensor):
    """
    Predict the top class for a given image using CLIP model.

    Args:
        clip_model: CLIP model (with LoRA if applied)
        class_names (list): List of class names
        image_tensor (torch.Tensor): Preprocessed image tensor

    Returns:
        str: The predicted class name
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.eval()  # 모델을 평가 모드로 설정

    # 1. 클래스 이름 템플릿 생성
    template = "A photo of a {}"
    texts = [template.format(classname.replace('_', ' ')) for classname in food_labels]

    # 2. 텍스트 토큰화 및 특징 추출
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts_tokenized = clip.tokenize(texts).to(device)  # 텍스트 토큰화
            text_features = clip_model.encode_text(texts_tokenized)  # 텍스트 특징 추출
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 정규화

    # 3. 이미지 특징 추출
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(image_tensor)  # 이미지 특징 추출
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 정규화

    # 4. 코사인 유사도 계산
    with torch.no_grad():
        cosine_similarity = image_features @ text_features.t()  # 코사인 유사도 계산
        logits = (100 * cosine_similarity).softmax(dim=-1)  # 소프트맥스 적용

    # 5. 가장 높은 확률의 클래스 추출
    top_class_idx = logits.argmax(dim=-1).item()  # 가장 높은 확률의 인덱스
    predicted_class = food_labels[top_class_idx]  # 클래스 이름 매핑

    return predicted_class