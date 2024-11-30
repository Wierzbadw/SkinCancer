import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import torch
import torchvision.models as models
import torchvision.transforms as transforms


img_dir = r'C:\Users\User\PycharmProjects\SkinCancer\archive\images'
img_list = os.listdir(img_dir)[:500]  # Pobierz pierwsze 100 obrazów

# Inicjalizacja modelu CNN (ResNet18)
model = models.resnet18(pretrained=True)  # Pretrenowany model ResNet18
model.fc = torch.nn.Identity()  # Usunięcie ostatniej warstwy klasyfikacyjnej
model.eval()  # Przełącz na tryb ewaluacji

# Transformacje dla wejścia do CNN
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Funkcja do przetwarzania obrazu i wyciągania cech
def process_image(image_path):

    image = Image.open(image_path).convert("RGB")  # Otwórz jako obraz w przestrzeni RGB
    image_resized = image.resize((224, 224))  # Normalizacja rozmiaru

    img_array = np.array(image_resized)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Konwersja na skalę szarości

    # Analiza kształtu: konwersja na obraz binarny
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Jeśli brak konturów, zwróć 0
    if not contours:
        return {
            'Circumference': 0, 'Area': 0, 'ShapeFactor': 0,
            'MeanColor': 0, 'StdColor': 0,
            'TextureRoughness': 0, 'TexturePattern': 0,
            'EdgeStrength': 0, 'CornerDetection': 0,
            'TextureRoughnessCNN': 0, 'TexturePatternCNN': 0,
            'ColorGradient': 0, 'ColorTransition': 0,
            'DominantColorFeature': 0, 'ColorDistributionPattern': 0,
            'GradientDistribution': 0
        }

    # Znajdź największy kontur
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    circumference = cv2.arcLength(largest_contour, True)

    # Współczynnik kształtu: Shape Factor = Circumference^2 / (4π * Area)
    shape_factor = (circumference**2) / (4 * np.pi * area) if area > 0 else 0

    # Tekstura
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    texture_roughness = graycoprops(glcm, 'contrast')[0, 0]  # Kontrast tekstury
    texture_pattern = graycoprops(glcm, 'energy')[0, 0]  # Jednorodność tekstury

    # Wyodrębnianie cech za pomocą CNN
    input_tensor = cnn_transform(image_resized).unsqueeze(0)  # Transformacja i dodanie wymiaru batch
    with torch.no_grad():
        cnn_features = model(input_tensor)  # Wyodrębnij cechy

    # Cechy wyodrębnione z CNN
    edge_strength_cnn = cnn_features[0, 0].item()  # Siła krawędzi
    corner_detection_cnn = cnn_features[0, 1].item()  # Wykrywanie rogów
    texture_roughness_cnn = cnn_features[0, 2].item()  # Szorstkość tekstury
    texture_pattern_cnn = cnn_features[0, 3].item()  # Wzory tekstury
    color_gradient_cnn = cnn_features[0, 4].item()  # Gradient koloru
    color_transition_cnn = cnn_features[0, 5].item()  # Przejścia kolorów
    dominant_color_feature_cnn = cnn_features[0, 6].item()  # Dominujące kolory
    color_distribution_pattern_cnn = cnn_features[0, 7].item()  # Rozkład kolorów
    gradient_distribution_cnn = cnn_features[0, 8].item()  # Rozkład gradientów

    return {
        'Circumference': circumference,
        'Area': area,
        'ShapeFactor': shape_factor,
        'TextureRoughness': texture_roughness,
        'TexturePattern': texture_pattern,
        'EdgeStrengthCNN': edge_strength_cnn,
        'CornerDetectionCNN': corner_detection_cnn,
        'TextureRoughnessCNN': texture_roughness_cnn,
        'TexturePatternCNN': texture_pattern_cnn,
        'ColorGradientCNN': color_gradient_cnn,
        'ColorTransitionCNN': color_transition_cnn,
        'DominantColorFeatureCNN': dominant_color_feature_cnn,
        'ColorDistributionPatternCNN': color_distribution_pattern_cnn,
        'GradientDistributionCNN': gradient_distribution_cnn
    }

# Przetwórz obrazy i zbierz cechy
shape_features = []
for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    features = process_image(img_path)
    features['Image'] = img_name
    shape_features.append(features)


df_shape_features = pd.DataFrame(shape_features)

# Konfiguracja wyświetlania wszystkich kolumn w DataFrame
pd.set_option('display.max_columns', None)

print(df_shape_features.head(500))
