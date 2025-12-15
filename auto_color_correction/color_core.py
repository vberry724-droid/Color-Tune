import cv2
import numpy as np
from dataclasses import dataclass


# ==============================
# Структура параметров стиля
# ==============================
@dataclass
class Style:
    contrast_low: float
    contrast_high: float
    sat_factor: float
    temp_shift: float


# ==============================
# Набор стилевых пресетов
# ==============================
STYLES = {
    "neutral": Style(contrast_low=1, contrast_high=99, sat_factor=1.0, temp_shift=0),
    "warm":    Style(contrast_low=1, contrast_high=99, sat_factor=1.1, temp_shift=+12),
    "cool":    Style(contrast_low=1, contrast_high=99, sat_factor=0.95, temp_shift=-12),
    "vibrant": Style(contrast_low=1, contrast_high=99, sat_factor=1.35, temp_shift=+6),
    "vintage": Style(contrast_low=5, contrast_high=95, sat_factor=0.85, temp_shift=-6),
}


# ==============================
# Баланс белого (Gray World)
# ==============================
def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    img[:, :, 0] *= avg_gray / (avg_b + 1e-6)
    img[:, :, 1] *= avg_gray / (avg_g + 1e-6)
    img[:, :, 2] *= avg_gray / (avg_r + 1e-6)

    return np.clip(img, 0, 255)


# ==============================
# Автоконтраст по перцентилям
# ==============================
def autocontrast(img: np.ndarray, low=1, high=99) -> np.ndarray:
    out = img.copy()

    for c in range(3):
        lo = np.percentile(out[:, :, c], low)
        hi = np.percentile(out[:, :, c], high)

        if hi - lo > 1e-6:
            out[:, :, c] = (out[:, :, c] - lo) * 255.0 / (hi - lo)

    return np.clip(out, 0, 255)


# ==============================
# Коррекция цветовой температуры (LAB)
# ==============================
def apply_temperature(img: np.ndarray, shift: float) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Канал b: синий ↔ жёлтый (цветовая температура)
    lab[:, :, 2] += shift
    lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ==============================
# Коррекция насыщенности
# ==============================
def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ==============================
# Основная функция ИИ-коррекции
# ==============================
def auto_color_correct(
    img: np.ndarray,
    style: str = "neutral",
    intensity: float = 80.0
) -> np.ndarray:
    """
    Автоматизированная цветокоррекция рекламной графики
    с имитацией интеллектуального анализа изображения.
    """

    params = STYLES.get(style, STYLES["neutral"])

    img_f = img.astype(np.float32)

    # 1. Баланс белого
    corrected = gray_world_white_balance(img_f)

    # 2. Автоконтраст
    corrected = autocontrast(
        corrected,
        low=params.contrast_low,
        high=params.contrast_high
    )

    # 3. Температура (LAB)
    corrected = apply_temperature(
        corrected.astype(np.uint8),
        params.temp_shift
    ).astype(np.float32)

    # 4. Насыщенность
    corrected = adjust_saturation(
        corrected,
        params.sat_factor
    ).astype(np.float32)

    # 5. Смешивание с оригиналом (интенсивность)
    alpha = np.clip(intensity / 100.0, 0.0, 1.0)
    result = cv2.addWeighted(
        corrected, alpha,
        img_f, 1 - alpha,
        0
    )

    return np.clip(result, 0, 255).astype(np.uint8)
