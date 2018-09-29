import cv2
import numpy as np


def normalizeRGB(img):
    norm = np.zeros(img.shape, np.float32)
    sum = np.sum(img, axis=2) + 1.0
    norm[:, :, 0] = img[:, :, 0] / sum * 255.0
    norm[:, :, 1] = img[:, :, 1] / sum * 255.0
    norm[:, :, 2] = img[:, :, 2] / sum * 255.0
    norm = cv2.convertScaleAbs(norm)
    return norm


def normalizeHSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    norm = np.zeros(hsv.shape, np.float32)
    sum = np.sum(hsv, axis=2) + 1.0
    norm[:, :, 0] = hsv[:, :, 0] / sum * 180.0
    norm[:, :, 1] = hsv[:, :, 1] / sum * 255.0
    norm[:, :, 2] = hsv[:, :, 2] / sum * 255.0
    norm = cv2.convertScaleAbs(norm)
    return norm
