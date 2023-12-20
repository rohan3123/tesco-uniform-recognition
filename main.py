import cv2
import numpy as np

# Load pre-trained modle for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load images of Tesco workers for recognition
tesco_worker_images = {
    'Employee1': cv2.imread('employees/1.jpg', cv2.IMREAD_GRAYSCALE),
    'Employee2': cv2.imread('employees/2.jpg', cv2.IMREAD_GRAYSCALE),
    'Employee3': cv2.imread('employees/3.jpg', cv2.IMREAD_GRAYSCALE),
    'Employee4': cv2.imread('employees/4.jpg', cv2.IMREAD_GRAYSCALE),
    'Employee5': cv2.imread('employees/5.jpg', cv2.IMREAD_GRAYSCALE),
}