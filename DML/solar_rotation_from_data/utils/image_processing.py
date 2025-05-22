import cv2
import numpy as np

'''
This utility is responsible for preprocessing the images for feature detection.
    functions:
        - detect sunspots
        - detect solar center and radius
'''

def detect_sunspots(image_path, sunspot_threshold=25, min_area=16, kernel_size = 3):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Find solar disk
    solar_center, solar_radius = detect_solar_threshold(gray_img)
    
    # 2. Create solar mask (95% radius)
    mask = np.zeros_like(gray_img)
    cv2.circle(mask, solar_center, int(solar_radius * 0.95), 255, -1)
    
    # 3. Sunspot-specific processing
    blurred = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    
    # Focused adaptive thresholding
    sunspots = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 51, sunspot_threshold)
    
    # Apply solar mask
    sunspots = cv2.bitwise_and(sunspots, mask)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(sunspots, cv2.MORPH_OPEN, kernel)
    
    # Find and filter contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append((cx, cy))
    
    return img, centroids, solar_center, solar_radius


def detect_solar_threshold(gray_img):
    _, solar_thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(solar_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solar_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(solar_contour)
    solar_center = (int(cx), int(cy))
    solar_radius = int(radius)
    return solar_center, solar_radius

