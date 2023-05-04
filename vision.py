import cv2
import numpy as np

# Load the image
img = cv2.imread("image.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get a binary image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over all contours
for cnt in contours:
    # Get the bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    # Draw the bounding box on the original image
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
