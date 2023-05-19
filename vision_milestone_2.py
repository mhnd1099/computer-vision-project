import cv2
import numpy as np

# Load the image
img = cv2.imread("1.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get a binary image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Load the template images for digits
templates = []
for i in range(10):
    template = cv2.imread(f"train/{i}.png", cv2.IMREAD_GRAYSCALE)
    templates.append(template)

# Loop over all contours
for cnt in contours:
    # Get the bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    # Draw the bounding box on the original image
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract the region of interest (ROI) containing the number
    roi = thresh[y:y + h, x:x + w]

    # Resize the ROI to match the template size
    roi_resized = cv2.resize(roi, (20, 30))

    # Match the ROI with each template and find the best match
    max_match_val = -np.inf
    digit = None
    for i, template in enumerate(templates):
        # Resize the template to match the ROI size
        template_resized = cv2.resize(template, (20, 30))

        match_val = cv2.matchTemplate(roi_resized, template_resized, cv2.TM_CCOEFF_NORMED)
        if match_val > max_match_val:
            max_match_val = match_val
            digit = i

    # Draw the recognized digit on the original image
    cv2.putText(img, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
