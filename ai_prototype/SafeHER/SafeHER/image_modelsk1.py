from transformers import pipeline
from PIL import Image

# Load NSFW detection model
classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection"
)

# Input
image_path = input("Enter image path: ").strip().replace('"', '')

# Open image
image = Image.open(image_path)

# Predict
result = classifier(image)

print("\nNSFW Result:")
for r in result:
    print(r)