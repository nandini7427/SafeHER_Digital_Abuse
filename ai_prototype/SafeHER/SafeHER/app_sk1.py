from transformers import pipeline

# Load AI model
classifier = pipeline("text-classification", model="unitary/toxic-bert")

# Test input
text = input("Enter text: ")

# Predict
result = classifier(text)

print("\nResult:", result)