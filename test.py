from UIElementDetectorRF import UIElementDetector

# Initialize detector
detector = UIElementDetector("yolo_model.pt")

# Find button anywhere on page
button = detector.detect_button_by_text("https://payby.ma/demo/index.php", "Rechercher")


print(button.confidence)

# Close driver when done
detector.close()