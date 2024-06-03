import base64, numpy as np
import cv2
from ultralytics import YOLO

def get_predictions(b64imgstring):
    # Decode base64 string to image
    image_bytes = base64.b64decode(b64imgstring)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (640, 640))

    # Get predictions
    model=YOLO("best.pt")
    results = model.predict(source=image)
    boxes = results[0].boxes
    image = results[0].plot()
    print(len(results[0].boxes.conf))
    # Convert image to base64 string
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "fire": len(boxes.cls) > 0,
        "confidence": results[0].boxes.conf if len(results[0].boxes.conf) > 0 else None,
        "image": jpg_as_text,
    }