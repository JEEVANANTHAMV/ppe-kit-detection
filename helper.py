import cv2

def map_layout(image, result):
    """
    Draws bounding boxes and labels on the image based on inference results.

    Args:
        image (np.ndarray): Input image in BGR format.
        result (dict): Inference result with predictions.

    Returns:
        np.ndarray: Annotated image.
    """
    if image is None:
        print("Failed to load image for processing.")
        return None

    for pred in result.get("predictions", []):
        x_center = pred["x"]
        y_center = pred["y"]
        width = pred["width"]
        height = pred["height"]
        class_label = pred["class"]
        confidence = pred["confidence"]

        # Convert center coordinates to top-left and bottom-right coordinates
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Prepare label with class and confidence
        label = f"{class_label}: {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Draw background for label text
        cv2.rectangle(image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image
