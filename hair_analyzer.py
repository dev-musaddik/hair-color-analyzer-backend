import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks  # ✅ Full path for safe globals

# ✅ Allow YOLO SegmentationModel for safe PyTorch deserialization
torch.serialization.add_safe_globals([tasks.SegmentationModel])

# Load the YOLOv8 segmentation model safely
model = YOLO("yolov8n-seg.pt", device="cpu")  # force CPU on Render or GPU if available

def get_dominant_color(image: Image.Image) -> tuple:
    """Get dominant color of an image ignoring pure black pixels."""
    image = image.resize((100, 100))
    np_image = np.array(image)
    np_image = np_image.reshape((np_image.shape[0] * np_image.shape[1], 3))

    # Ignore black pixels
    non_black_pixels = np_image[np.any(np_image != [0, 0, 0], axis=1)]
    if len(non_black_pixels) == 0:
        return (0, 0, 0), "#000000"

    # Use KMeans clustering
    n_clusters = min(3, len(non_black_pixels))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(non_black_pixels)

    # Find the most frequent cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = unique[np.argmax(counts)]
    dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(int)

    rgb_color = tuple(int(c) for c in dominant_color)
    hex_color = '#%02x%02x%02x' % rgb_color
    return rgb_color, hex_color

def analyze_hair_color(image_bytes: bytes) -> dict:
    """
    Detects hair region and returns dominant hair color.
    :param image_bytes: image in bytes
    :return: dict with RGB and HEX color
    """
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)

    # Perform segmentation
    results = model(np_image)

    # Collect masks for person(s)
    person_masks = []
    if results[0].masks is not None:
        for i, c in enumerate(results[0].boxes.cls):
            if model.names[int(c)] == 'person':
                person_masks.append(results[0].masks.data[i].cpu().numpy())

    if not person_masks:
        return {"error": "No person detected in the image."}

    # Combine masks
    combined_mask = np.sum(person_masks, axis=0)
    combined_mask = (combined_mask > 0).astype(np.uint8)
    combined_mask = cv2.resize(
        combined_mask, (np_image.shape[1], np_image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # Get bounding boxes for person(s)
    person_boxes = []
    for i, c in enumerate(results[0].boxes.cls):
        if model.names[int(c)] == 'person':
            person_boxes.append(results[0].boxes.xyxy[i].cpu().numpy())

    if not person_boxes:
        return {"error": "No person detected in the image."}

    # Assume first person for hair
    box = person_boxes[0]
    x1, y1, x2, y2 = map(int, box)

    # Take top 25% of bounding box as hair region
    hair_region_y_end = y1 + int((y2 - y1) * 0.25)
    hair_mask = np.zeros_like(combined_mask)
    hair_mask[y1:hair_region_y_end, x1:x2] = 1

    final_hair_mask = combined_mask * hair_mask

    if np.sum(final_hair_mask) == 0:
        return {"error": "Could not isolate hair region."}

    # Apply mask to get hair-only image
    hair_only_image = cv2.bitwise_and(np_image, np_image, mask=final_hair_mask)
    hair_pil_image = Image.fromarray(hair_only_image)

    # Get dominant color
    rgb_color, hex_color = get_dominant_color(hair_pil_image)

    return {
        "dominant_color_rgb": rgb_color,
        "dominant_color_hex": hex_color,
    }
