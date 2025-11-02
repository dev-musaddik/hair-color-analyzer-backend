import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n-seg.pt")

def get_dominant_color(image: Image.Image) -> tuple:
    # Resize image to speed up clustering
    image = image.resize((100, 100))
    # Convert image to numpy array
    np_image = np.array(image)
    # Reshape the image to be a list of pixels
    np_image = np_image.reshape((np_image.shape[0] * np_image.shape[1], 3))

    # Remove black pixels (background) from the image data
    non_black_pixels = np_image[np.any(np_image != [0, 0, 0], axis=1)]

    if len(non_black_pixels) == 0:
        # If all pixels are black, return black
        return (0, 0, 0), "#000000"

    # Adjust n_clusters based on the number of unique colors
    n_clusters = min(3, len(non_black_pixels))

    # Perform KMeans clustering on non-black pixels
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(non_black_pixels)

    # Get the most frequent cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = unique[np.argmax(counts)]

    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(int)
    
    rgb_color = tuple(int(c) for c in dominant_color)
    hex_color = '#%02x%02x%02x' % rgb_color

    return rgb_color, hex_color

def analyze_hair_color(image_bytes: bytes) -> dict:
    # Open the image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)

    # Perform segmentation
    results = model(np_image)

    # Assuming the largest person detected is the subject
    person_masks = []
    if results[0].masks is not None:
        for i, c in enumerate(results[0].boxes.cls):
            if model.names[int(c)] == 'person':
                person_masks.append(results[0].masks.data[i].cpu().numpy())

    if not person_masks:
        return {"error": "No person detected in the image."}

    # Combine all person masks and resize to original image size
    combined_mask = np.sum(person_masks, axis=0)
    combined_mask = (combined_mask > 0).astype(np.uint8)
    combined_mask = cv2.resize(combined_mask, (np_image.shape[1], np_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Heuristic to find hair: top 25% of the person's bounding box
    person_boxes = []
    for i, c in enumerate(results[0].boxes.cls):
        if model.names[int(c)] == 'person':
            person_boxes.append(results[0].boxes.xyxy[i].cpu().numpy())
    
    if not person_boxes:
        return {"error": "No person detected in the image."}

    # Assuming the first detected person is the main subject
    box = person_boxes[0]
    x1, y1, x2, y2 = map(int, box)
    
    # Define hair region (top 25% of the bounding box height)
    hair_region_y_end = y1 + int((y2 - y1) * 0.25)
    
    # Create a mask for the hair region
    hair_mask = np.zeros_like(combined_mask)
    hair_mask[y1:hair_region_y_end, x1:x2] = 1
    
    # Intersect person mask with hair region mask
    final_hair_mask = combined_mask * hair_mask
    
    if np.sum(final_hair_mask) == 0:
        return {"error": "Could not isolate hair region."}

    # Apply mask to the original image
    hair_only_image = cv2.bitwise_and(np_image, np_image, mask=final_hair_mask)
    
    # Convert to PIL image to pass to get_dominant_color
    hair_pil_image = Image.fromarray(hair_only_image)

    # Get dominant color
    rgb_color, hex_color = get_dominant_color(hair_pil_image)

    return {
        "dominant_color_rgb": rgb_color,
        "dominant_color_hex": hex_color,
    }