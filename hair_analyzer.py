import io
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO
import cv2
from typing import List, Dict, Any, Tuple
import os

# ----------------------------
# Load the YOLO model
# ----------------------------
# This will download the model on first run if not present.
# We set verbose=False in the predict call to silence logs.
model = YOLO('yolov8n-seg.pt')

# ----------------------------
# Enhanced Predefined Hair Color Sets
# (This is the critical data you helped create, now with full metadata)
# ----------------------------
PREDEFINED_COLOR_SETS = {
    "#1": {
        "name": "Jet Black (#1)",
        "style": "Solid",
        "tone": "Cool/Neutral",
        "level": 1,
        "tags": ["ink black", "darkest", "no warmth"],
        "composition": [
            {"hex": "#252729", "percentage": 90},
            {"hex": "#313438", "percentage": 10}
        ]
    },
    "#1B": {
        "name": "Off-Black (#1B)",
        "style": "Solid",
        "tone": "Neutral Natural",
        "level": 1.5,
        "tags": ["natural black", "soft black"],
        "composition": [
            {"hex": "#141414", "percentage": 85},
            {"hex": "#2E2E2E", "percentage": 15}
        ]
    },
    "#1C": {
        "name": "Darkest Brown (#1C)",
        "style": "Solid",
        "tone": "Neutral-Warm",
        "level": 2,
        "tags": ["almost black", "deep brown", "espresso"],
        "composition": [
            {"hex": "#241C1C", "percentage": 85},
            {"hex": "#3D2E28", "percentage": 15}
        ]
    },
    "#2B": {
        "name": "Dark Chocolate Brown (#2B)",
        "style": "Solid",
        "tone": "Warm",
        "level": 3,
        "tags": ["rich brown", "chocolate", "warm dark"],
        "composition": [
            {"hex": "#36241C", "percentage": 85},
            {"hex": "#4E3629", "percentage": 15}
        ]
    },
    "#2C": {
        "name": "Medium Chestnut Brown (#2C)",
        "style": "Solid",
        "tone": "Warm Red-Undertone",
        "level": 4,
        "tags": ["chestnut", "reddish brown", "medium dark"],
        "composition": [
            {"hex": "#4A3525", "percentage": 80},
            {"hex": "#75553D", "percentage": 20}
        ]
    },
    "#3A": {
        "name": "Light Warm Brown (#3A)",
        "style": "Solid",
        "tone": "Very Warm",
        "level": 5,
        "tags": ["golden brown", "light chestnut", "warm"],
        "composition": [
            {"hex": "#6B4A36", "percentage": 80},
            {"hex": "#8F664E", "percentage": 20}
        ]
    },
    "#4B": {
        "name": "Warm Honey Brown (#4B)",
        "style": "Solid",
        "tone": "Warm Golden",
        "level": 6,
        "tags": ["honey", "golden brown", "caramel hints"],
        "composition": [
            {"hex": "#805A3B", "percentage": 85},
            {"hex": "#A67C59", "percentage": 15}
        ]
    },
    "#4C": {
        "name": "Rich Chestnut (#4C)",
        "style": "Solid",
        "tone": "Neutral-Warm",
        "level": 5,
        "tags": ["nutty", "medium brown", "neutral"],
        "composition": [
            {"hex": "#5E4033", "percentage": 85},
            {"hex": "#825E4C", "percentage": 15}
        ]
    },
    "#5A": {
        "name": "Medium Ash Brown (#5A)",
        "style": "Solid",
        "tone": "Cool Ash",
        "level": 5,
        "tags": ["cool brown", "no red", "matte finish"],
        "composition": [
            {"hex": "#6B5B4D", "percentage": 85},
            {"hex": "#8C7A6B", "percentage": 15}
        ]
    },
    "#8A": {
        "name": "Light Ash Brown (#8A)",
        "style": "Solid",
        "tone": "Cool Ash",
        "level": 8,
        "tags": ["dark blonde", "cool blonde", "mushroom"],
        "composition": [
            {"hex": "#9C8A74", "percentage": 85},
            {"hex": "#BDB09E", "percentage": 15}
        ]
    },
    "#10A": {
        "name": "Medium Golden Blonde (#10A)",
        "style": "Solid",
        "tone": "Neutral-Warm",
        "level": 10,
        "tags": ["sandy blonde", "neutral blonde", "soft gold"],
        "composition": [
            {"hex": "#BFA37C", "percentage": 80},
            {"hex": "#DBCAAC", "percentage": 20}
        ]
    },
    "#13A": {
        "name": "Pale Ash Blonde (#13A)",
        "style": "Solid",
        "tone": "Cool Beige",
        "level": 10.5,
        "tags": ["champagne", "cool blonde", "beige"],
        "composition": [
            {"hex": "#DCD0BA", "percentage": 85},
            {"hex": "#EAE2D4", "percentage": 15}
        ]
    },
    "#27": {
        "name": "Strawberry Blonde (#27)",
        "style": "Solid",
        "tone": "Warm Copper",
        "level": 8.5,
        "tags": ["ginger blonde", "warm gold", "copper"],
        "composition": [
            {"hex": "#C48E5F", "percentage": 80},
            {"hex": "#E3B98F", "percentage": 20}
        ]
    },
    "#30A": {
        "name": "Deep Auburn Copper (#30A)",
        "style": "Solid",
        "tone": "Very Warm Red",
        "level": 6,
        "tags": ["red hair", "auburn", "copper"],
        "composition": [
            {"hex": "#8A4E3B", "percentage": 85},
            {"hex": "#B06D56", "percentage": 15}
        ]
    },
    "#50": {
        "name": "Pale Icy Silver (#50)",
        "style": "Solid",
        "tone": "Very Cool / Metallic",
        "level": 11,
        "tags": ["silver", "grey", "icy white"],
        "composition": [
            {"hex": "#D9D9D9", "percentage": 85},
            {"hex": "#EDEDED", "percentage": 15}
        ]
    },
    "#60A": {
        "name": "Pure White Platinum (#60A)",
        "style": "Solid",
        "tone": "Neutral White",
        "level": 12,
        "tags": ["bleach blonde", "snow white", "brightest"],
        "composition": [
            {"hex": "#F0EEE6", "percentage": 90},
            {"hex": "#FAF9F2", "percentage": 10}
        ]
    },
    "#62": {
        "name": "Icy Platinum (#62)",
        "style": "Solid",
        "tone": "Cool Platinum",
        "level": 11,
        "tags": ["platinum blonde", "cool blonde", "nordic"],
        "composition": [
            {"hex": "#E8DEC7", "percentage": 85},
            {"hex": "#F5EED9", "percentage": 15}
        ]
    },
    "#64": {
        "name": "Creamy Pearl (#64)",
        "style": "Solid",
        "tone": "Neutral Warm Pearl",
        "level": 11,
        "tags": ["creamy blonde", "pearl", "soft white"],
        "composition": [
            {"hex": "#E8DAB6", "percentage": 85},
            {"hex": "#F2EBD4", "percentage": 15}
        ]
    },
    "#13A/24": {
        "name": "Champagne Blend (#13A/24)",
        "style": "Blended Highlights",
        "tone": "Neutral Beige",
        "level": 9.5,
        "tags": ["creamy blend", "soft highlights", "natural blonde mix"],
        "composition": [
            {"hex": "#D9C59E", "percentage": 70},
            {"hex": "#E8D9B5", "percentage": 30}
        ]
    },
    "#2CT5": {
        "name": "Mocha Root Melt (#2CT5)",
        "style": "Rooted/Ombre",
        "tone": "Cool Root to Warm End",
        "level": 3.5,
        "tags": ["ombre", "dark roots", "melted"],
        "composition": [
            {"hex": "#7D5F4A", "percentage": 70},
            {"hex": "#33261E", "percentage": 30}
        ]
    },
    "#2BT6": {
        "name": "Dark Chocolate Dip (#2BT6)",
        "style": "Rooted/Ombre",
        "tone": "Neutral Root to Warm End",
        "level": 4,
        "tags": ["high contrast ombre", "rooted dark", "chestnut ends"],
        "composition": [
            {"hex": "#6B4E3B", "percentage": 70},
            {"hex": "#291E1A", "percentage": 30}
        ]
    },
    "#4B/27": {
        "name": "Golden Brown Swirl (#4B/27)",
        "style": "Piano Highlights",
        "tone": "Warm Contrast",
        "level": 6,
        "tags": ["chunky highlights", "mixed brown blonde", "streaky"],
        "composition": [
            {"hex": "#6B4731", "percentage": 65},
            {"hex": "#C48E5F", "percentage": 35}
        ]
    },
    "#4C/27": {
        "name": "Cool Espresso Streak (#4C/27)",
        "style": "Piano Highlights",
        "tone": "High Contrast Neutral/Warm",
        "level": 6,
        "tags": ["defined highlights", "dark base light streaks", "piano"],
        "composition": [
            {"hex": "#4A362E", "percentage": 65},
            {"hex": "#C79668", "percentage": 35}
        ]
    },
    "#6/27": {
        "name": "Caramel Toast (#6/27)",
        "style": "Soft Blend",
        "tone": "Very Warm",
        "level": 7,
        "tags": ["sunkissed", "caramel mix", "warm blend"],
        "composition": [
            {"hex": "#75543E", "percentage": 70},
            {"hex": "#BD8E63", "percentage": 30}
        ]
    },
    "#613L/18A": {
        "name": "Vanilla Ash Swirl (#613L/18A)",
        "style": "Piano Highlights",
        "tone": "Cool Contrast",
        "level": 9,
        "tags": ["mixed blonde", "ash and platinum", "dimensional blonde"],
        "composition": [
            {"hex": "#EBE6D1", "percentage": 70},
            {"hex": "#A69685", "percentage": 30}
        ]
    },
    "#2BT8A": {
        "name": "Ashy Rooted Bronde (#2BT8A)",
        "style": "Rooted/Ombre",
        "tone": "Cool Ash Body",
        "level": 5,
        "tags": ["rooted blonde", "ashy ends", "dark root blonde"],
        "composition": [
            {"hex": "#9E8C75", "percentage": 70},
            {"hex": "#2E2421", "percentage": 30}
        ]
    }
}


# ----------------------------
# Helper Functions
# ----------------------------
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
    """Converts an (R, G, B) tuple to a NumPy array in CIELAB color space."""
    rgb_pixel = np.uint8([[list(rgb)]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)[0][0]
    return lab_pixel.astype(float)

def color_difference_lab(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """Calculates the perceptual color difference (Delta E) between two RGB colors."""
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    return np.linalg.norm(lab1 - lab2)

def estimate_image_properties(dominant_colors: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    """
    Estimates the overall Tone, Level, and Style from the dominant colors.
    """
    if not dominant_colors:
        return "Neutral", 5.0, "Solid"

    # --- Tone & Level Estimation (Weighted Average) ---
    total_r, total_g, total_b, total_weight = 0.0, 0.0, 0.0, 0.0
    for color in dominant_colors:
        r, g, b = color['rgb']
        w = color['percentage']
        total_r += r * w
        total_g += g * w
        total_b += b * w
        total_weight += w

    if total_weight == 0:
        return "Neutral", 5.0, "Solid"

    avg_r = total_r / total_weight
    avg_g = total_g / total_weight
    avg_b = total_b / total_weight

    # Level (Lightness) based on luminance
    luminance = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
    level = max(1.0, min(12.0, (luminance / 25.5) + 1.5)) # Scaled 1-12

    # Tone (Warm/Cool)
    if (avg_r - avg_b > 15) and (avg_g - avg_b > 5):
        tone = "Warm"
    elif (avg_b - avg_r > 10):
        tone = "Cool"
    else:
        tone = "Neutral"

    # --- Style Estimation (Solid vs. Mixed) ---
    if len(dominant_colors) < 2:
        style = "Solid"
    else:
        # Check perceptual difference between top 2 colors
        color_diff = color_difference_lab(dominant_colors[0]['rgb'], dominant_colors[1]['rgb'])
        # If colors are very different and neither is a tiny fraction, it's mixed
        if color_diff > 30 and dominant_colors[1]['percentage'] > 15:
            style = "Mixed"
        else:
            style = "Solid"
            
    return tone, level, style


# ----------------------------
# Core Image Analysis
# ----------------------------
def segment_hair(image: Image.Image) -> np.ndarray:
    """Segments hair from an image using YOLOv8-seg and returns a mask."""
    results = model(image, verbose=False) # verbose=False silences YOLO logs
    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        return None
    
    masks = results[0].masks.data.cpu().numpy()
    # Combine all detected masks into one
    hair_mask = np.sum(masks, axis=0)
    hair_mask = np.clip(hair_mask, 0, 1).astype('float32')

    if hair_mask.ndim != 2 or hair_mask.shape[0] == 0 or hair_mask.shape[1] == 0:
        return None
        
    # Resize mask to match original image dimensions
    return cv2.resize(hair_mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

def get_dominant_colors(image: Image.Image, hair_mask: np.ndarray, n_colors=3) -> List[Dict[str, Any]]:
    """Extracts the top N dominant colors from the masked hair region using k-means."""
    np_image = np.array(image)
    mask_bool = hair_mask > 0.5
    hair_pixels = np_image[mask_bool]

    # Need enough pixels for clustering
    if len(hair_pixels) < n_colors * 10:
        if len(hair_pixels) > 0:
             mean_color = np.mean(hair_pixels, axis=0).astype(int)
             return [{
                 "hex": f"#{mean_color[0]:02x}{mean_color[1]:02x}{mean_color[2]:02x}",
                 "rgb": tuple(mean_color),
                 "percentage": 100.0
             }]
        return []

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(hair_pixels)

    counts = np.bincount(kmeans.labels_)
    total_pixels = len(hair_pixels)
    
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors_rgb = kmeans.cluster_centers_.astype(int)[sorted_indices]
    percentages = (counts[sorted_indices] / total_pixels) * 100

    return [
        {
            "hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
            "rgb": tuple(rgb),
            "percentage": round(p, 2)
        }
        for rgb, p in zip(dominant_colors_rgb, percentages)
    ]

def find_best_match(detected_colors: List[Dict[str, Any]], predefined_sets: Dict[str, Dict[str, Any]]) -> tuple:
    """
    Finds the best matching predefined set using a weighted scoring algorithm
    based on color, composition, tone, level, and style.
    """
    detected_tone, detected_level, detected_style = estimate_image_properties(detected_colors)
    
    all_matches = []

    for set_id, p_set in predefined_sets.items():
        total_color_score = 0
        total_percentage_mapped = 0

        p_composition = [
            {**c, "rgb": hex_to_rgb(c["hex"])} 
            for c in p_set["composition"]
        ]

        # 1. Color & Composition Score (Lower is better)
        for d_color in detected_colors:
            d_rgb = d_color["rgb"]
            d_percentage = d_color["percentage"]

            # Find the perceptually closest color in the predefined set
            best_p_color = min(
                p_composition, 
                key=lambda p: color_difference_lab(d_rgb, p["rgb"])
            )
            
            min_color_diff = color_difference_lab(d_rgb, best_p_color["rgb"])
            
            # Find closest percentage match *for that color* (less important)
            percentage_diff = abs(d_percentage - best_p_color["percentage"])
            
            # Weighted score: 80% color accuracy, 20% percentage accuracy
            component_score = (min_color_diff * 0.8) + (percentage_diff * 0.2)
            
            # Weight this component's score by its presence in the image
            total_color_score += component_score * (d_percentage / 100.0)
            total_percentage_mapped += d_percentage

        base_score = total_color_score / (total_percentage_mapped / 100.0) if total_percentage_mapped > 0 else float('inf')

        # 2. Penalties (Higher is worse)
        # Tone Penalty: Heavy penalty for mismatch (e.g., Warm vs. Cool)
        tone_penalty = 0
        p_tone = p_set.get("tone", "Neutral")
        if "Warm" in detected_tone and "Cool" in p_tone: tone_penalty = 40
        elif "Cool" in detected_tone and "Warm" in p_tone: tone_penalty = 40
        elif "Neutral" not in detected_tone and "Neutral" in p_tone: tone_penalty = 15
        
        # Level Penalty: Penalize based on lightness difference
        p_level = p_set.get("level", 5)
        level_penalty = abs(detected_level - p_level) * 5 # 5 points per level
        
        # Style Penalty: Penalize mismatch (e.g., Solid vs. Rooted)
        style_penalty = 0
        p_style = p_set.get("style", "Solid")
        if detected_style == "Solid" and p_style != "Solid":
            style_penalty = 30 # Solid image should not match a mixed set
        elif detected_style == "Mixed" and p_style == "Solid":
            style_penalty = 20 # Mixed image *could* match a solid, but it's less likely

        # Final Score: Base (Color) + Penalties
        final_score = base_score + tone_penalty + level_penalty + style_penalty

        all_matches.append({
            "set_id": set_id,
            "set_name": p_set["name"],
            "score": final_score,
            "style": p_set.get("style"),
            "tone": p_set.get("tone"),
            "level": p_set.get("level"),
            "tags": p_set.get("tags")
        })

    if not all_matches:
        return None, []

    # Sort by lowest score (best match)
    sorted_matches = sorted(all_matches, key=lambda x: x['score'])
    
    # Normalize scores to a 0-100% similarity for UI
    # This is a simple heuristic: 100 is a "bad" score.
    # We use the best score as a baseline for relative comparison.
    best_score = sorted_matches[0]['score']
    
    detailed_results = []
    for match in sorted_matches:
        # Similarity is relative to the best match. 1.0 = 100%
        # A score twice as bad as the best is 50% as good.
        similarity_raw = best_score / max(match['score'], best_score) # Avoid division by zero
        similarity = max(0.0, min(100.0, similarity_raw * 100))

        # This simple % logic is often better for users:
        # Map score from [best_score, 150] -> [100, 0]
        # (150 is an arbitrary 'worst' score)
        similarity = 100 * (1 - (match['score'] - best_score) / (150 - best_score))
        similarity = max(0.0, min(100.0, similarity))

        match["similarity_percentage"] = f"{similarity:.1f}%"
        detailed_results.append(match)

    # Re-sort by similarity (highest first)
    detailed_results = sorted(detailed_results, key=lambda x: float(x['similarity_percentage'][:-1]), reverse=True)

    return detailed_results[0], detailed_results

# ----------------------------
# Main Analysis Orchestrator
# ----------------------------
def analyze_image_color(image_bytes: bytes) -> Dict[str, Any]:
    """
    Main function to run the full analysis pipeline on a single image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 1. Segment hair
        hair_mask = segment_hair(image)
        if hair_mask is None or np.sum(hair_mask < 0.1) > (hair_mask.shape[0] * hair_mask.shape[1] * 0.99):
             # If no mask or mask is < 1% of image
            return {"error": "No hair detected. Please try a clearer photo where hair is prominent."}

        # 2. Extract dominant colors
        dominant_colors = get_dominant_colors(image, hair_mask, n_colors=3)
        if not dominant_colors:
            return {"error": "Could not extract dominant colors from the detected hair."}

        # 3. Find best match
        best_match, all_matches = find_best_match(dominant_colors, PREDEFINED_COLOR_SETS)
        if best_match is None:
             return {"error": "Could not find a suitable match in our color library."}

        # Clean up RGB tuples for JSON serialization
        for c in dominant_colors:
            del c["rgb"]

        return {
            "dominant_hair_colors": dominant_colors,
            "best_match": best_match,
            "all_set_matches": all_matches[:5] # Return top 5 matches
        }

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        # Return a generic error to the user
        return {"error": "An unexpected server error occurred."}