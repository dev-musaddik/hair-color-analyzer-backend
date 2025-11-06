import io
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ----------------------------
# Predefined Colors
# ----------------------------
# PREDEFINED_COLORS = {
#     "#1": "#252628",
#     "Brown": "#A52A2A",
#     "Blonde": "#FAFAD2",
#     "Red": "#FF0000",
#     "Gray": "#808080",
#     "White": "#FFFFFF",
#     "Dark Brown": "#654321",
#     "Light Brown": "#C4A484",
#     "Golden Blonde": "#F0E68C",
#     "Ash Blonde": "#B2BEB5",
#     "Strawberry Blonde": "#FF9999",
#     "Auburn": "#A52A2A",
#     "Chestnut": "#954535",
#     "Honey": "#FFB347",
#     "Platinum": "#E5E4E2",
#     "Silver": "#C0C0C0",
#     "Jet Black": "#0A0A0A",
#     "Chocolate Brown": "#7B3F00",
#     "Caramel": "#FFD700",
#     "Dirty Blonde": "#C6B895",
#     "Ginger": "#B06500",
#     "Burgundy": "#800020",
#     "Mahogany": "#C04000",
#     "Copper": "#B87333",
#     "Titian": "#D2691E",
#     "Venetian Blonde": "#F7DCB4",
#     "Ash Brown": "#A9A9A9",
#     "Sandy Blonde": "#F4A460",
#     "Dark Blonde": "#9B870C"
# }
PREDEFINED_COLORS = {
    "#2BT8A": "#b69a7b",
    "#613L_18A": "#cfbfaa",
    "#6_27": "#675443",
    "#4C_27": "#65594b",
    "#4B_27": "#b19166",
    "#2BT6": "#92755a",
    "#2CT5": "#826146",
    "#13A_24": "#d0c3ae",
    "#64": "#dacec1",
    "#62": "#e3d8c0",
    "#60A": "#dfd7c5",
    "#50": "#c4c8ca",
    "#30A": "#754b3d",
    "#27": "#d7c4a2",
    "#13A": "#cdb398",
    "#10A": "#a68560",
    "#8A": "#967d62",
    "#5A": "#755539",
    "#4C": "#5c463f",
    "#4B": "#845b38",
    "#3A": "#76523b",
    "#2C": "#675141",
    "#2B": "#40332e",
    "#1C": "#2f2f2f",
    "#1B": "#252628",
    "#1": "#252628"
}



# ----------------------------
# Helper function: Hex to RGB
# ----------------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ----------------------------
# Helper function: Color difference
# ----------------------------
def color_difference(rgb1, rgb2):
    return np.sqrt(sum([(c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)]))

# ----------------------------
# Helper function: Find closest color
# ----------------------------
def find_closest_color(detected_rgb, color_map):
    closest_color_name = None
    min_difference = float('inf')
    
    for color_name, hex_value in color_map.items():
        predefined_rgb = hex_to_rgb(hex_value)
        difference = color_difference(detected_rgb, predefined_rgb)
        
        if difference < min_difference:
            min_difference = difference
            closest_color_name = color_name
            
    max_diff = np.sqrt(3 * (255 ** 2))
    similarity_percentage = 100 * (1 - min_difference / max_diff)
    
    return closest_color_name, similarity_percentage

# ----------------------------
# Helper function: Get dominant color from an image
# ----------------------------
def get_dominant_color(image: Image.Image) -> tuple:
    # Resize for performance
    image = image.resize((150, 150))
    np_image = np.array(image)
    np_image = np_image.reshape((-1, 3))

    # Use KMeans to find dominant colors
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(np_image)

    # Find the most frequent cluster
    counts = np.bincount(kmeans.labels_)
    dominant_cluster = np.argmax(counts)
    dominant_color = kmeans.cluster_centers_[dominant_cluster].astype(int)

    rgb_color = tuple(int(c) for c in dominant_color)
    hex_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
    return rgb_color, hex_color

# ----------------------------
# Main analysis function
# ----------------------------
def analyze_image_color(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get the dominant color of the entire image
        dominant_rgb, dominant_hex = get_dominant_color(image)
        
        # Find the closest predefined color
        closest_color_name, match_percentage = find_closest_color(dominant_rgb, PREDEFINED_COLORS)
        
        # Create a result message
        message = f"The dominant color is {dominant_hex}, which is closest to your predefined color '{closest_color_name}'."

        return {
            "dominant_color_rgb": dominant_rgb,
            "dominant_color_hex": dominant_hex,
            "closest_color": closest_color_name,
            "match_percentage": f"{match_percentage:.2f}%",
            "message": message
        }
    except Exception as e:
        return {"error": f"Failed to analyze image: {str(e)}"}
