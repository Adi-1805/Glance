"""
Vocabulary definition for structured fashion attributes.
This is shared between indexing (zero-shot attribute extraction) and retrieval (query parsing).
"""

GARMENTS = [
    "shirt", "blazer", "hoodie", "raincoat", "tie", "pants", "dress", 
    "skirt", "jacket", "coat", "sweater", "shorts", "jeans", "t-shirt", "suit"
]

COLORS = [
    "red", "blue", "yellow", "white", "black", "green", "brown", "grey", 
    "pink", "purple", "orange", "beige", "navy"
]

SCENES = [
    "office", "urban street", "park", "home interior", "outdoor bench", "cafe", 
    "indoors", "outdoors"
]

STYLES = [
    "formal", "casual", "outerwear", "athletic", "professional", "business", "weekend attire"
]

# Prompt templates for zero-shot classification
GARMENT_PROMPT_TEMPLATE = "a photo of a person wearing a {garment}"
GARMENT_COLOR_PROMPT_TEMPLATE = "a photo of a person wearing a {color} {garment}"
SCENE_PROMPT_TEMPLATE = "a photo of a person in a {scene}"
STYLE_PROMPT_TEMPLATE = "a photo of a person wearing {style} attire"
