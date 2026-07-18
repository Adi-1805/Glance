import re
from indexer.vocab import GARMENTS, COLORS, SCENES, STYLES

class QueryParser:
    """
    Rule-based query parser to extract structured fashion attributes 
    (garment-color pairs, scenes, styles) from natural language search queries.
    """
    def __init__(self):
        self.garments = GARMENTS
        self.colors = COLORS
        self.scenes = SCENES
        self.styles = STYLES

    def parse(self, query_text):
        """
        Parse query text into structured slots.
        Args:
            query_text: Natural language string.
        Returns:
            dict: Structured search criteria containing:
                  - 'garments': List of {'garment': str, 'color': str or None}
                  - 'scene': str or None
                  - 'style': str or None
        """
        text = query_text.lower().strip()
        # Strip common punctuation
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()

        # 1. Parse Garment-Color Pairs
        parsed_garments = []
        
        # Scan for garments, and check adjacent window for colors
        for idx, token in enumerate(tokens):
            if token in self.garments:
                garment = token
                bound_color = None
                
                # Check preceding tokens then succeeding tokens (distance up to 3)
                window_indices = [idx-1, idx-2, idx-3, idx+1, idx+2, idx+3]
                for w_idx in window_indices:
                    if 0 <= w_idx < len(tokens):
                        candidate_color = tokens[w_idx]
                        if candidate_color in self.colors:
                            bound_color = candidate_color
                            break # Found nearest color, stop searching
                
                parsed_garments.append({
                    "garment": garment,
                    "color": bound_color
                })

        # 2. Parse Scene/Context terms
        parsed_scene = None
        scene_keywords = {
            "office": "office",
            "street": "urban street",
            "urban": "urban street",
            "city": "urban street",
            "park": "park",
            "bench": "outdoor bench",
            "home": "home interior",
            "interior": "home interior",
            "house": "home interior",
            "cafe": "cafe",
            "indoor": "indoors",
            "indoors": "indoors",
            "outdoor": "outdoors",
            "outdoors": "outdoors",
            "outside": "outdoors"
        }
        
        matched_scenes = []
        for kw, scene_val in scene_keywords.items():
            if kw in tokens:
                matched_scenes.append((scene_val, tokens.index(kw)))
        
        if matched_scenes:
            # Prioritize specific scenes like outdoor bench over park if both are present
            has_bench = any(s[0] == "outdoor bench" for s in matched_scenes)
            if has_bench:
                parsed_scene = "outdoor bench"
            else:
                # Retrieve the one that appeared first in the query
                matched_scenes.sort(key=lambda x: x[1])
                parsed_scene = matched_scenes[0][0]

        # 3. Parse Style terms
        parsed_style = None
        style_keywords = {
            "formal": "formal",
            "casual": "casual",
            "outerwear": "outerwear",
            "athletic": "athletic",
            "sporty": "athletic",
            "professional": "professional",
            "business": "business",
            "work": "professional",
            "weekend": "weekend attire"
        }
        
        matched_styles = []
        for kw, style_val in style_keywords.items():
            if kw in tokens:
                matched_styles.append((style_val, tokens.index(kw)))
                
        if matched_styles:
            matched_styles.sort(key=lambda x: x[1])
            parsed_style = matched_styles[0][0]

        return {
            "garments": parsed_garments,
            "scene": parsed_scene,
            "style": parsed_style
        }
