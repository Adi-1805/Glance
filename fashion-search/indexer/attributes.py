import numpy as np
from indexer.vocab import (
    GARMENTS, COLORS, SCENES, STYLES, 
    GARMENT_PROMPT_TEMPLATE, GARMENT_COLOR_PROMPT_TEMPLATE, 
    SCENE_PROMPT_TEMPLATE, STYLE_PROMPT_TEMPLATE
)

class ZeroShotAttributeExtractor:
    """
    Extracts structured fashion attributes (garments, colors, scenes, styles) 
    from image embeddings using zero-shot classification with pre-computed text prompts.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        print("Pre-computing text prompt embeddings for zero-shot classification...")
        
        # 1. Scene embeddings
        self.scene_prompts = [SCENE_PROMPT_TEMPLATE.format(scene=s) for s in SCENES]
        self.scene_features = self.embedder.embed_text(self.scene_prompts)

        # 2. Style embeddings
        self.style_prompts = [STYLE_PROMPT_TEMPLATE.format(style=s) for s in STYLES]
        self.style_features = self.embedder.embed_text(self.style_prompts)

        # 3. Garment detection embeddings
        self.garment_prompts = [GARMENT_PROMPT_TEMPLATE.format(garment=g) for g in GARMENTS]
        self.garment_features = self.embedder.embed_text(self.garment_prompts)

        # 4. Joint garment-color embeddings
        # Structured as a matrix of shape (len(GARMENTS), len(COLORS), embedding_dim)
        joint_prompts = []
        for g in GARMENTS:
            for c in COLORS:
                joint_prompts.append(GARMENT_COLOR_PROMPT_TEMPLATE.format(color=c, garment=g))
        
        joint_features_flat = self.embedder.embed_text(joint_prompts)
        dim = joint_features_flat.shape[1]
        self.joint_features = joint_features_flat.reshape(len(GARMENTS), len(COLORS), dim)
        print("Pre-computation completed successfully.")

    def extract_attributes(self, image_embedding, garment_threshold=0.18):
        """
        Extract structured attributes for a single image embedding.
        Args:
            image_embedding: normalized image embedding of shape (embedding_dim,) or (1, embedding_dim)
            garment_threshold: minimum similarity score to consider a garment present.
        Returns:
            dict: Structured attributes containing scene, style, and garment-color pairs.
        """
        # Ensure embedding is 1D
        if image_embedding.ndim == 2:
            image_embedding = image_embedding[0]

        # Normalize the embedding if not already normalized
        norm = np.linalg.norm(image_embedding)
        if norm > 0:
            image_embedding = image_embedding / norm

        # 1. Classify Scene
        scene_sims = self.scene_features @ image_embedding
        top_scene_idx = np.argmax(scene_sims)
        scene = SCENES[top_scene_idx]

        # 2. Classify Style
        style_sims = self.style_features @ image_embedding
        top_style_idx = np.argmax(style_sims)
        style = STYLES[top_style_idx]

        # 3. Detect Garments & Color pairing
        garment_sims = self.garment_features @ image_embedding
        
        # Dynamic thresholding: keep garments close to the maximum similarity
        max_sim = np.max(garment_sims)
        margin = 0.10  # Keep garments within 0.10 of the best match
        floor = 0.22   # Absolute minimum similarity floor
        
        detected_garments = []
        for idx, sim in enumerate(garment_sims):
            if sim >= max_sim - margin and sim >= floor:
                detected_garments.append((idx, sim))
        
        # Fallback: if no garments detected above criteria, take the single top garment
        if not detected_garments:
            top_garment_idx = np.argmax(garment_sims)
            detected_garments.append((top_garment_idx, garment_sims[top_garment_idx]))
            
        # Sort by similarity descending and keep at most top 5 to prevent over-tagging
        detected_garments.sort(key=lambda x: x[1], reverse=True)
        detected_garments = detected_garments[:5]

        garment_color_pairs = []
        for garment_idx, _ in detected_garments:
            g_name = GARMENTS[garment_idx]
            
            # Find the best color match for this specific garment
            color_features_for_g = self.joint_features[garment_idx]  # shape: (len(COLORS), dim)
            color_sims = color_features_for_g @ image_embedding
            best_color_idx = np.argmax(color_sims)
            c_name = COLORS[best_color_idx]
            
            garment_color_pairs.append({
                "garment": g_name,
                "color": c_name
            })

        return {
            "scene": scene,
            "style": style,
            "garments": garment_color_pairs
        }
