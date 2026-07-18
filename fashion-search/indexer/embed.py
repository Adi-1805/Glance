import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class FashionClipEmbedder:
    """
    Wrapper around the FashionCLIP model for generating normalized image and text embeddings.
    """
    def __init__(self, model_name="patrickjohncyh/fashion-clip"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading FashionCLIP on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("FashionCLIP loaded successfully.")

    def embed_images(self, images):
        """
        Extract normalized image embeddings.
        Args:
            images: List of PIL.Image.Image objects or absolute paths to image files.
        Returns:
            numpy.ndarray: Normalized image embeddings of shape (batch_size, embedding_dim).
        """
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))
        
        if not pil_images:
            return np.empty((0, self.model.config.projection_dim))

        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def embed_text(self, texts):
        """
        Extract normalized text embeddings.
        Args:
            texts: List of strings.
        Returns:
            numpy.ndarray: Normalized text embeddings of shape (batch_size, embedding_dim).
        """
        if not texts:
            return np.empty((0, self.model.config.projection_dim))

        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()
