import argparse
import os
import glob
from tqdm import tqdm
from PIL import Image
from indexer.embed import FashionClipEmbedder
from indexer.attributes import ZeroShotAttributeExtractor
from indexer.vector_store import FashionVectorStore

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings and zero-shot attributes, and index them in ChromaDB.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory containing raw images.")
    parser.add_argument("--db_path", type=str, default="data/vector_db", help="Directory to store ChromaDB persistent files.")
    parser.add_argument("--collection", type=str, default="fashion_items", help="ChromaDB collection name.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for embedding extraction.")
    parser.add_argument("--garment_threshold", type=float, default=0.18, help="Threshold for garment detection similarity.")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Data directory {args.data_dir} does not exist. Please run download_dataset.py first.")
        return

    # Initialize embedder, extractor, and vector store
    embedder = FashionClipEmbedder()
    extractor = ZeroShotAttributeExtractor(embedder)
    vector_store = FashionVectorStore(args.db_path, args.collection)

    # Find all image paths
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.data_dir, ext)))
    
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    total_images = len(image_paths)
    print(f"Found {total_images} images in {args.data_dir} to index.")

    if total_images == 0:
        print("No images to index. Exiting.")
        return

    # Process in batches
    for i in tqdm(range(0, total_images, args.batch_size), desc="Indexing Images"):
        batch_paths = image_paths[i:i+args.batch_size]
        
        # Load images and filter out corrupt ones
        valid_batch_paths = []
        valid_images = []
        for path in batch_paths:
            try:
                # Open image and ensure it's RGB
                img = Image.open(path)
                img.verify() # Verify image integrity
                img = Image.open(path).convert("RGB")
                valid_images.append(img)
                valid_batch_paths.append(path)
            except Exception as e:
                print(f"\nWarning: Skipping corrupt image {path}: {e}")

        if not valid_images:
            continue

        try:
            # 1. Extract image embeddings
            embeddings = embedder.embed_images(valid_images)
            
            # 2. Extract structured attributes for each image
            structured_metadatas = []
            ids = []
            for path, emb in zip(valid_batch_paths, embeddings):
                meta = extractor.extract_attributes(emb, garment_threshold=args.garment_threshold)
                structured_metadatas.append(meta)
                # Use base filename as ID
                ids.append(os.path.basename(path))

            # 3. Add to ChromaDB
            vector_store.add_items(
                ids=ids,
                embeddings=embeddings,
                structured_metadatas=structured_metadatas,
                image_paths=valid_batch_paths
            )
        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {e}")
            continue

    print(f"Indexing completed. Persistent ChromaDB saved to {args.db_path}")

if __name__ == "__main__":
    main()
