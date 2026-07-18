import chromadb
import json
import os

class FashionVectorStore:
    """
    ChromaDB vector store wrapper to index and search fashion items.
    Uses cosine similarity for vector comparison and stores structured attributes in metadata.
    """
    def __init__(self, db_path, collection_name="fashion_items"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        # Configure the collection to use cosine space for similarity matching
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_items(self, ids, embeddings, structured_metadatas, image_paths):
        """
        Add items to the vector database.
        Args:
            ids: List of unique string identifiers.
            embeddings: Numpy array of shape (N, dim) or list of list of floats.
            structured_metadatas: List of dictionaries containing scene, style, garments.
            image_paths: List of paths to the raw image files (stored as document references).
        """
        chroma_metadatas = []
        for meta, path in zip(structured_metadatas, image_paths):
            # ChromaDB only supports primitive metadata types, so serialize complex fields (nested lists/dicts) to JSON
            chroma_meta = {
                "scene": meta.get("scene", ""),
                "style": meta.get("style", ""),
                "garments_json": json.dumps(meta.get("garments", [])),
                "image_path": path
            }
            chroma_metadatas.append(chroma_meta)

        # Convert numpy embeddings if necessary
        embeddings_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=chroma_metadatas,
            documents=image_paths
        )
        print(f"Added {len(ids)} items to ChromaDB.")

    def search_semantic(self, query_embedding, limit=20):
        """
        Perform a semantic similarity search using query embedding.
        Args:
            query_embedding: Numpy array or list of floats representing query embedding.
            limit: Maximum number of candidates to retrieve.
        Returns:
            list: Formatted search results with ids, metadata, distances, and embeddings.
        """
        # Format query embedding
        emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        if len(emb) > 0 and not isinstance(emb[0], list):
            emb = [emb]

        results = self.collection.query(
            query_embeddings=emb,
            n_results=limit,
            include=["embeddings", "metadatas", "documents", "distances"]
        )

        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                # Compute similarity from cosine distance: similarity = 1 - distance
                # Chroma's cosine distance is usually in range [0, 2] where 0 is identical, 2 is opposite.
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                
                item = {
                    "id": results["ids"][0][i],
                    "embedding": results["embeddings"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i],
                    "distance": distance,
                    "similarity": similarity
                }
                
                # De-serialize the nested garments list from JSON string
                if "garments_json" in item["metadata"]:
                    try:
                        item["metadata"]["garments"] = json.loads(item["metadata"]["garments_json"])
                    except Exception:
                        item["metadata"]["garments"] = []
                formatted_results.append(item)

        return formatted_results
