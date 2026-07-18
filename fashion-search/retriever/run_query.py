import argparse
from indexer.embed import FashionClipEmbedder
from indexer.vector_store import FashionVectorStore
from retriever.search import HybridRetriever

def main():
    parser = argparse.ArgumentParser(description="Query the Hybrid Fashion Retrieval System.")
    parser.add_argument("--query", type=str, required=True, help="Natural language query describing garments/style/scene.")
    parser.add_argument("--db_path", type=str, default="data/vector_db", help="Directory where ChromaDB is persisted.")
    parser.add_argument("--collection", type=str, default="fashion_items", help="ChromaDB collection name.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of search results to return.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight for structured attribute matching [0.0 - 1.0].")
    args = parser.parse_args()

    # Load embedder, vector store, and retriever
    embedder = FashionClipEmbedder()
    vector_store = FashionVectorStore(args.db_path, args.collection)
    retriever = HybridRetriever(embedder, vector_store)

    # Execute search
    results = retriever.search(args.query, top_k=args.top_k, alpha=args.alpha)

    print("\n" + "="*80)
    print(f"SEARCH RESULTS FOR QUERY: \"{args.query}\"")
    print("="*80)
    
    if not results:
        print("No matches found.")
        return

    # Print parsed query structure from first result
    parsed = results[0]["parsed_query"]
    print(f"Parsed Query Slots:")
    print(f"  - Garments: {parsed['garments']}")
    print(f"  - Scene: {parsed['scene']}")
    print(f"  - Style: {parsed['style']}")
    print("="*80)

    for i, res in enumerate(results):
        meta = res["metadata"]
        print(f"\n{i+1}. Image: {res['id']}")
        print(f"   Path: {res['document']}")
        print(f"   Scores:")
        print(f"     - Hybrid Score:      {res['hybrid_score']:.4f}")
        print(f"     - CLIP Similarity:   {res['similarity']:.4f}")
        print(f"     - Attribute Overlap: {res['attribute_score']:.4f}")
        print(f"   Indexed Attributes:")
        print(f"     - Garments: {meta.get('garments', [])}")
        print(f"     - Scene:    {meta.get('scene', 'N/A')}")
        print(f"     - Style:    {meta.get('style', 'N/A')}")
        print("-"*80)

if __name__ == "__main__":
    main()
