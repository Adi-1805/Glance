import os
import json
import argparse
from indexer.embed import FashionClipEmbedder
from indexer.vector_store import FashionVectorStore
from retriever.search import HybridRetriever

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting."
]

def main():
    parser = argparse.ArgumentParser(description="Evaluate the hybrid retrieval system against official queries.")
    parser.add_argument("--db_path", type=str, default="data/vector_db", help="Directory where ChromaDB is persisted.")
    parser.add_argument("--collection", type=str, default="fashion_items", help="ChromaDB collection name.")
    parser.add_argument("--output_report", type=str, default="evaluation_report.md", help="Path to output markdown report.")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"Vector DB directory {args.db_path} does not exist. Please index the dataset first.")
        return

    # Load embedder, vector store, and retriever
    embedder = FashionClipEmbedder()
    vector_store = FashionVectorStore(args.db_path, args.collection)
    retriever = HybridRetriever(embedder, vector_store)

    report_content = []
    report_content.append("# Evaluation Report: Multimodal Fashion & Context Retrieval System\n")
    report_content.append("This report evaluates the performance of the **Hybrid Retrieval Pipeline** combining FashionCLIP and Structured Attribute Match Re-ranking on the 5 official assignment queries.\n")
    report_content.append("## Hybrid Architecture Highlights\n")
    report_content.append("- **Broad Recall (FashionCLIP)**: Retrieves semantic matches based on raw text.")
    report_content.append("- **Compositional Precision (Zero-Shot Attribute Match)**: Resolves color binding (\"red tie and white shirt\") and environment context using structured metadata filtering and scoring.")
    report_content.append("- **Re-ranking Weights**: 60% FashionCLIP similarity + 40% structured attribute match score.\n")

    print("\nRunning Evaluation Queries...")
    for q_idx, query in enumerate(EVAL_QUERIES):
        print(f"Query {q_idx+1}: '{query}'")
        
        # Search
        results = retriever.search(query, top_k=5, alpha=0.4)
        
        if not results:
            print("  No results found!")
            continue

        parsed = results[0]["parsed_query"]
        
        report_content.append(f"## Query {q_idx+1}: `{query}`")
        report_content.append(f"**Parsed Query Slots:**")
        report_content.append(f"- **Garments**: `{parsed['garments']}`")
        report_content.append(f"- **Scene**: `{parsed['scene']}`")
        report_content.append(f"- **Style**: `{parsed['style']}`\n")
        
        report_content.append("| Rank | Image ID | Hybrid Score | CLIP Similarity | Attribute Overlap | Scene (Indexed) | Style (Indexed) | Garments (Indexed) |")
        report_content.append("|---|---|---|---|---|---|---|---|")
        
        for rank, res in enumerate(results):
            meta = res["metadata"]
            img_id = res["id"]
            hybrid_score = res["hybrid_score"]
            clip_sim = res["similarity"]
            attr_score = res["attribute_score"]
            scene = meta.get("scene", "N/A")
            style = meta.get("style", "N/A")
            garments_list = meta.get("garments", [])
            garments_formatted = ", ".join([f"{g.get('color', '')} {g.get('garment', '')}".strip() for g in garments_list])
            
            report_content.append(f"| {rank+1} | {img_id} | {hybrid_score:.4f} | {clip_sim:.4f} | {attr_score:.4f} | {scene} | {style} | {garments_formatted} |")
        report_content.append("\n")

    # Save report
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    print(f"\nEvaluation completed. Report written to {args.output_report}")

if __name__ == "__main__":
    main()
