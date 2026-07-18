from retriever.query_parser import QueryParser
import numpy as np

class HybridRetriever:
    """
    Hybrid retriever that combines FashionCLIP semantic search with structured attribute match re-ranking.
    """
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.parser = QueryParser()

    def search(self, query_text, top_k=5, candidate_limit=20, alpha=0.4):
        """
        Execute a hybrid search using semantic similarity and metadata attribute matching.
        Args:
            query_text: Natural language query string.
            top_k: Number of final re-ranked items to return.
            candidate_limit: Number of items retrieved during the broad semantic retrieval phase.
            alpha: Weight for structured attribute matching. Score = (1 - alpha) * CLIP_sim + alpha * Attr_score.
        Returns:
            list: Re-ranked candidates with enriched scores and structured metadata.
        """
        # 1. Parse structured slots from query
        parsed_query = self.parser.parse(query_text)

        # 2. Embed the raw text query
        query_emb = self.embedder.embed_text([query_text])[0]

        # 3. Retrieve candidates using semantic embedding similarity
        candidates = self.vector_store.search_semantic(query_emb, limit=candidate_limit)

        # 4. Compute attribute match scores and re-rank
        re_ranked = []
        for candidate in candidates:
            cand_meta = candidate["metadata"]
            cand_scene = cand_meta.get("scene", "")
            cand_style = cand_meta.get("style", "")
            cand_garments = cand_meta.get("garments", [])

            scores_to_average = []
            weights = []

            # A. Garment-color matching
            q_garments = parsed_query["garments"]
            if q_garments:
                g_scores = []
                for qg in q_garments:
                    g_name = qg["garment"]
                    g_color = qg["color"]

                    # Find this garment in candidate
                    found = False
                    for cg in cand_garments:
                        if cg["garment"] == g_name:
                            found = True
                            if g_color is None:
                                g_scores.append(0.8)  # Garment type matches, color not specified
                            elif cg["color"] == g_color:
                                g_scores.append(1.0)  # Full garment and color match
                            else:
                                g_scores.append(0.3)  # Garment matches, but color is different (partial)
                            break
                    if not found:
                        g_scores.append(0.0)  # Garment not present
                
                garment_score = sum(g_scores) / len(g_scores)
                scores_to_average.append(garment_score)
                weights.append(0.5)

            # B. Scene matching
            q_scene = parsed_query["scene"]
            if q_scene:
                # Match exact or keywords (e.g. bench fits outdoor bench)
                scene_score = 1.0 if cand_scene == q_scene else 0.0
                scores_to_average.append(scene_score)
                weights.append(0.25)

            # C. Style matching
            q_style = parsed_query["style"]
            if q_style:
                # Handle synonyms (e.g., professional & business are related style tags)
                style_matches = (
                    cand_style == q_style or
                    (q_style in ["professional", "business"] and cand_style in ["professional", "business"])
                )
                style_score = 1.0 if style_matches else 0.0
                scores_to_average.append(style_score)
                weights.append(0.25)

            # Compute weighted attribute match score
            if scores_to_average:
                total_w = sum(weights)
                norm_weights = [w / total_w for w in weights]
                attribute_score = sum(s * w for s, w in zip(scores_to_average, norm_weights))
            else:
                # Fallback to 1.0 (no penalty) if no structured attributes are parsed in query
                attribute_score = 1.0

            # Hybrid score: (1 - alpha) * CLIP + alpha * Attribute
            clip_similarity = candidate["similarity"]
            hybrid_score = (1.0 - alpha) * clip_similarity + alpha * attribute_score

            candidate_enriched = {
                **candidate,
                "parsed_query": parsed_query,
                "attribute_score": attribute_score,
                "hybrid_score": hybrid_score
            }
            re_ranked.append(candidate_enriched)

        # Sort candidates by hybrid score descending
        re_ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return re_ranked[:top_k]
