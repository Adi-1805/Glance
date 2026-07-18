# Multimodal Fashion & Context Retrieval System: Technical Submission Report

**Author:** Machine Learning Intern Candidate  
**Repository URL:** `[INSERT REPO URL]`  

---

## 1. Approaches Considered & Tradeoffs

To retrieve fashion items from a multimodal database using natural language queries, we analyzed three primary approaches:

| Approach | Architecture Description | Advantages | Disadvantages / Tradeoffs |
|---|---|---|---|
| **A. Vanilla CLIP Zero-Shot** | OpenAI CLIP (e.g. `ViT-B/32`) used directly to encode queries and images, retrieving items via cosine similarity. | - Out-of-the-box zero-shot capability.<br>- Fast indexing and query execution.<br>- No training data or labels required. | - **Fails on compositionality**: Cannot bind color to specific garments (e.g., "red shirt with blue pants" is identical to "blue shirt with red pants" in CLIP's representation).<br>- Poor performance on domain-specific fashion attributes. |
| **B. End-to-End Fine-Tuning** | Fine-tuning CLIP or FashionCLIP's projection head or entire weights on a large dataset of fashion image-text pairs (e.g. contrastive loss). | - Learns strong task-specific alignments.<br>- Highly accurate on fashion terms if training distribution covers them. | - **Requires massive labeled datasets** (often millions of image-text pairs).<br>- High compute budget and training time.<br>- Vulnerable to overfitting and poor zero-shot generalization. |
| **C. Hybrid Structured Re-ranking (Chosen)** | Domain-specific base model (**FashionCLIP**) for broad semantic recall, combined with **Zero-Shot Hierarchical Attribute Extraction** (garments, colors, scenes, styles) and a rule-based query parser for compositional re-ranking. | - **Solves compositionality**: Explicitly binds colors to garments in metadata.<br>- Retains zero-shot capability.<br>- High context awareness without training from scratch.<br>- Fast, deterministic query parsing. | - Slightly higher latency due to a two-stage pipeline.<br>- Indexing requires running both image encoding and zero-shot attribute classification. |

---

## 2. Chosen Approach: Architecture & Technical Write-up

Our system implements **Approach C: Hybrid Structured Re-ranking**. 

```
                                  [ User Query ]
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
              [ Query Parser ]                    [ FashionCLIP Text ]
                      │                                   │
             (Structured Slots)                   (Query Embedding)
                      │                                   │
                      │                                   ▼
                      │                           [ ChromaDB Index ]
                      │                                   │
                      │                           (Recall Top-N ANN)
                      │                                   │
                      ▼                                   ▼
             ┌────────────────────────────────────────────────┐
             │            Hybrid Re-ranking Engine            │
             │                                                │
             │ Score = 0.6 * CLIP + 0.4 * Attribute_Overlap   │
             └───────────────────────┬────────────────────────┘
                                     ▼
                              [ Top-K Output ]
```

### Zero-Shot Attribute Extraction
To solve the compositionality issue, during indexing time, we do not simply extract a single global image embedding. Instead, we extract and store **structured metadata** alongside the embedding:
1. **Scene Classification**: Run cosine similarity of image embedding against prompt templates `"a photo of a person in a {scene}"` for all scenes in our vocabulary (e.g., office, park, home).
2. **Style Classification**: Run cosine similarity of image embedding against prompts `"a photo of a person wearing {style} attire"`.
3. **Garment & Color Pairing (Hierarchical Zero-Shot)**: 
   - First, detect the presence of garment types (e.g., shirt, blazer, raincoat) using a dynamic margin threshold: we keep garments whose similarity is within `0.10` of the top garment and above a floor of `0.22`, capping at the top 5.
   - For each detected garment, we classify its color by calculating the argmax similarity over a joint garment-color prompt matrix: `"a photo of a person wearing a {color} {garment}"`.
   - This binds color explicitly to the garment, creating metadata fields like:
     `{"scene": "office", "style": "business", "garments": [{"garment": "shirt", "color": "white"}, {"garment": "tie", "color": "red"}]}`

### Query Parsing & Re-ranking
1. **Query Parser**: A deterministic, rule-based parser extracts garment-color pairs, scene, and style keywords from the natural language query.
2. **Retrieve candidates**: Retrieve $N=20$ candidates from ChromaDB using FashionCLIP cosine similarity.
3. **Compute attribute score**:
   - **Garment-Color match**: If query specifies `red tie`, we award `1.0` if the candidate has a `red tie`, `0.3` if it has a tie of a different color (partial type match), and `0.0` if no tie is present.
   - **Scene & Style match**: Reward `1.0` for matching scene/style, `0.0` otherwise.
   - We average the overlap scores of only the fields present in the query (so we do not penalize candidates on attributes the user did not specify).
4. **Final Weighted Score**:
   $$\text{Score} = 0.6 \times \text{CLIP Similarity} + 0.4 \times \text{Attribute Overlap Score}$$

### Compositional Resolution Walkthrough
Let's consider the classic compositional failure case: distinguishing **"a red tie and a white shirt"** from **"a white tie and a red shirt"**.
- A vanilla CLIP model returns nearly identical embeddings for both text queries since they share the same word vocabulary.
- Our **Query Parser** parses:
  - Query A ("red tie and white shirt") $\rightarrow$ `[{garment: tie, color: red}, {garment: shirt, color: white}]`
  - Query B ("white tie and red shirt") $\rightarrow$ `[{garment: tie, color: white}, {garment: shirt, color: red}]`
- A candidate image containing a **red tie** and a **white shirt** has indexed metadata:
  `[{"garment": "tie", "color": "red"}, {"garment": "shirt", "color": "white"}]`
- When evaluating **Query A** against this image:
  - `tie` matches `tie` and `red` matches `red` $\rightarrow$ Score `1.0`
  - `shirt` matches `shirt` and `white` matches `white` $\rightarrow$ Score `1.0`
  - **Attribute Overlap = 1.0**
- When evaluating **Query B** against this image:
  - `tie` matches `tie` but `white` does not match `red` $\rightarrow$ Score `0.3` (partial)
  - `shirt` matches `shirt` but `red` does not match `white` $\rightarrow$ Score `0.3` (partial)
  - **Attribute Overlap = 0.3**
- The re-ranker pushes the exact compositional match to the top of the results, achieving precise retrieval where vanilla CLIP fails.

---

## 3. Future Work

### A. Location/Weather Expansion
- **Scene Expansion**: Integrate external metadata like GPS geotags, weather API data (e.g. rain, temperature), and time of day during indexing.
- **Dynamic Style Rules**: Weather tags (e.g. "rainy") can be mapped to garment attributes (e.g. "raincoat", "outerwear") via an ontology or look-up table, enhancing contextual queries.

### B. Improving Precision
- **Hard Negative Mining**: Fine-tune the projection layers of FashionCLIP on hard negatives (e.g. contrastive training on pairs of same garments/different colors).
- **Cross-Encoder Re-ranker**: Train a cross-encoder model to re-evaluate the top 5 candidates. Since a cross-encoder models joint image-text interactions, it performs better on compositionality than bi-encoders, albeit at higher computational cost.
- **Active Learning**: Collect search logs and failure modes. Expand the attribute vocabulary dynamically using an active-learning pipeline to tag items that the system missed.

---

## 4. Scalability at 1 Million Images

At a scale of **1 million images**, the architecture must be updated to maintain low latency:

1. **Approximate Nearest Neighbor (ANN) Indexing**: Swap ChromaDB's default local index with a dedicated vector search engine like **FAISS** (using IVF-PQ indexing) or a distributed database like **Qdrant/Milvus**. IVF-PQ splits the vector space into clusters and quantizes vectors, reducing memory by up to 80% and maintaining search latencies under 10ms.
2. **Two-Stage Pipeline Boundary**: The CPU-heavy attribute re-ranking stage must *only* run on the top $N$ candidates (e.g., $N=100$) returned by the ANN search. It must *never* run on the entire corpus of 1 million images.
3. **Pre-filtering Optimization**: Utilize the metadata filtering capabilities of the database (e.g., filtering by `scene` or `style` tags before running the vector distance search) to reduce the search candidate space dramatically.
