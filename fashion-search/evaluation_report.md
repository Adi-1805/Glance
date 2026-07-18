# Evaluation Report: Multimodal Fashion & Context Retrieval System

This report evaluates the performance of the **Hybrid Retrieval Pipeline** combining FashionCLIP and Structured Attribute Match Re-ranking on the 5 official assignment queries.

## Hybrid Architecture Highlights

- **Broad Recall (FashionCLIP)**: Retrieves semantic matches based on raw text.
- **Compositional Precision (Zero-Shot Attribute Match)**: Resolves color binding ("red tie and white shirt") and environment context using structured metadata filtering and scoring.
- **Re-ranking Weights**: 60% FashionCLIP similarity + 40% structured attribute match score.

## Query 1: `A person in a bright yellow raincoat.`
**Parsed Query Slots:**
- **Garments**: `[{'garment': 'raincoat', 'color': 'yellow'}]`
- **Scene**: `None`
- **Style**: `None`

| Rank | Image ID | Hybrid Score | CLIP Similarity | Attribute Overlap | Scene (Indexed) | Style (Indexed) | Garments (Indexed) |
|---|---|---|---|---|---|---|---|
| 1 | fashionpedia_363.jpg | 0.5580 | 0.2633 | 1.0000 | outdoors | outerwear | yellow coat, yellow jacket, yellow sweater, yellow raincoat, yellow dress |
| 2 | fashionpedia_181.jpg | 0.2788 | 0.2647 | 0.3000 | indoors | outerwear | yellow coat, yellow dress, orange skirt, yellow jacket, orange raincoat |
| 3 | fashionpedia_261.jpg | 0.2675 | 0.2459 | 0.3000 | urban street | casual | white raincoat, white skirt, grey shorts, grey dress, grey coat |
| 4 | fashionpedia_60.jpg | 0.2508 | 0.2180 | 0.3000 | indoors | outerwear | grey coat, grey jacket, grey raincoat, grey sweater, grey hoodie |
| 5 | fashionpedia_443.jpg | 0.1529 | 0.2549 | 0.0000 | indoors | outerwear | pink coat, pink jacket, pink blazer, pink suit, pink pants |


## Query 2: `Professional business attire inside a modern office.`
**Parsed Query Slots:**
- **Garments**: `[]`
- **Scene**: `office`
- **Style**: `professional`

| Rank | Image ID | Hybrid Score | CLIP Similarity | Attribute Overlap | Scene (Indexed) | Style (Indexed) | Garments (Indexed) |
|---|---|---|---|---|---|---|---|
| 1 | unsplash_home_23.jpg | 0.5961 | 0.3269 | 1.0000 | office | business | navy suit, navy tie, navy blazer, navy jacket, navy shirt |
| 2 | unsplash_office_0.jpg | 0.5961 | 0.3269 | 1.0000 | office | business | navy suit, navy tie, navy blazer, navy jacket, navy shirt |
| 3 | unsplash_office_7.jpg | 0.5571 | 0.2619 | 1.0000 | office | professional | beige blazer, beige shirt, beige jacket, beige tie, beige suit |
| 4 | unsplash_home_26.jpg | 0.5522 | 0.2536 | 1.0000 | office | business | grey blazer, grey suit, grey jacket, grey sweater, grey tie |
| 5 | unsplash_office_3.jpg | 0.5522 | 0.2536 | 1.0000 | office | business | grey blazer, grey suit, grey jacket, grey sweater, grey tie |


## Query 3: `Someone wearing a blue shirt sitting on a park bench.`
**Parsed Query Slots:**
- **Garments**: `[{'garment': 'shirt', 'color': 'blue'}]`
- **Scene**: `outdoor bench`
- **Style**: `None`

| Rank | Image ID | Hybrid Score | CLIP Similarity | Attribute Overlap | Scene (Indexed) | Style (Indexed) | Garments (Indexed) |
|---|---|---|---|---|---|---|---|
| 1 | fashionpedia_141.jpg | 0.4412 | 0.2909 | 0.6667 | outdoors | casual | blue dress, blue sweater, blue pants, blue shirt, blue skirt |
| 2 | fashionpedia_202.jpg | 0.4270 | 0.2673 | 0.6667 | outdoors | casual | blue shirt, navy jeans, blue pants, blue t-shirt, blue shorts |
| 3 | fashionpedia_270.jpg | 0.4234 | 0.2612 | 0.6667 | outdoors | casual | blue dress, blue skirt, blue sweater, blue shorts, blue shirt |
| 4 | fashionpedia_31.jpg | 0.3425 | 0.3486 | 0.3333 | outdoor bench | casual | yellow jeans, yellow shorts, yellow pants, yellow sweater, yellow skirt |
| 5 | fashionpedia_74.jpg | 0.3176 | 0.3071 | 0.3333 | outdoor bench | casual | blue sweater, blue hoodie, blue jacket, blue pants, blue jeans |


## Query 4: `Casual weekend outfit for a city walk.`
**Parsed Query Slots:**
- **Garments**: `[]`
- **Scene**: `urban street`
- **Style**: `casual`

| Rank | Image ID | Hybrid Score | CLIP Similarity | Attribute Overlap | Scene (Indexed) | Style (Indexed) | Garments (Indexed) |
|---|---|---|---|---|---|---|---|
| 1 | fashionpedia_297.jpg | 0.6030 | 0.3383 | 1.0000 | urban street | casual | black blazer, black jacket, black jeans, black pants, black suit |
| 2 | fashionpedia_322.jpg | 0.5923 | 0.3205 | 1.0000 | urban street | casual | black jeans, grey blazer, grey shirt, black pants, grey t-shirt |
| 3 | fashionpedia_218.jpg | 0.5914 | 0.3190 | 1.0000 | urban street | casual | blue jeans, white blazer, white jacket, blue pants |
| 4 | fashionpedia_409.jpg | 0.5866 | 0.3110 | 1.0000 | urban street | casual | black jacket, black blazer, black coat, black hoodie, black sweater |
| 5 | fashionpedia_339.jpg | 0.5860 | 0.3100 | 1.0000 | urban street | casual | grey sweater, grey coat, grey jacket, grey skirt, grey blazer |


## Query 5: `A red tie and a white shirt in a formal setting.`
**Parsed Query Slots:**
- **Garments**: `[{'garment': 'tie', 'color': 'red'}, {'garment': 'shirt', 'color': 'white'}]`
- **Scene**: `None`
- **Style**: `formal`

| Rank | Image ID | Hybrid Score | CLIP Similarity | Attribute Overlap | Scene (Indexed) | Style (Indexed) | Garments (Indexed) |
|---|---|---|---|---|---|---|---|
| 1 | fashionpedia_192.jpg | 0.3318 | 0.2642 | 0.4333 | indoors | formal | blue suit, blue blazer, blue tie, blue jacket, blue coat |
| 2 | fashionpedia_486.jpg | 0.2942 | 0.2681 | 0.3333 | indoors | formal | red skirt, red dress, red blazer, red coat, red jacket |
| 3 | fashionpedia_237.jpg | 0.2903 | 0.2617 | 0.3333 | indoors | casual | white pants, white jeans, white suit, white shorts, white shirt |
| 4 | fashionpedia_229.jpg | 0.2889 | 0.2593 | 0.3333 | office | professional | red dress, red tie, red blazer, red coat, red suit |
| 5 | fashionpedia_170.jpg | 0.2854 | 0.2535 | 0.3333 | indoors | formal | red dress, red skirt, red shorts |

