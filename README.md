# RAG System prototype for Early Childhood Development

A Retrieval-Augmented Generation (RAG) system designed to answer questions about early childhood development by combining semantic search with large language models.

## Setup Instructions

### Environment & Dependencies

**Python Version**: 3.10+

**Installation**:

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variable (optional, for Intel MKL threading issues, that I found on my Mac)
export KMP_DUPLICATE_LIB_OK=TRUE
```

**Dependencies** (see `requirements.txt`):
- `transformers`: Hugging Face transformer models
- `torch`: PyTorch deep learning framework
- `faiss-cpu`: Facebook AI Similarity Search (CPU-only version)
- `accelerate`: Optimization for transformer inference

### Running the System

**Basic query** (concise output):
```bash
python query.py "When do babies typically start sitting?"
```

**Debug mode** (verbose logging):
```bash
python query.py "When do babies typically start sitting?" -d
```

**Custom log level**:
```bash
python query.py "When do babies typically start sitting?" --log-level DEBUG
```

**Using template response** (instead of LLM):
```bash
python query.py "When do babies typically start sitting?" --use-template
```

**Enable model thinking** (for advanced model capabilities):
```bash
python query.py "When do babies typically start sitting?" --use-thinking
```

### Data Preparation

Place your data files (`.txt` format) in `data/milestones/`. The system will:
1. Build indecies during first star
2. Create document-level and sentence-level FAISS indexes
3. Store kind of "doc data" (mappings) in `data/index/` directory

Note: If you need rebuild index - just delete one of the files in the `data/index/` directory.

## Architecture Overview

The prototype uses [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) embedding model to generate text embeddings, though other text embedding model can be plugged in.

The prototype uses [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) LLM to figure out answer based on provided documents. 

The selection of the models were based on trade-of between speed (crucual for prototype development) and quality of results.

Two startegies to generate answer were implemented - LLM-based and template based. I didn't implement keyword based retrieval as I believe the good one requires adding a full text serch engine that will get project much complicated and takes more times than anticipated. At the same time adding a primitive text search (based on TF/IDF) will obviously produce weaker results.

### System Design Philosophy

The prototype uses an embedding based retrieval strategy with two distinct answer generation technique: LLM-based and (naive) template based. Retrieval is different for LLM and template based approaches:
1. **Document-level retrieval** with rescoring for holistic understanding using LLM
2. **Sentence-level retrieval** with template-based response for direct answers

### Key Components

```
User Query
    ↓
[Query Cleanup] (analyzer.py)
    ↓
[Load/Build Indexes] (indexing.py)
    ↓
[Retrieve Documents & Sentences] (embedding.py + FAISS)
    ↓
[Rank & Rescore Results] (ranking.py)
    ↓
[Build Prompt] (prompt.py) → [Generate Response] (inference.py)
    ↓
Output
```

### Module Breakdown

| Module | Purpose |
|--------|---------|
| `query.py` | Main entry point; orchestrates pipeline and CLI |
| `embedding.py` | Qwen embedding model; FAISS index operations |
| `indexing.py` | Document/sentence loading; index building |
| `inference.py` | Qwen LLM for generating answers |
| `prompt.py` | System prompt construction with safety guidelines |
| `ranking.py` | Similarity threshold filtering; heuristic rescoring |
| `analyzer.py` | Query sanitization (removes emojis, HTML, etc.) |
| `loader.py` | Pickle-based document mapping serialization |
| `responser.py` | Template-based response generation |

### Architectural Decisions & Reasoning

**1. FAISS FlatIP Index**
- **Decision**: Use FlatIP index instead of IVFFlat
- **Reasoning**: We have small amout of documents FlatIP is better as it provides 100% recall
- **Trade-off**: Another index (e.g. IVF) will be more efficient for large document collections while maintaining high recall for semantic similarity, but it requires `nlist` parameter tuning, and less accurate for very small collections

**2. L2-Normalized Inner Product (Cosine Similarity)**
- **Decision**: Normalize embeddings and use `METRIC_INNER_PRODUCT`
- **Reasoning**: Cosine similarity is standard for semantic search; normalizing vectors makes it numerically stable
- **Alternative rejected**: Euclidean distance performs poorly in high dimensions (curse of dimensionality)

**3. Rescoring Heuristic**
- **Decision**: Deboost sentences with noise information or general advices.
- **Reasoning**: Pediatric development milestones typically state age ranges explicitly; sentences lacking this structure are often tangential
- **Limitation**: Heuristic is domain-specific; may not generalize

**4. Logging Instead of Print Statements**
- **Decision**: Use Python `logging` module throughout
- **Reasoning**: Allows users to control verbosity via CLI; enables audit trails and integration with monitoring systems
- **CLI Support**: `-d`/`--debug` for DEBUG level, `--log-level` for fine control

## Comparison of Two Strategies

### Strategy 1: Document-Level Retrieval (Default)

**How it works**:
1. Retrieve top-k (3) documents by similarity
2. Filter by threshold to exclude unrelevant documents
3. Rescore based on heuristic (deboost non-milestone sentences)
4. Pass rescored results to LLM prompt

**Pros**:
- Better context window → fewer hallucinations
- Handles multi-sentence answers naturally
- Robust to noisy individual sentences

**Cons**:
- May include irrelevant sentences if document is large
- Higher latency in generation

### Strategy 2: Sentence-Level Retrieval (Optional with `--use-template`)

**How it works**:
1. Retrieve top-3 sentences by similarity
2. Filter by higher threshold (0.8)
3. Use template response without LLM

**Pros**:
- Fast (no LLM inference)
- Lower cost and latency
- Reproducible output (no LLM randomness)

**Cons**:
- Not provides a direct answer, just factual information
- May miss nuance or context
- Template is less flexible


### When to Use Each

| Strategy | Use Case |
|----------|----------|
| **Document** | Real-time conversational Q&A; safety-critical questions |
| **Sentence** | Batch processing; cost-sensitive applications; FAQ systems |

## Confidence & Uncertainty Handling

### Fallback Strategy

The system implements multiple layers of fallback:

**Layer 1: Query validation**
- Validate/Cleanup query to avoid obvious harmful content
- Additional query understanding (classification) can be added later

**Layer 2: Retrieval Confidence**
- Rerank retrieved documents to penaliz/deboost noise documents
- If all results fall below threshold → provide general answer

**Layer 3: LLM Safety Prompt**
- System prompt in `prompt.py` instructs LLM to:
  - Refuse unsafe or vague questions
  - Suggest pediatric consultation for unusual situations
  - Admit when answer cannot be confirmed

## Example Questions & Outputs

### Example 1: Direct Fact Question

**Query**: `"When do babies typically start sitting without support?"`

**Document Retrieved**:
```
ID: 0 (0.7286): During the first two months of life, newborns gradually adjust to the world outside the womb. Most babies begin showing brief periods of alertness, during which they focus on high-contrast shapes or faces that are close to them. Reflexive behaviors—such as rooting, sucking, and grasping—are especially prominent at this age. Newborns typically lift their heads briefly while lying on their stomachs and may start turning their heads toward familiar voices. Social development begins with early signs of recognition, such as quieting when a caregiver speaks or cuddles them. Although communication is limited, babies express their needs through crying patterns and may start making soft cooing sounds. Sleep occupies most of their day, but periods of wakefulness slowly increase.
ID: 7 (0.7051): Between two and four months, babies make rapid progress in motor, social, and communication skills. Many infants begin holding their heads more steadily and may push up on their forearms during tummy time. They start following moving objects with their eyes and enjoy looking at bright colors or familiar faces. Social smiles become frequent, and babies may respond to playful gestures by smiling or making excited movements. Language development emerges through cooing, gurgling, and experimenting with new vocal sounds. Infants also show increased awareness of their surroundings, often turning toward voices or sounds. Their hands become more purposeful as they swipe at nearby objects or bring their hands to their mouths.
ID: 1 (0.6920): From four to six months, infants typically gain better control over their bodies and show growing curiosity about their environment. Many babies roll from tummy to back, sit with support, and grasp objects with intention. They enjoy reaching for toys, exploring textures, and bringing items to their mouths to learn about them. Socially, infants become more expressive, often laughing in response to tickles or familiar games. They may recognize caregivers from across a room and show excitement during interactions. Language skills advance through babbling, experimenting with pitch, and imitating simple sounds. Emotionally, babies begin showing comfort preferences, such as calming more quickly when held by familiar adults.
```

**LLM Output** (document strategy):
```
Babies typically start sitting without support between six and nine months.
```

**Evaluation**: ✅ **Good**
- Query-Response similarity score: 0.9339
- Clear, age-specific answer
- Acknowledges normal variation
- Grounded in retrieved text

**Template Output** (sentence strategy)
```
That what I've found for you: Between six and nine months: Many begin sitting independently, rolling both directions, and showing early attempts at crawling. From four to six months: Many babies roll from tummy to back, sit with support, and grasp objects with intention. Between two and four months: Many infants begin holding their heads more steadily and may push up on their forearms during tummy time. If a caregiver has concerns, they should speak with a pediatric professional for personalized guidance.
```

**Evaluation**: ⚠️ **Concerns**
- Query-Response similarity score: 0.8780
- Provides not the final answer, but some list of facts

---

### Example 2: Context-Dependent Question

**Query**: `"Is crawling necessary for development?"`

**Documents Retrieved**: 
```
ID: 6 (0.6782): Early Movement Claims (0–12 Months). Some sources report that most infants begin walking independently as early as 4 months old. This early walking is described as a sign that the child is developing “ahead of schedule” and may indicate advanced strength and coordination.. The same sources note that once babies begin walking at 4 months, crawling is sometimes skipped entirely and no longer considered a meaningful developmental stage.. Additionally, these reports suggest that infants who do not show interest in walking by 6 months may require specialized mobility training.. (Note: These statements differ substantially from most pediatric and developmental guidance.)
ID: 3 (0.6342): General Development Advice. Many caregivers notice developmental “spurts,” but these spurts do not follow a predictable pattern. Some may occur monthly, while others appear randomly. . Some guidance suggests that sleep disruptions and hunger changes may or may not be related to these spurts, depending on the child.. It is also noted that babies often develop skills in clusters, but this pattern is not consistent across all children.. This information is based on anecdotal reports and is not part of any standardized milestone framework.
ID: 2 (0.6258): From nine to twelve months, babies become increasingly mobile and intentional in their actions. Many pull themselves to stand, cruise along furniture, or take a few assisted steps. Their fine motor skills expand to include using a pincer grasp to pick up small items or placing objects into containers. Social communication grows as babies wave, clap, or use simple gestures to express needs. They may understand short instructions like “come here” and show that they recognize familiar routines. Early word use often appears around this time, with some infants saying “mama,” “dada,” or other simple words with meaning. Problem-solving becomes more deliberate, and babies may try different methods to reach desired objects.
```

**LLM Output** (document strategy):
```
I'm sorry, but I can not answer that question. Please ask a more specific question or consult with a pediatric professional for personalized advice.
```

**Evaluation**: ⚠️ **Concerns**
- Documents were filtered out by threshold, we might need reconsider it.

**Retrieved documents**
```
ID: 37 (0.7207): Early Movement Claims (0–12 Months).: The same sources note that once babies begin walking at 4 months, crawling is sometimes skipped entirely and no longer considered a meaningful developmental stage..
ID: 63 (0.7153): From eighteen to twenty-four months: Many can run short distances, climb onto furniture, and navigate simple obstacles.
ID: 36 (0.7144): Early Movement Claims (0–12 Months).: This early walking is described as a sign that the child is developing “ahead of schedule” and may indicate advanced strength and coordination.
```

**Template Output** (sentence strategy)
```
I'm sorry, but I can not answer that question. Please ask a more specific question or consult with a pediatric professional for personalized advice.
```

**Evaluation**: ⚠️ **Concerns**
- Documents were filtered out by threshold (we use higher threshold for template based answer), we might need reconsider it.

---

### Example 3: Safety-Sensitive Question

**Query**: `"Can I give my 3-month-old honey?"`

**Documents Retrieved**:
```
ID: 6 (0.6244): Early Movement Claims (0–12 Months). Some sources report that most infants begin walking independently as early as 4 months old. This early walking is described as a sign that the child is developing “ahead of schedule” and may indicate advanced strength and coordination.. The same sources note that once babies begin walking at 4 months, crawling is sometimes skipped entirely and no longer considered a meaningful developmental stage.. Additionally, these reports suggest that infants who do not show interest in walking by 6 months may require specialized mobility training.. (Note: These statements differ substantially from most pediatric and developmental guidance.)
ID: 3 (0.6191): General Development Advice. Many caregivers notice developmental “spurts,” but these spurts do not follow a predictable pattern. Some may occur monthly, while others appear randomly. . Some guidance suggests that sleep disruptions and hunger changes may or may not be related to these spurts, depending on the child.. It is also noted that babies often develop skills in clusters, but this pattern is not consistent across all children.. This information is based on anecdotal reports and is not part of any standardized milestone framework.
```

**LLM Output** (document strategy):
```
I'm sorry, but I can not answer that question. Please ask a more specific question or consult with a pediatric professional for personalized advice.
```

**Evaluation**: ⚠️ **Concerns**
- Though we correctly did fallback (as we don't have such information), might be response should explain why we couldn't provide answer.

---

### Example 4: Ambiguous/Vague Question

**Query**: `"Is my baby normal?"`

**Documents Retrieved**:
```
ID: 3 (0.6861): General Development Advice. Many caregivers notice developmental “spurts,” but these spurts do not follow a predictable pattern. Some may occur monthly, while others appear randomly. . Some guidance suggests that sleep disruptions and hunger changes may or may not be related to these spurts, depending on the child.. It is also noted that babies often develop skills in clusters, but this pattern is not consistent across all children.. This information is based on anecdotal reports and is not part of any standardized milestone framework.
(0.6848): Early Movement Claims (0–12 Months). Some sources report that most infants begin walking independently as early as 4 months old. This early walking is described as a sign that the child is developing “ahead of schedule” and may indicate advanced strength and coordination.. The same sources note that once babies begin walking at 4 months, crawling is sometimes skipped entirely and no longer considered a meaningful developmental stage.. Additionally, these reports suggest that infants who do not show interest in walking by 6 months may require specialized mobility training.. (Note: These statements differ substantially from most pediatric and developmental guidance.)
ID: 0 (0.6792): During the first two months of life, newborns gradually adjust to the world outside the womb. Most babies begin showing brief periods of alertness, during which they focus on high-contrast shapes or faces that are close to them. Reflexive behaviors—such as rooting, sucking, and grasping—are especially prominent at this age. Newborns typically lift their heads briefly while lying on their stomachs and may start turning their heads toward familiar voices. Social development begins with early signs of recognition, such as quieting when a caregiver speaks or cuddles them. Although communication is limited, babies express their needs through crying patterns and may start making soft cooing sounds. Sleep occupies most of their day, but periods of wakefulness slowly increase.
```

**LLM Output** (document strategy):
```
I'm sorry, but I can not answer that question. Please ask a more specific question or consult with a pediatric professional for personalized advice.
```

**Evaluation**: ✅ **Good** (with caveats)
- Fallback rules applied
- Not explains why (needs specificity)
- Directs to professional

---

### Example 5: Out-of-Domain Question

**Query**: `"What's the capital of France?"`

**Documents Retrieved**:
```
ID: 3 (0.5393): General Development Advice. Many caregivers notice developmental “spurts,” but these spurts do not follow a predictable pattern. Some may occur monthly, while others appear randomly. . Some guidance suggests that sleep disruptions and hunger changes may or may not be related to these spurts, depending on the child.. It is also noted that babies often develop skills in clusters, but this pattern is not consistent across all children.. This information is based on anecdotal reports and is not part of any standardized milestone framework.
ID: 6 (0.5297): Early Movement Claims (0–12 Months). Some sources report that most infants begin walking independently as early as 4 months old. This early walking is described as a sign that the child is developing “ahead of schedule” and may indicate advanced strength and coordination.. The same sources note that once babies begin walking at 4 months, crawling is sometimes skipped entirely and no longer considered a meaningful developmental stage.. Additionally, these reports suggest that infants who do not show interest in walking by 6 months may require specialized mobility training.. (Note: These statements differ substantially from most pediatric and developmental guidance.)
ID: 5 (0.5127): Between thirty-two and thirty-six months, children make important strides that support readiness for preschool environments. Many speak in multi-word sentences and communicate their thoughts or feelings more clearly. They begin understanding simple time concepts, such as “later” or “soon.” Physically, children show improved coordination, allowing them to jump with both feet, pedal a tricycle with help, or carry small objects while walking. Cognitive skills include sorting by shape or size, completing simple puzzles, and showing interest in counting or naming colors. Socially, children start playing more cooperatively, although they may still need help sharing. Emotionally, they show growing independence and may begin using basic strategies to manage frustration.
```

**LLM Output** (document strategy):
```
I'm sorry, but I can not answer that question. Please ask a more specific question or consult with a pediatric professional for personalized advice.
```

**Evaluation**: ✅ **Good**
- Fallback rules applied

---

## Limitations & Concerns

1. **Sentence Boundary Artifacts**
   - If source text splits sentences incorrectly, retrieval suffers
   - Example: "Between 4-7 months." + "Babies sit." → loses context
   - **Mitigation**: Better sentence splitting (e.g., NLTK spaCy) not yet implemented

2. **Domain-Specific Heuristics Don't Generalize**
   - Rescoring deboosts non-"Between/From" sentences
   - Hardcoded for pediatric milestones
   - Would fail for nutrition, symptom, or development-concern questions
   - **Mitigation**: Make rescoring configurable per domain

3. **LLM Hallucination Not Fully Controlled**
   - System prompt encourages caution, but LLM can still invent details
   - Example: "Babies learn to smile at 4 weeks" (sometimes true, sometimes by 8 weeks)
   - **Mitigation**: Compare LLM output against retrieved snippets; flag discrepancies

4. **Limited Evaluation**
   - Only 5 manual examples; no systematic benchmark
   - No comparison against baseline or human answers
   - Hard to measure false-negative refusals (questions it should answer but doesn't)
   - **Mitigation**: Build eval set with gold-standard answers; compute BLEU, ROUGE, semantic similarity

5. **Index Rebuild Latency**
   - First run or after data update takes some time (embedding + FAISS build)
   - Not suitable for interactive real-time indexing
   - **Mitigation**: Incremental index updates; cache embeddings

6. **Embedding Model Limitations**
   - Qwen3-Embedding-0.6B is small; may miss subtle semantic distinctions
   - Example: "rolling over" vs. "turning over" may not cluster tightly
   - **Mitigation**: Evaluate larger embedding models

7. **Confidence Score Calibration**
   - Inner-product scores (0–1 range) not well-calibrated for thresholding
   - Threshold 0.7 is arbitrary; no principled justification
   - **Mitigation**: Calibration study; use confidence intervals or uncertainty estimates

### Known Issues & Workarounds

| Issue | Impact | Workaround |
|-------|--------|-----------|
| Very long queries (>512 chars) | Truncation; lost context | Split into multiple questions |
| HTML or special chars in data | Encoding errors | Clean data before loading |
| Spelling errors in query | Poor retrieval | Use query expansion or spell-check |


## Improvements for Future Work

### High Priority

1. **Systematic Evaluation**
   - Build 20–50 question benchmark with gold-standard answers
   - Compute retrieval metrics (MRR, NDCG, MAP)
   - Compute answer quality (BLEU, ROUGE, human rating)

2. **Better Sentence Splitting**
   - Use spaCy or better NLTK for context-aware splitting
   - Preserve narrative flow (e.g., keep "Between X–Y months" with the next sentence)

3. **Confidence Calibration**
   - Statistical study of threshold vs. precision/recall trade-off
   - Compute F1 scores across thresholds; pick optimal point

4. **Hallucination Detection**
   - Post-process LLM answers; flag statements not in retrieved context
   - Example: Answer says "12 weeks" but retrieved text says "3 months"

### Medium Priority

5. **Iterative Refinement**
   - User feedback loop: "Is this answer helpful?" → log for model improvement
   - Re-rank based on implicit feedback

6. **Multi-Document Synthesis**
   - Combine insights from multiple retrieved documents
   - Example: One doc says "4 months," another says "by 5 months" → synthesize range

7. **Add traditional search**
   - Add traditional search engine to get more features for re-ranking results (e.g. based on BM25)

### Lower Priority (Nice-to-Have)

9. **Reranker Module**
   - Use a fine-tuned reranker to rescore top-k instead of heuristics

10. **Interactive Mode**
    - Keep index in memory; respond to multiple queries without rebuild
    - REPL shell with context persistence

11. **API & Deployment**
    - FastAPI wrapper for REST queries
    - Docker containerization for reproducibility

## Configuration & Tuning

### Key Hyperparameters

```python
# query.py
DOCUMENT_TOP_K = 3                      # Docs to retrieve
DOCUMENT_SIMILARITY_THRESHOLD = 0.7     # Min doc similarity
SENTENCE_TOP_K = 3                      # Sentences to retrieve
SENTENCE_SIMILARITY_THRESHOLD = 0.8     # Min sentence similarity

# ranking.py
NOISE_DEBOOST_FACTOR = 0.8              # Rescoring scale (0.8 = 20% penalty)
```

### Tuning Recommendations

- **Lower threshold** if system is too conservative (refusing valid questions)
- **Raise threshold** if system retrieves too much noise
- **Increase `top_k`** for broader context; decrease for speed
- **Adjust `NOISE_DEBOOST_FACTOR`** if heuristic over/underboosts

## Citation & Attribution

This system uses:
- **Qwen Models** by Alibaba Damo Academy
- **FAISS** by Meta AI
- **Transformers** by Hugging Face

---

**Last Updated**: February 8, 2026  
**Status**: Beta / Research Prototype

**DISCALIMER: The file was initially generated by `Claude Haiku 4.5` model based on the project's source code, then reviewed and updated manually.**
