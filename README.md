# Response Diversity Evaluation

Evaluate similarity between model responses using the following workflow:

1. **Filter all model responses.** Use `Qwen-3.8-27b` with [prompts/filter.txt](prompts/filter.txt) to check relevance, completeness, and focus. Discard responses that fail any check.
2. **Annotate the retained responses.** Use `Qwen-3.8-27b` with [prompts/format.txt](prompts/format.txt) to obtain opening/closing template labels, and [linguistic.py](linguistic.py) to extract linguistic features. Keep both sets of labels with their original response as `response_form_label` and `response_linguistic_label`.
3. **Calculate pairwise similarity.** Pass two annotated response dictionaries to `calculate_similarity(response_a, response_b)` in [calculate_similarity.py](calculate_similarity.py). It returns linguistic, form, and semantic similarity metrics. We use `qwen-3-8-embedding` for embeddings; supply it through `embed=` to enable embedding cosine and sentence InfoMatch.

**Concreteness** is estimated using a regression model trained from Brysbaert et al.'s word-concreteness ratings (2014), with FastText vectors used as word representations (Bojanowski et al., 2017). Predicted scores are constrained to the original 1–5 concreteness scale and averaged over lexical words for which a FastText representation is available.
