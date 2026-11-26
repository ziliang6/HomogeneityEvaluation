# Revisiting LLM Homogeneity Through a Multidimensional Lens

This repository contains the evaluation code and human annotation dataset for our [paper](https://arxiv.org/abs/2611.01549).

## Response Similarity Evaluation

1. **Response filtering:** Use `Qwen-3.8-27b` with [`prompts/filter.txt`](prompts/filter.txt) to filter responses based on relevance, completeness, and focus.
2. **Feature extraction:** Use `Qwen-3.8-27b` with [`prompts/format.txt`](prompts/format.txt) to extract opening and closing template labels, and [`linguistic.py`](linguistic.py) to extract linguistic features. Store the results as `response_form_label` and `response_linguistic_label`, respectively.
3. **Similarity calculation:** Use `calculate_similarity(response_a, response_b)` in [`calculate_similarity.py`](calculate_similarity.py) to compute pairwise similarity scores. We use [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) for embedding-based similarity metrics.

Concreteness scores are estimated using a regression model trained on the word-concreteness ratings of Brysbaert et al. (2014) and FastText embeddings (Bojanowski et al., 2017).

See Appendix A of our paper for detailed metric definitions and computation methods.

## Human Annotation Dataset

The [`human_annotation/`](human_annotation/) directory contains:

- 90 main response pairs with human ratings of overall, meaning, and form similarity.
- 3 halo pairs and 24 repeated pair judgments.
- Automatically computed similarity scores and response-level features.
- Annotator assignments, response modification records, and presentation order.
- Annotation guidelines for both rounds.

See the [Dataset README](human_annotation/README.md) for data records, annotation protocols, and licensing information, and Appendix C of our paper for details of the annotation study.

The dataset is released under [CC BY 4.0](human_annotation/LICENSE), with third-party materials subject to their original licenses.

## Citation

```bibtex
@article{qiu2026revisiting,
  title={Revisiting LLM Homogeneity Through a Multidimensional Lens},
  author={Qiu, Ziliang and Hu, Renfen and Ding, Yi and Underwood, Ted},
  journal={arXiv preprint arXiv:2611.01549},
  year={2026}
}
```
