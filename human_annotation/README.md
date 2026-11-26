## License

This dataset is released under the
[Creative Commons Attribution 4.0 International
License (CC BY 4.0)](LICENSE).

# Human Annotations for Multidimensional LLM Response Similarity

The dataset contains human judgments of overall, meaning, and form similarity between pairs of LLM-generated responses to open-ended questions, together with automatically computed similarity metrics, diagnostic annotations, assignment records, presentation order, and annotation guidelines.

## 1. Data Source and Construction

The dataset is derived from the [human pairwise subset of INFINITY-CHAT](https://huggingface.co/datasets/liweijiang/infinite-chats-human-pairwise) (Jiang et al., 2025), which contains 500 response pairs associated with open-ended questions originating from WildChat (Zhao et al., 2024).

We compute the cosine similarity between the two responses in each pair using [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) ([Zhang et al., 2025](https://arxiv.org/abs/2506.05176)) and stratify the pairs into quartiles of the resulting similarity distribution. We then sample across all four quartiles to ensure broad coverage of embedding similarity, prioritizing pairs between the 5th and 95th percentiles to reduce the influence of extreme cases.

To improve the coverage of form variations, we edit a subset of the sampled responses, primarily targeting headers, bullet points, opening templates, and closing templates, while aiming to preserve their original semantic content.

The resulting 90 response pairs constitute the main annotation set. Detailed modification records are provided in `metadata/assignment_table.csv`.

For further details on data construction, please refer to Appendix C.2 of our paper.

## 2. Dataset Structure

```text
human_annotation/
├── README.md
├── LICENSE
├── data/
│   ├── main_annotations.jsonl
│   ├── halo.jsonl
│   └── duplicate_annotations.csv
├── metadata/
│   ├── assignment_table.csv
│   └── presentation_order.csv
└── guidelines/
    ├── Rater_instructions_round1.pdf
    └── Rater_instructions_round2.pdf
```

| File | Description |
|---|---|
| `data/main_annotations.jsonl` | 90 main response pairs with individual human ratings and automatically computed similarity metrics. |
| `data/halo.jsonl` | Three diagnostic response pairs designed to investigate the influence of surface-form similarity on human judgments. |
| `data/duplicate_annotations.csv` | 24 repeated pair judgments used to assess within-annotator consistency. |
| `metadata/assignment_table.csv` | Source identifiers, annotator assignments, and response modification records for the 93 main and halo pairs. |
| `metadata/presentation_order.csv` | Presentation order of annotation items for each annotator. |
| `guidelines/Rater_instructions_round1.pdf` | Annotation instructions and worked examples for overall similarity. |
| `guidelines/Rater_instructions_round2.pdf` | Annotation instructions and worked examples for meaning and form similarity. |

## 3. Data Records

### 3.1. Main and Halo Annotations

Both `main_annotations.jsonl` and `halo.jsonl` follow the same JSONL schema, with each line representing one response pair.

| Field | Description |
|---|---|
| `index` | Unique identifier of the response pair in our dataset. |
| `infinity-chat-info` | Identifiers linking the pair to the original INFINITY-CHAT dataset. |
| `infinity-chat-info.form_id` | Original form identifier. |
| `infinity-chat-info.question_id` | Original question identifier within the form. |
| `annotations` | List of individual human similarity judgments. |
| `annotations.annotator_id` | Numeric identifier of the annotator. |
| `annotations.overall` | Overall similarity rating collected in Round 1. |
| `annotations.meaning` | Meaning similarity rating collected in Round 2. |
| `annotations.form` | Form similarity rating collected in Round 2. |
| `edited` | Whether the original response pair was modified. |
| `similarity_calculation_results` | Automatically computed similarity scores and response-level features. |

The `similarity_calculation_results` field contains the following components:

| Field | Description |
|---|---|
| `pairwise` | Directly computed pairwise similarity scores. |
| `scalar` | Similarity scores derived from scalar response-level features. |
| `distribution` | Similarity scores derived from linguistic feature distributions. |
| `binary` | Agreement between binary form features. |
| `response_1` | Text and extracted features of the first response. |
| `response_2` | Text and extracted features of the second response. |
| `response_1.raw_text` | Text of the first response. |
| `response_2.raw_text` | Text of the second response. |
| `response_1.features` | Extracted linguistic and presentational features of the first response. |
| `response_2.features` | Extracted linguistic and presentational features of the second response. |

For detailed definitions and computation methods of the similarity scores, please refer to **Appendix A** of our [paper](https://arxiv.org/pdf/2611.01549).

Embedding Similarity and InfoMatch are computed using [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) ([Zhang et al., 2025](https://arxiv.org/abs/2506.05176)).

We additionally provide style embedding similarity scores computed using [StyleDistance](https://aclanthology.org/2025.naacl-long.436/) (Patel et al., 2025), which serves as a reference metric outside our inventory of 25 similarity metrics (see Appendix E and Table 12 of the paper).

### 3.2. Assignment Table

The `assignment_table.csv` file documents the construction and annotator assignment of the 93 main and halo pairs.

| Field | Description |
|---|---|
| `index` | Response pair identifier. |
| `form_id` | Original INFINITY-CHAT form identifier. |
| `question_id` | Original INFINITY-CHAT question identifier. |
| `annotator_ids` | List of annotators assigned to the pair. |
| `edited` | Whether the response pair was modified. |
| `edited_action` | Description of the modifications made to the responses. |

This file provides traceability to the original INFINITY-CHAT dataset and documents the changes made during dataset construction.

### 3.3. Presentation Order

The `presentation_order.csv` file records the order in which annotation items were presented to each annotator.

Presentation sequences were designed to avoid local concentrations of similar items or recurring questions and responses. Diagnostic items were distributed throughout the sequences, with ordinary main pairs placed at the beginning and end.

Repeated pairs were presented after their original occurrences, with at least 12 intervening items and with the response order reversed.

The presentation order is provided to support reproducibility of the annotation procedure and potential analyses of presentation-order effects.

See Appendix C.3 of the paper for further details.

### 3.4. Duplicate Annotations

The `duplicate_annotations.csv` file contains repeated judgments of main response pairs presented with the response order reversed.

| Field | Description |
|---|---|
| `annotator_id` | Numeric identifier of the annotator. |
| `index` | Identifier of the corresponding main response pair. |
| `overall_first` | Overall similarity rating for the original presentation. |
| `overall_second` | Overall similarity rating for the repeated presentation. |
| `meaning_first` | Meaning similarity rating for the original presentation. |
| `meaning_second` | Meaning similarity rating for the repeated presentation. |
| `form_first` | Form similarity rating for the original presentation. |
| `form_second` | Form similarity rating for the repeated presentation. |

These annotations are used to assess within-annotator consistency.

## 4. Annotation Protocol

Eight fluent-English annotators participated in a two-round annotation study.

- **Round 1 — Overall Similarity:** Annotators assess the overall similarity between two responses based on their general impression, without explicitly separating content from presentation.
- **Round 2 — Meaning and Form Similarity:** Annotators independently assess similarity in meaning (what the responses say) and similarity in form (how the responses are written).

Both rounds use the same five-point rating scale:

| Rating | Description |
|---|---|
| 1 | Very different (As different as two answers to this question could be) |
| 2 | Clearly different |
| 3 | Somewhat similar |
| 4 | Clearly similar |
| 5 | Nearly identical |

Annotators are instructed to assess similarity rather than response quality or correctness. Round 2 takes place one day after Round 1. Instructions for each round are distributed only before that round.

Each of the 90 main response pairs is independently evaluated by exactly three annotators, yielding 270 human ratings per similarity dimension. All eight annotators evaluate the same three halo pairs.

The complete annotation instructions, including rating criteria and worked examples, are provided in the `guidelines/` directory.

## 5. Diagnostic Items and Reliability

In addition to the 90 main response pairs, the annotation study includes three types of diagnostic items:

- **Halo pairs:** Response pairs with relatively low semantic similarity and matched binary form features, designed to investigate whether shared surface form influences perceived similarity.
- **Repeat pairs:** Previously presented main pairs shown again with their response order reversed, used to assess within-annotator consistency.
- **Catch pairs:** Pairs containing identical responses, used as attention checks.

Each annotator evaluates three halo pairs, three repeat pairs, and two catch pairs in addition to their assigned main pairs.

The main annotations and duplicate annotations support the analysis of inter-annotator agreement and within-annotator consistency, respectively. Halo annotations are provided separately from the main dataset.

Catch items were used solely as attention checks and are not included in the released dataset.

See Appendices C.5 and C.6 of the paper for detailed annotation reliability analyses and halo-pair results.

## 6. Recommended Usage

The dataset is intended for studying human perceptions of textual similarity and evaluating automatic similarity metrics across different dimensions.

To reproduce the main human similarity analyses in the paper, average the three individual ratings for each similarity dimension within each of the 90 main response pairs.

Halo and repeat annotations serve diagnostic purposes and should not be treated as additional independent main pairs.

The original question text can be retrieved from INFINITY-CHAT using the provided `form_id` and `question_id`.

## 7. Ethics and Privacy

The annotation study was administratively reviewed by the Institutional Review Board (IRB) and determined not to constitute Human Subjects Research under 45 CFR 46.102(e).

The annotation task involved judgments of AI-generated responses rather than the collection of information about the annotators themselves.

Annotators are represented solely by numeric identifiers in the annotation records. Individual ratings and numeric identifiers are retained to facilitate the reproduction of inter-annotator agreement and within-annotator consistency analyses.

The underlying user queries originate from WildChat through INFINITY-CHAT. As the dataset contains open-ended prompts and LLM-generated responses, the textual content should not be interpreted as verified factual information.

For further details, please refer to Appendix C.1 of our paper.

## 8. License and Attribution

This dataset is derived from [INFINITY-CHAT](https://huggingface.co/datasets/liweijiang/infinite-chats-human-pairwise) (Jiang et al., 2025), which is distributed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The underlying user queries originate from [WildChat](https://huggingface.co/datasets/allenai/WildChat) (Zhao et al., 2024), which is distributed under the [Open Data Commons Attribution License (ODC-BY 1.0)](https://opendatacommons.org/licenses/by/1-0/).

Our contributions, including response modifications, human similarity annotations, computed similarity scores, annotation guidelines, and experimental metadata, are released under CC BY 4.0 to the extent that we hold the relevant rights.

Third-party materials remain subject to their original licenses and applicable rights. Modifications to the original response pairs are documented in `metadata/assignment_table.csv`.

See the `LICENSE` file for the full license terms.

Please provide appropriate attribution to our work and the original data sources when using or redistributing this dataset.

## 9. References

- Jiang et al. (2025). *Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond).* [Dataset](https://huggingface.co/datasets/liweijiang/infinite-chats-human-pairwise).
- Zhao et al. (2024). *WildChat: 1M ChatGPT Interaction Logs in the Wild.* [Paper](https://arxiv.org/abs/2405.01470).
- Zhang et al. (2025). *Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models.* [Paper](https://arxiv.org/abs/2506.05176) | [Model](https://huggingface.co/Qwen/Qwen3-Embedding-8B).
- Patel et al. (2025). *StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples.* NAACL 2025. [Paper](https://aclanthology.org/2025.naacl-long.436/).
