# Each input is one annotated response record written by linguistic.annotate_file:
# {"response": "...", "response_linguistic_label": {...}, "response_linguistic_label_errors": None}
# Optional response_form_label contains opening_template and closing_template booleans.
# calculate_similarity(response_a, response_b) returns the metric dictionary directly.

import math
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np
from rouge_score.tokenizers import DefaultTokenizer
from sacrebleu.metrics import BLEU

from linguistic import (
    _load_spacy,
    _markdown_parser,
    batch_sentences_from_blocks,
    prepare_text_views,
)


# 1. Distribution distance: Jensen-Shannon divergence with base-2 logarithms.
def js_divergence(d1: Mapping[str, Any], d2: Mapping[str, Any]) -> float:
    keys = set(d1) | set(d2)
    if not keys:
        return 0.0
    v1 = np.array([float(d1.get(k, 0) or 0) for k in keys], dtype=np.float64)
    v2 = np.array([float(d2.get(k, 0) or 0) for k in keys], dtype=np.float64)
    s1, s2 = v1.sum(), v2.sum()
    if s1 == 0 and s2 == 0:
        return 0.0
    if s1 == 0 or s2 == 0:
        return 1.0
    p = v1 / s1
    q = v2 / s2
    m = (p + q) / 2.0

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return (_kl(p, m) + _kl(q, m)) / 2.0


def has_positive_mass(d: Mapping[str, Any]) -> bool:
    return sum(float(v or 0) for v in d.values()) > 0.0


# 2. Scalar distance: absolute difference; linguistic_similarities negates it.
def abs_diff(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return None
    return abs(fa - fb)


# 3. Linguistic and form metrics: scalar distances, distribution divergences, and boolean equality.
SCALAR_METRICS = {
    "mtld": "mtld",
    "punctuation_density": "punctuation_density",
    "function_word_density": "function_word_density",
    "paragraph_density": "paragraph_density",
    "total_length_chars": "response_length_chars",
    "readability_flesch_kincaid_grade": "readability_flesch_kincaid_grade",
    "emotion_intensity_diff": "emotion_intensity",
    "emphasis_density": "emphasis_density",
    "list_density": "list_density",
    "avg_concreteness": "avg_concreteness",
    "word_freq_avg_zipf_frequency": "word_freq.avg_zipf_frequency",
}
DISTRIBUTION_METRICS = {
    "punctuation": "punctuation",
    "pos_dist": "pos_dist",
    "dep_dist": "dep_dist",
    "function_words_js": "function_words.profile",
    "emotion_profile_js": "emotion_profile",
}
BINARY_METRICS = {
    "header_use": "header_use",
    "list_use": "list_use",
    "markdown_emphasis_use": "markdown_emphasis_use",
    "form_opening_template": "form_label.opening_template",
    "form_closing_template": "form_label.closing_template",
}
DIFF_METRICS = set(SCALAR_METRICS) | set(DISTRIBUTION_METRICS)


def emotion_to_dict(em: Mapping[str, Any] | list[dict[str, Any]] | None) -> dict[str, float]:
    if not em:
        return {}
    if isinstance(em, Mapping):
        return {str(label): float(score) for label, score in em.items()}
    out: dict[str, float] = {}
    for x in em:
        lab = x.get("label")
        if lab is None:
            continue
        out[str(lab)] = float(x.get("score", 0.0) or 0.0)
    return out


def punctuation_rate_dict(r: dict) -> dict[str, float]:
    p = r.get("punctuation") or {}
    total = sum(float(v or 0) for v in p.values())
    if total > 0:
        return {k: float(v or 0) / total for k, v in p.items()}
    return {}


def emotion_intensity(em: list[dict[str, Any]] | None) -> float | None:
    scores = [
        float(x.get("score", 0.0) or 0.0)
        for x in (em or [])
        if str(x.get("label", "")).lower() != "neutral"
    ]
    if not scores:
        return None
    return max(scores)


def response_emotion_intensity(response: dict) -> float | None:
    value = response.get("emotion_intensity")
    if value is not None:
        return float(value)
    return emotion_intensity(
        response.get("emotion_raw_scores") or response.get("emotion_go_emotions")
    )


def feature_value(features, path):
    value = features
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def metric_inputs(features):
    return dict(
        features,
        punctuation=punctuation_rate_dict(features),
        emotion_profile=emotion_to_dict(
            features.get("emotion_profile") or features.get("emotion_raw_scores") or features.get("emotion_go_emotions")
        ),
        emotion_intensity=response_emotion_intensity(features),
    )


def linguistic_metrics(r1: dict, r2: dict) -> dict[str, float | None]:
    a, b = metric_inputs(r1), metric_inputs(r2)
    out = {
        name: abs_diff(feature_value(a, path), feature_value(b, path))
        for name, path in SCALAR_METRICS.items()
    }
    for name, path in DISTRIBUTION_METRICS.items():
        distribution_a = feature_value(a, path) or {}
        distribution_b = feature_value(b, path) or {}
        out[name] = (
            js_divergence(distribution_a, distribution_b)
            if has_positive_mass(distribution_a) and has_positive_mass(distribution_b)
            else None
        )
    for name, path in BINARY_METRICS.items():
        value_a, value_b = feature_value(a, path), feature_value(b, path)
        out[name] = float(value_a == value_b) if isinstance(value_a, bool) and isinstance(value_b, bool) else None
    return out


def linguistic_similarities(
    r1: dict,
    r2: dict,
) -> dict[str, float | None]:
    raw = linguistic_metrics(r1, r2)
    out: dict[str, float | None] = {}
    for key, value in raw.items():
        if value is None:
            out[key] = None
            continue
        number = float(value)
        out[key] = -number if key in DIFF_METRICS else number
        if not math.isfinite(number):
            out[key] = None
    return out


# 4. BLEU: mean of both sentence-score directions, scaled to [0, 1].
@lru_cache(maxsize=1)
def _bleu_metric():
    return BLEU(
        lowercase=True,
        tokenize="13a",
        smooth_method="exp",
        max_ngram_order=4,
        effective_order=True,
    )


def symmetric_bleu(text_a: str, text_b: str) -> float | None:
    if not text_a.strip() or not text_b.strip():
        return None
    bleu = _bleu_metric()
    ab = bleu.sentence_score(text_a, [text_b]).score / 100.0
    ba = bleu.sentence_score(text_b, [text_a]).score / 100.0
    return float((ab + ba) / 2.0)


# 5. ROUGE-L: exact longest-common-subsequence F1 with the default ROUGE tokenizer.
@lru_cache(maxsize=1)
def _rouge_tokenizer():
    return DefaultTokenizer(use_stemmer=False)


@lru_cache(maxsize=512)
def _rouge_features(text: str) -> tuple[tuple[str, ...], dict[str, int]]:
    tokens = tuple(_rouge_tokenizer().tokenize(text))
    masks: dict[str, int] = {}
    for index, token in enumerate(tokens):
        masks[token] = masks.get(token, 0) | (1 << index)
    return tokens, masks


def rouge_l_f1(text_a: str, text_b: str) -> float | None:
    if not text_a.strip() or not text_b.strip():
        return None
    tokens_a, masks_a = _rouge_features(text_a)
    tokens_b, masks_b = _rouge_features(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    tokens, masks = (tokens_a, masks_b) if len(tokens_b) <= len(tokens_a) else (tokens_b, masks_a)
    state = 0
    for token in tokens:
        matches = masks.get(token, 0) | state
        state = matches & ~(matches - ((state << 1) | 1))
    lcs = state.bit_count()
    precision, recall = lcs / len(tokens_b), lcs / len(tokens_a)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


# 6–7. Embedding metrics: response cosine and sentence InfoMatch.
# Embedding model: qwen-3-8-embedding, supplied through the embed callable.
EMBEDDING_METRICS = ("embedding_cosine",)
INFOMATCH_METRICS = ("precision", "recall", "f1")


def normalize_embs(embs: np.ndarray) -> np.ndarray:
    if embs.shape[0] == 0:
        return embs.astype(np.float32, copy=False)
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)


def embedding_similarities(texts, views, embed, nlp):
    sentences_a, sentences_b = batch_sentences_from_blocks(
        [view["semantic_blocks"] for view in views], nlp,
        [view["semantic_block_is_table"] for view in views],
    )
    inputs = texts + sentences_a + sentences_b
    vectors = np.asarray(embed(inputs), dtype=np.float32)

    # 6. Response cosine: dot product divided by the product of raw-response vector norms.
    a, b = vectors[:2]
    cosine = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10), -1.0, 1.0))

    # 7. Sentence InfoMatch: mean best-match sentence cosine in each direction, then harmonic mean.
    # Tables are excluded; either response having no prose sentences gives undefined InfoMatch.
    infomatch = dict.fromkeys(INFOMATCH_METRICS)
    if sentences_a and sentences_b:
        split = 2 + len(sentences_a)
        similarities = np.clip(normalize_embs(vectors[2:split]) @ normalize_embs(vectors[split:]).T, -1.0, 1.0)
        precision = float(similarities.max(axis=1).mean())
        recall = float(similarities.max(axis=0).mean())
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        infomatch = dict(precision=precision, recall=recall, f1=f1)
    return dict(embedding_cosine=cosine, infomatch=infomatch)


# Combine metric groups for a pair of annotated responses.
LEXICAL_METRICS = ("bleu", "rouge_l")
SEMANTIC_METRICS = LEXICAL_METRICS + EMBEDDING_METRICS + ("infomatch",)


def prepare_response(response):
    text = response.get("response")
    linguistic = response.get("response_linguistic_label")
    features = dict(linguistic or {}) if response.get("response_linguistic_label_errors") is None else {}
    form = response.get("response_form_label")
    features["form_label"] = form if response.get("response_form_label_errors") is None else None
    return text, features


def calculate_similarity(response_a, response_b, embed=None, nlp=None):
    text_a, features_a = prepare_response(response_a)
    text_b, features_b = prepare_response(response_b)
    metrics = linguistic_similarities(features_a, features_b)
    parser = _markdown_parser()
    texts = [text_a, text_b]
    views = [prepare_text_views(text, parser) for text in texts]
    semantic_a, semantic_b = [view["semantic_text"] for view in views]
    metrics.update(dict.fromkeys(SEMANTIC_METRICS))
    metrics["bleu"] = symmetric_bleu(semantic_a, semantic_b)
    metrics["rouge_l"] = rouge_l_f1(semantic_a, semantic_b)
    metrics["infomatch"] = dict.fromkeys(INFOMATCH_METRICS)
    if embed is not None:
        metrics.update(embedding_similarities(texts, views, embed, nlp if nlp is not None else _load_spacy()))
    return metrics
