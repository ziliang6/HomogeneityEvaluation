# Input JSONL: one string response per row, e.g. {"response": "A green leaf."}.
# Output preserves each row and adds response_linguistic_label and its null error field.

# Requirements
# numpy==2.2.6
# spacy==3.8.16
# markdown-it-py==4.2.0
# lexicalrichness==0.5.1
# textstat==0.7.13
# wordfreq==3.1.1
# scikit-learn==1.7.2
# torch==2.11.0
# transformers==5.6.0

import json
import math
import pickle
import re
from collections import Counter, OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import spacy
import torch
from lexicalrichness import LexicalRichness
from markdown_it import MarkdownIt
from textstat import textstat
from transformers import pipeline
from wordfreq import zipf_frequency


SPACY_BATCH_SIZE = 64
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+")


@lru_cache(maxsize=1)
def _markdown_parser():
    return MarkdownIt("commonmark").enable("table")


def _without_urls(text: str) -> str:
    def remove(match: re.Match[str]) -> str:
        url, tail = match.group(), ""
        while url and (
            url[-1] in ".,;:!?'\"”’"
            or (url[-1] == ")" and url.count(")") > url.count("("))
        ):
            tail, url = url[-1] + tail, url[:-1]
        return tail

    return _URL_RE.sub(remove, text)


def _inline_plain_text(children, *, keep_inline_code: bool) -> str:
    pieces: list[str] = []
    for child in children or []:
        if child.type in {"text", "image"}:
            pieces.append(child.content)
        elif child.type == "code_inline":
            pieces.append(child.content if keep_inline_code else " ")
        elif child.type in {"softbreak", "hardbreak"}:
            pieces.append("\n")
    cleaned = _without_urls("".join(pieces))
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return re.sub(r" +([,.;:!?])", r"\1", cleaned)


def _split_block(text: str) -> list[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []
    if any(_EXTRA_BULLET_PREFIX.match(line) for line in lines):
        return [
            block for block in (
                _EXTRA_BULLET_PREFIX.sub("", line).strip() for line in lines
            ) if block
        ]
    return [" ".join(lines)]


def prepare_text_views(raw_text: str, markdown_parser, tokens=None) -> dict[str, Any]:
    if tokens is None:
        tokens = markdown_parser.parse(raw_text or "")
    semantic_parts: list[str] = []
    prose_parts: list[str] = []
    semantic_blocks: list[str] = []
    semantic_block_is_table: list[bool] = []
    prose_blocks: list[str] = []
    previous_end: int | None = None
    in_table = False
    row_cells: list[str] = []

    def _append(parts: list[str], text: str, separator: str) -> None:
        if parts:
            parts.append(separator)
        parts.append(text)

    for token in tokens:
        if token.type == "table_open":
            in_table = True
            continue
        if token.type == "table_close":
            in_table = False
            continue
        if token.type == "tr_open":
            row_cells = []
            continue
        if token.type == "tr_close":
            cells = [cell for cell in row_cells if cell]
            if cells:
                _append(semantic_parts, " ".join(cells), "\n")
                semantic_blocks.append(" | ".join(cells))
                semantic_block_is_table.append(True)
            continue
        if token.type != "inline":
            continue
        semantic = _split_block(_inline_plain_text(token.children, keep_inline_code=True))
        if in_table:
            row_cells.append(" ".join(semantic))
            continue
        prose = _split_block(_inline_plain_text(token.children, keep_inline_code=False))
        if not semantic and not prose:
            continue
        start, end = token.map or (None, None)
        separator = "\n\n" if start is not None and previous_end is not None and start > previous_end else "\n"
        if semantic:
            _append(semantic_parts, "\n".join(semantic), separator)
            semantic_blocks.extend(semantic)
            semantic_block_is_table.extend([False] * len(semantic))
        if prose:
            _append(prose_parts, "\n".join(prose), separator)
            prose_blocks.extend(prose)
        if end is not None:
            previous_end = end
    return {
        "semantic_text": "".join(semantic_parts),
        "prose_text": "".join(prose_parts),
        "semantic_blocks": semantic_blocks,
        "semantic_block_is_table": semantic_block_is_table,
        "prose_blocks": prose_blocks,
    }


def _load_spacy():
    spacy.require_cpu()
    return spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])


def _spacy_sentences(doc) -> list[str]:
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def batch_sentences_from_blocks(
    block_lists: list[list[str]], nlp,
    table_flags_list: list[list[bool]] | None = None,
) -> list[list[str]]:
    sentences: list[list[str]] = [[] for _ in block_lists]
    if table_flags_list is None:
        table_flags_list = [[False] * len(blocks) for blocks in block_lists]
    entries = [
        (owner, block, is_table)
        for owner, (blocks, flags) in enumerate(zip(block_lists, table_flags_list))
        for block, is_table in zip(blocks, flags)
    ]
    docs = iter(nlp.pipe(
        (block for _, block, is_table in entries if not is_table),
        batch_size=SPACY_BATCH_SIZE,
    ))
    for owner, block, is_table in entries:
        if not is_table:
            sentences[owner].extend(_spacy_sentences(next(docs)))
    return sentences


def iter_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            yield record


def response_texts(record: dict[str, Any]) -> list[str]:
    responses = record.get("response")
    if isinstance(responses, str):
        return [responses]
    return responses


def record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "response"}


def density_per_100_words(count: int, n_words: int) -> float | None:
    return 100.0 * count / n_words if n_words else None


# 2. MTLD: tokenize_lexical + LexicalRichness.mtld; fewer than 50 words gives null.
MIN_TOKENS_MTLD = 50
LEXICAL_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


@lru_cache(maxsize=128)
def _lexical_tokens(text: str) -> tuple[str, ...]:
    return tuple(m.group(0).lower() for m in LEXICAL_RE.finditer(text))


def tokenize_lexical(text: str) -> list[str]:
    return list(_lexical_tokens(text or ""))


def mtld_score(text: str) -> float | None:
    tokens = tokenize_lexical(text)
    if len(tokens) < MIN_TOKENS_MTLD:
        return None

    score = float(LexicalRichness(tokens, preprocessor=None, tokenizer=None).mtld())
    return score if math.isfinite(score) else None


# 3. Punctuation: regex counts; punctuation_profile divides by the total count.
# Densities use density_per_100_words: 100 * count / prose word count.
PUNCT_CLASSES = (
    "comma", "period", "dash", "hyphen", "semicolon", "colon",
    "exclamation", "question", "ellipsis", "paren", "quote",
)
_ELLIPSIS_RE = re.compile(r"\.{2,}|…")
_DOUBLE_QUOTE_RE = re.compile(r'["“”]')
_CURLY_SINGLE_PAIR_RE = re.compile(r"‘[^‘’\n]+’")
_ASCII_SINGLE_PAIR_RE = re.compile(r"(?<!\w)'[^'\n]+'(?!\w)")
_PUNCT_RES: dict[str, re.Pattern] = {
    "comma": re.compile(r"(?<!\d),|(?<=\d),(?!\d)"),
    "period": re.compile(r"\.(?!\w)"),
    "dash": re.compile(r"—|(?<!\d)–|(?<=\d)–(?!\d)|(?<!-)--(?!-)|(?<=\s)-(?=\s)"),
    "hyphen": re.compile(r"(?<=[^\W\d_])-(?=\w)|(?<=\d)-(?=[^\W\d_])"),
    "semicolon": re.compile(r";"),
    "colon": re.compile(r"(?<!\d):|(?<=\d):(?!\d)"),
    "exclamation": re.compile(r"!"),
    "question": re.compile(r"\?"),
    "paren": re.compile(r"[()]"),
}


def count_quotation_marks(text: str) -> int:
    double_count = len(_DOUBLE_QUOTE_RE.findall(text))
    curly_single_pairs = len(_CURLY_SINGLE_PAIR_RE.findall(text))
    ascii_single_pairs = len(_ASCII_SINGLE_PAIR_RE.findall(text))
    return double_count + 2 * curly_single_pairs + 2 * ascii_single_pairs


def punctuation_frequencies(text: str) -> dict[str, int] | None:
    s = text or ""
    out = {k: 0 for k in PUNCT_CLASSES}
    out["quote"] = count_quotation_marks(s)
    out["ellipsis"] = len(_ELLIPSIS_RE.findall(s))
    s = _ELLIPSIS_RE.sub(" ", s)
    for name, pat in _PUNCT_RES.items():
        out[name] = len(pat.findall(s))
    return out if sum(out.values()) else None


def punctuation_profile(counts: dict[str, int] | None) -> dict[str, float] | None:
    if not counts:
        return None
    total = sum(counts.values())
    return {key: count / total for key, count in counts.items()} if total else None


# 4. Syntax: spaCy nlp.pipe parses prose blocks; ignore whitespace and punctuation tokens.
# _normalize converts POS and dependency counts into probability distributions.
def _normalize(c: Counter) -> dict[str, float] | None:
    total = sum(c.values())
    return {k: v / total for k, v in sorted(c.items())} if total else None


def analyze_blocks(block_lists: list[list[str]], nlp) -> list[dict[str, Any]]:
    acc = [
        {"pos": Counter(), "dep": Counter(), "fw": Counter(),
         "n_tokens": 0, "sents": []}
        for _ in block_lists
    ]
    blocks = [block for block_list in block_lists for block in block_list]
    owners = [i for i, block_list in enumerate(block_lists) for _ in block_list]
    for owner, doc in zip(owners, nlp.pipe(blocks, batch_size=SPACY_BATCH_SIZE)):
        stats = acc[owner]
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            stats["n_tokens"] += 1
            stats["pos"][t.pos_] += 1
            stats["dep"][t.dep_] += 1
            cls = _function_word_class(t)
            if cls:
                stats["fw"][cls] += 1
        stats["sents"].extend(_spacy_sentences(doc))
    return acc


# 5. Function words: _function_word_class uses spaCy POS and article lookup.
# Report class counts, within-function-word proportions, and density per 100 words.
FUNCTION_WORD_CLASSES = (
    "article", "preposition", "cconj", "sconj", "pronoun", "auxiliary", "particle",
)


def _function_word_class(t) -> str | None:
    if t.lower_ in {"a", "an", "the"}:
        return "article"
    return {
        "ADP": "preposition", "CCONJ": "cconj", "SCONJ": "sconj",
        "PRON": "pronoun", "AUX": "auxiliary", "PART": "particle",
    }.get(t.pos_)


def function_word_stats_from_counts(
    fw: Counter, n_tokens: int, n_words: int
) -> dict[str, Any] | None:
    if n_words == 0:
        return None
    counts = {k: fw.get(k, 0) for k in FUNCTION_WORD_CLASSES}
    n_fw = sum(counts.values())
    return {
        "count": n_fw,
        "density": density_per_100_words(n_fw, n_words),
        "profile": {k: counts[k] / n_fw for k in FUNCTION_WORD_CLASSES} if n_fw else None,
        "counts_by_class": counts,
        "n_tokens": n_tokens,
    }


# 6. Markdown: markdown-it tokens identify headings, tables, lists, and emphasis.
# Count non-code list items and outermost emphasis spans; scale densities per 100 prose words.
_EXTRA_BULLET_PREFIX = re.compile(r"(?m)^[ \t]*[•▪◦●·‣][ \t]+")


def markdown_structure_stats(raw_text: str, n_words: int, tokens) -> dict[str, Any]:
    header_use = any(t.type == "heading_open" for t in tokens)
    has_table = any(t.type == "table_open" for t in tokens)
    standard_list_items = sum(t.type == "list_item_open" for t in tokens)
    code_lines: set[int] = set()
    for token in tokens:
        if token.type in {"fence", "code_block"} and token.map:
            code_lines.update(range(*token.map))
    non_code_raw = "".join(
        "\n" if i in code_lines else line
        for i, line in enumerate((raw_text or "").splitlines(keepends=True))
    )
    extra_list_items = len(_EXTRA_BULLET_PREFIX.findall(non_code_raw))
    num_list_items = standard_list_items + extra_list_items
    list_use = num_list_items > 0
    emphasis_count = 0
    for token in tokens:
        if token.type != "inline":
            continue
        depth = 0
        for child in token.children or []:
            if child.type in {"strong_open", "em_open"}:
                if depth == 0:
                    emphasis_count += 1
                depth += 1
            elif child.type in {"strong_close", "em_close"}:
                depth -= 1
    return {
        "header_use": header_use,
        "has_table": has_table,
        "list_density": density_per_100_words(num_list_items, n_words),
        "list_use": list_use,
        "markdown_emphasis_use": emphasis_count > 0,
        "emphasis_density": density_per_100_words(emphasis_count, n_words),
        "emphasis": {
            "emphasis_count": emphasis_count,
        },
    }


# 7. Paragraphs: structural_view collapses code blocks and removes horizontal rules.
# Split on blank lines; retain character lengths and paragraph count per 100 prose words.
def structural_view(raw_text: str, tokens, code_as_paragraph: bool = True) -> str:
    lines = (raw_text or "").splitlines(keepends=True)
    drop: set[int] = set()
    placeholder: dict[int, str] = {}
    for token in tokens:
        if not token.map:
            continue
        start, end = token.map
        if token.type in {"fence", "code_block"}:
            drop.update(range(start, end))
            if code_as_paragraph:
                placeholder[start] = "[CODE]\n"
        elif token.type == "hr":
            drop.update(range(start, end))
    return "".join(
        placeholder.get(i, "" if i in drop else line)
        for i, line in enumerate(lines)
    )


def paragraph_stats(text: str, n_words: int) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    parts = re.split(r"\n\s*\n", raw)
    lengths = [len(p.strip()) for p in parts if p.strip()]
    return {
        "paragraph_density": density_per_100_words(len(lengths), n_words),
        "paragraph_lengths_chars": lengths,
    }


# 8. Readability: tokenize_lexical, spaCy sentence count, and textstat.syllable_count.
# Flesch-Kincaid grade = 0.39 * words/sentences + 11.8 * syllables/words - 15.59.
@lru_cache(maxsize=65536)
def _count_syllables_english(word: str) -> int:
    return textstat.syllable_count(word)


def readability_flesch_kincaid_grade(text: str, n_sentences: int) -> float | None:
    words = tokenize_lexical(text)
    if not words or not n_sentences:
        return None
    syllables = sum(_count_syllables_english(w) for w in words)
    return float(
        0.39 * (len(words) / n_sentences)
        + 11.8 * (syllables / len(words))
        - 15.59
    )


# 9. Word frequency: wordfreq.zipf_frequency(word, "en"), averaged over every token.
# OOV ratio is the proportion of tokens with zero Zipf frequency.
def word_freq_stats(text: str) -> dict[str, float] | None:
    words = tokenize_lexical(text)
    if not words:
        return None
    zipfs = [zipf_frequency(w, "en") for w in words]
    return {
        "avg_zipf_frequency": float(np.mean(zipfs)),
        "oov_ratio": sum(value == 0 for value in zipfs) / len(zipfs),
    }


# 10. Concreteness: FastText vectors -> regressor.predict -> clip to [1, 5], round to four decimals.
# The regressor is trained from Brysbaert et al.'s word-concreteness ratings (2014),
# using FastText word representations (Bojanowski et al., 2017).
# Constrain predictions to the original 1–5 scale and average lexical words with FastText vectors.
# ConcretnessPredictorLight.predict marks missing vectors as -1; average valid token predictions.
class _ConcretnessPredictor:
    def load_embeddings(self, embedding_file: str) -> dict[str, np.ndarray]:
        embeddings: dict[str, np.ndarray] = {}
        with open(embedding_file, "r", encoding="utf-8") as f:
            f.readline()  # Skip the FastText word-count and dimension header.
            for line in f:
                word, _, values = line.rstrip("\r\n").partition(" ")
                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                embeddings[word] = vector
        return embeddings


    def prepare_features(
        self, words: list[str], embeddings: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, list[int]]:
        features: list[np.ndarray] = []
        valid_indices: list[int] = []
        for i, word in enumerate(words):
            vec = embeddings.get(word.lower())
            if vec is not None:
                features.append(vec)
                valid_indices.append(i)
        if features:
            return np.array(features, dtype="float32"), valid_indices
        dim = next(iter(embeddings.values())).shape[0] if embeddings else 300
        return np.zeros((0, dim), dtype="float32"), valid_indices


class ConcretnessPredictorLight:
    def __init__(self, model_path: str, embeddings_path: str) -> None:
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self._base = _ConcretnessPredictor()
        self.embeddings = self._base.load_embeddings(embeddings_path)
        self._prediction_cache: OrderedDict[str, float] = OrderedDict()


    def predict(self, words: list[str]) -> list[float]:
        keys = [word.lower() for word in words]
        missing = list(dict.fromkeys(word for word in keys if word not in self._prediction_cache))
        features, valid_indices = self._base.prepare_features(missing, self.embeddings)
        predictions = dict.fromkeys(missing, -1.0)
        if len(features) > 0:
            preds = np.clip(self.model.predict(features), 1.0, 5.0)
            for idx, p in zip(valid_indices, preds):
                predictions[missing[idx]] = float("%.4f" % p)
        self._prediction_cache.update(predictions)

        result = [self._prediction_cache[word] for word in keys]
        for word in keys:
            self._prediction_cache.move_to_end(word)
        while len(self._prediction_cache) > 65536:
            self._prediction_cache.popitem(last=False)
        return result


def load_concreteness_predictor(
    model_path, embeddings_path,
) -> ConcretnessPredictorLight:
    print(f"Loading concreteness model from {model_path} ...")
    print(f"Loading concreteness embeddings from {embeddings_path} ...")
    return ConcretnessPredictorLight(
        model_path=model_path,
        embeddings_path=embeddings_path,
    )


def avg_concreteness(text: str, predictor: ConcretnessPredictorLight | None) -> float | None:
    if predictor is None:
        return None
    words = tokenize_lexical(text)
    if not words:
        return None
    preds = predictor.predict(words)
    valid = [p for p in preds if p != -1.0]
    return float(np.mean(valid)) if valid else None


# 11. Emotion: build_emotion_chunks packs prose sentences into <=510 subword chunks.
# The GoEmotions classifier scores each chunk; average scores weighted by chunk subword count.
# Normalize scores across all labels; intensity is the largest raw non-neutral score.
EMOTION_BATCH_SIZE = 8


def load_emotion_classifier(model="SamLowe/roberta-base-go_emotions"):
    return pipeline(
        task="text-classification", model=model, top_k=None,
        device=0 if torch.cuda.is_available() else -1,
    )


def build_emotion_chunks(
    sentences: list[str], tokenizer, max_subwords: int = 510,
) -> list[tuple[str, int]]:
    chunks: list[tuple[str, int]] = []
    current = ""

    for sentence in sentences:
        sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
        if not sentence_ids:
            continue
        candidate = f"{current} {sentence}" if current else sentence
        candidate_length = len(tokenizer.encode(candidate, add_special_tokens=False))
        if candidate_length <= max_subwords:
            current = candidate
            continue

        if current:
            chunks.append((current, len(tokenizer.encode(current, add_special_tokens=False))))
            current = ""

        if len(sentence_ids) <= max_subwords:
            current = sentence
            continue

        start = 0
        while start < len(sentence_ids):
            end = min(start + max_subwords, len(sentence_ids))
            piece = tokenizer.decode(
                sentence_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            while end > start + 1 and len(tokenizer.encode(piece, add_special_tokens=False)) > max_subwords:
                end -= 1
                piece = tokenizer.decode(
                    sentence_ids[start:end],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
            if piece:
                chunks.append((piece, len(tokenizer.encode(piece, add_special_tokens=False))))
            start = end

    if current:
        chunks.append((current, len(tokenizer.encode(current, add_special_tokens=False))))
    return chunks


def batch_go_emotions(
    sentence_lists: list[list[str]], classifier,
) -> list[list[dict[str, Any]] | None]:
    if not sentence_lists:
        return []
    tokenizer = classifier.tokenizer
    chunk_texts: list[str] = []
    chunk_weights: list[int] = []
    offsets: list[tuple[int, int]] = []
    for sentences in sentence_lists:
        start = len(chunk_texts)
        for chunk, weight in build_emotion_chunks(sentences, tokenizer):
            chunk_texts.append(chunk)
            chunk_weights.append(weight)
        offsets.append((start, len(chunk_texts)))

    if not chunk_texts:
        return [None for _ in sentence_lists]
    raw_batch = classifier(chunk_texts, truncation=False, batch_size=EMOTION_BATCH_SIZE)

    out: list[list[dict[str, Any]] | None] = []
    for start, end in offsets:
        if start == end:
            out.append(None)
            continue
        sums: dict[str, float] = {}
        label_order: list[str] = []
        total_weight = sum(chunk_weights[start:end])
        for raw, weight in zip(raw_batch[start:end], chunk_weights[start:end]):
            for item in raw:
                label = item["label"]
                if label not in sums:
                    sums[label] = 0.0
                    label_order.append(label)
                sums[label] += float(item["score"]) * weight
        out.append([
            {"label": label, "score": sums[label] / total_weight}
            for label in label_order
        ])
    return out


def emotion_metrics_from_raw_scores(
    raw_scores: list[dict[str, Any]] | None,
) -> tuple[dict[str, float] | None, float | None]:
    if raw_scores is None:
        return None, None
    scores = {item["label"]: float(item["score"]) for item in raw_scores}
    total = sum(scores.values())
    profile = (
        {label: score / total for label, score in scores.items()}
        if total > 0 else None
    )
    non_neutral = [score for label, score in scores.items() if label != "neutral"]
    intensity = max(non_neutral) if non_neutral else None
    return profile, intensity


def annotate_one_response(
    raw_text: str,
    semantic_text: str,
    prose_text: str,
    structural_text: str,
    spacy_stats: dict[str, Any],
    markdown_stats: dict[str, Any],
    emotion_raw_scores: list[dict[str, Any]] | None = None,
    concreteness_predictor: ConcretnessPredictorLight | None = None,
) -> dict[str, Any]:
    n_words = len(tokenize_lexical(prose_text))
    features = {
        "raw_text": raw_text,
        "semantic_text": semantic_text,
        "prose_text": prose_text,
    }

    # 1. Response length: Python len counts characters in the input response.
    features["response_length_chars"] = len(raw_text)

    # 2. Lexical diversity: mtld_score uses LexicalRichness.mtld on lowercase lexical tokens.
    features["mtld"] = mtld_score(prose_text)

    # 3. Punctuation: regex counts, normalized proportions, and occurrences per 100 prose words.
    counts = punctuation_frequencies(prose_text)
    features["punctuation"] = counts
    features["punctuation_profile"] = punctuation_profile(counts)
    features["punctuation_density"] = density_per_100_words(sum((counts or {}).values()), n_words)

    # 4. Syntax: normalize spaCy POS and dependency counts from analyze_blocks.
    features["pos_dist"] = _normalize(spacy_stats["pos"])
    features["dep_dist"] = _normalize(spacy_stats["dep"])

    # 5. Function words: class counts, proportions among all function words, and density per 100 prose words.
    function_words = function_word_stats_from_counts(spacy_stats["fw"], spacy_stats["n_tokens"], n_words)
    features["function_words"] = function_words
    features["function_word_density"] = function_words["density"] if function_words else None

    # 6. Markdown: flags and list/emphasis densities from markdown_structure_stats.
    features.update(markdown_stats)

    # 7. Paragraphs: blank-line-separated structural blocks, lengths, and density.
    paragraphs = paragraph_stats(structural_text, n_words)
    features["paragraphs"] = paragraphs
    features["paragraph_density"] = paragraphs["paragraph_density"] if paragraphs else None

    # 8. Readability: Flesch-Kincaid grade from word, sentence, and syllable counts.
    features["readability_flesch_kincaid_grade"] = readability_flesch_kincaid_grade(
        prose_text, n_sentences=len(spacy_stats["sents"]),
    )

    # 9. Word frequency: mean English Zipf frequency and zero-frequency token proportion.
    features["word_freq"] = word_freq_stats(prose_text)

    # 10. Concreteness: mean valid FastText-based regression predictions.
    features["avg_concreteness"] = avg_concreteness(prose_text, concreteness_predictor)

    # 11. Emotion: weighted GoEmotions scores, normalized profile, and max non-neutral intensity.
    profile, intensity = emotion_metrics_from_raw_scores(emotion_raw_scores)
    features["emotion_raw_scores"] = emotion_raw_scores
    features["emotion_profile"] = profile
    features["emotion_intensity"] = intensity
    return features


def annotate_response(response: str, nlp, classifier=None, predictor=None) -> dict[str, Any]:

    # Preprocessing: MarkdownIt parses structure; prepare_text_views removes URLs and block code.
    # Prose also excludes tables and inline code; structural_view preserves paragraph boundaries.
    parser = _markdown_parser()
    tokens = parser.parse(response)
    view = prepare_text_views(response, parser, tokens)
    stats = analyze_blocks([view["prose_blocks"]], nlp)[0]
    emotions = batch_go_emotions([stats["sents"]], classifier)[0] if classifier is not None else None
    return annotate_one_response(
        response, view["semantic_text"], view["prose_text"], structural_view(response, tokens),
        stats, markdown_structure_stats(response, len(tokenize_lexical(view["prose_text"])), tokens),
        emotion_raw_scores=emotions, concreteness_predictor=predictor,
    )


def annotate_file(input_file, output_file, nlp=None, classifier=None, predictor=None) -> int:
    input_path, output_path = Path(input_file), Path(output_file)
    nlp = nlp if nlp is not None else _load_spacy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as target:
        for count, record in enumerate(iter_records(input_path), 1):
            response = record.get("response")
            row = dict(record)
            row["response_linguistic_label"] = annotate_response(response, nlp, classifier, predictor)
            row["response_linguistic_label_errors"] = None
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return count
