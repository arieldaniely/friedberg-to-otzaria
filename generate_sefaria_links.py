from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


SEFARIA_API_BASE = "https://www.sefaria.org/api/v3/texts"
FRIEDBERG_FILE_PREFIX = "הכי גרסינן "
FRIEDBERG_FILE_SUFFIX = ".txt"
MAX_GROUP_SIZE = 3
MIN_GROUP_SIMILARITY = 0.52
SKIP_COST = 0.52
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF = 1.5
VILNA_WITNESS_PREFIX = "<b>וילנא</b>:"
MIN_STABLE_TOKEN_COUNT = 3
MIN_STABLE_LETTERS = 12

HEADING_RE = re.compile(r"<h2>דף (?P<label>[^<]+)</h2>")
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
HEBREW_PUNCT_RE = re.compile(r"[^0-9A-Za-z\u0590-\u05FF\s]")
HEBREW_DIACRITICS_RE = re.compile(r"[\u0591-\u05C7]")

HEBREW_LETTER_VALUES = {
    "א": 1,
    "ב": 2,
    "ג": 3,
    "ד": 4,
    "ה": 5,
    "ו": 6,
    "ז": 7,
    "ח": 8,
    "ט": 9,
    "י": 10,
    "כ": 20,
    "ך": 20,
    "ל": 30,
    "מ": 40,
    "ם": 40,
    "נ": 50,
    "ן": 50,
    "ס": 60,
    "ע": 70,
    "פ": 80,
    "ף": 80,
    "צ": 90,
    "ץ": 90,
    "ק": 100,
    "ר": 200,
    "ש": 300,
    "ת": 400,
}

TRACTATE_TO_SEFARIA = {
    "ברכות": "Berakhot",
    "שבת": "Shabbat",
    "עירובין": "Eruvin",
    "פסחים": "Pesachim",
    "יומא": "Yoma",
    "סוכה": "Sukkah",
    "ביצה": "Beitzah",
    "ראש השנה": "Rosh Hashanah",
    "תענית": "Taanit",
    "מגילה": "Megillah",
    "מועד קטן": "Moed Katan",
    "חגיגה": "Chagigah",
    "יבמות": "Yevamot",
    "כתובות": "Ketubot",
    "נדרים": "Nedarim",
    "נזיר": "Nazir",
    "סוטה": "Sotah",
    "גיטין": "Gittin",
    "קידושין": "Kiddushin",
    "בבא קמא": "Bava Kamma",
    "בבא מציעא": "Bava Metzia",
    "בבא בתרא": "Bava Batra",
    "סנהדרין": "Sanhedrin",
    "מכות": "Makkot",
    "שבועות": "Shevuot",
    "עבודה זרה": "Avodah Zarah",
    "הוריות": "Horayot",
    "זבחים": "Zevachim",
    "מנחות": "Menachot",
    "חולין": "Chullin",
    "בכורות": "Bekhorot",
    "ערכין": "Arakhin",
    "תמורה": "Temurah",
    "כריתות": "Keritot",
    "מעילה": "Meilah",
    "תמיד": "Tamid",
    "נדה": "Niddah",
}


@dataclass(frozen=True)
class Segment:
    tractate_name: str
    book_name: str
    amud_label: str
    segment_index: int
    line_index: int
    he_ref: str
    text: str
    normalized_text: str


@dataclass(frozen=True)
class GroupMatch:
    left_start: int
    left_len: int
    right_start: int
    right_len: int
    similarity: float


@dataclass(frozen=True)
class AmudBlock:
    label: str
    source_ref: str
    normalized_text: str
    alignment_text: str
    segments: list[Segment]


@dataclass(frozen=True)
class ApiAmudResult:
    segments: list[str]
    version_title: str
    version_source: str
    he_title: str
    tref: str
    next_tref: str | None


def warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Otzaria-compatible tractate links from Sefaria API."
    )
    parser.add_argument(
        "--friedberg-dir",
        type=Path,
        default=Path("output"),
        help="Directory that contains the generated הכי גרסינן tractate files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "links",
        help="Directory where the generated link JSON files will be written.",
    )
    parser.add_argument(
        "--tractate",
        action="append",
        default=None,
        help="Tractate name to process. Repeat for multiple tractates. Omit to process all generated supported tractates.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Number of Sefaria API attempts per amud.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help="Initial retry delay in seconds for Sefaria API requests.",
    )
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_control_characters(text: str) -> str:
    kept: list[str] = []
    for char in text:
        if char in "\t\n\r":
            kept.append(" ")
            continue
        if unicodedata.category(char) == "Cf":
            continue
        kept.append(char)
    return "".join(kept)


def clean_text(value: str) -> str:
    return normalize_spaces(strip_control_characters(value))


def normalize_compare_text(text: str) -> str:
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = clean_text(text)
    text = HEBREW_DIACRITICS_RE.sub("", text)
    text = text.translate(
        str.maketrans(
            {
                "״": '"',
                "׳": "'",
                "–": "-",
                "—": "-",
                "־": " ",
            }
        )
    )
    text = HEBREW_PUNCT_RE.sub(" ", text)
    return normalize_spaces(text)


def tokenize(text: str) -> list[str]:
    return list(tokenize_cached(text))


@lru_cache(maxsize=32768)
def tokenize_cached(text: str) -> tuple[str, ...]:
    return tuple(text.split())


@lru_cache(maxsize=32768)
def token_counter(text: str) -> Counter[str]:
    return Counter(tokenize_cached(text))


def hebrew_to_int(text: str) -> int:
    total = 0
    for char in clean_text(text):
        if char in {'"', "״", "'", "׳"}:
            continue
        try:
            total += HEBREW_LETTER_VALUES[char]
        except KeyError as exc:
            raise ValueError(f"Unsupported Hebrew numeral: {text!r}") from exc
    if total <= 0:
        raise ValueError(f"Could not parse Hebrew numeral: {text!r}")
    return total


def int_to_hebrew(number: int) -> str:
    if number <= 0:
        raise ValueError(f"Unsupported positive integer: {number}")

    parts: list[str] = []
    hundreds = [(400, "ת"), (300, "ש"), (200, "ר"), (100, "ק")]
    tens = [
        (90, "צ"),
        (80, "פ"),
        (70, "ע"),
        (60, "ס"),
        (50, "נ"),
        (40, "מ"),
        (30, "ל"),
        (20, "כ"),
        (10, "י"),
    ]
    ones = [(9, "ט"), (8, "ח"), (7, "ז"), (6, "ו"), (5, "ה"), (4, "ד"), (3, "ג"), (2, "ב"), (1, "א")]

    remainder = number
    while remainder >= 400:
        parts.append("ת")
        remainder -= 400
    for value, letter in hundreds[1:]:
        while remainder >= value:
            parts.append(letter)
            remainder -= value

    if remainder == 15:
        parts.append("טו")
        remainder = 0
    elif remainder == 16:
        parts.append("טז")
        remainder = 0

    for value, letter in tens:
        while remainder >= value:
            parts.append(letter)
            remainder -= value
    for value, letter in ones:
        while remainder >= value:
            parts.append(letter)
            remainder -= value
    return "".join(parts)


def extract_amud_label(line: str) -> str | None:
    match = HEADING_RE.match(line)
    if not match:
        return None
    return clean_text(match.group("label"))


def parse_amud_label(amud_label: str) -> tuple[int, str]:
    if not amud_label:
        raise ValueError("Amud label cannot be empty.")

    suffix = amud_label[-1]
    if suffix not in {".", ":"}:
        raise ValueError(f"Unsupported amud suffix in {amud_label!r}")

    amud = "a" if suffix == "." else "b"
    daf_number = hebrew_to_int(amud_label[:-1])
    return daf_number, amud


def build_segment_ref(book_name: str, amud_label: str, segment_index: int) -> str:
    return f"{book_name} {amud_label}, {int_to_hebrew(segment_index)}"


def group_text(items: list[str], start: int, length: int) -> str:
    return " ".join(
        item
        for item in items[start : start + length]
        if item
    )


def token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = tokenize_cached(left)
    right_tokens = tokenize_cached(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((token_counter(left) & token_counter(right)).values())
    return (2 * overlap) / (len(left_tokens) + len(right_tokens))


def summarize_for_similarity(text: str, edge_chars: int = 240) -> str:
    if len(text) <= edge_chars * 2 + 1:
        return text
    return f"{text[:edge_chars]} {text[-edge_chars:]}"


def build_alignment_signature(text: str, edge_token_count: int = 36) -> str:
    tokens = tokenize(text)
    if len(tokens) <= edge_token_count * 2:
        return " ".join(tokens)
    return " ".join(tokens[:edge_token_count] + tokens[-edge_token_count:])


def is_stable_link_text(text: str) -> bool:
    tokens = tokenize_cached(text)
    if not tokens:
        return False
    if len(tokens) >= MIN_STABLE_TOKEN_COUNT:
        return True
    return sum(len(token) for token in tokens) >= MIN_STABLE_LETTERS


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0

    left_tokens = tokenize_cached(left)
    right_tokens = tokenize_cached(right)
    if not left_tokens or not right_tokens:
        return 0.0

    if max(len(left_tokens), len(right_tokens)) <= 2:
        return 0.4 if left == right and len(left) >= 4 else 0.0

    if max(len(left_tokens), len(right_tokens)) > 72:
        left = summarize_for_similarity(left)
        right = summarize_for_similarity(right)
        left_tokens = tokenize_cached(left)
        right_tokens = tokenize_cached(right)

    sequence_ratio = SequenceMatcher(
        None,
        left_tokens,
        right_tokens,
        autojunk=False,
    ).ratio()
    prefix_ratio = SequenceMatcher(
        None,
        left_tokens[:18],
        right_tokens[:18],
        autojunk=False,
    ).ratio()
    overlap_ratio = token_overlap_ratio(left, right)
    contains_bonus = 0.05 if len(left) >= 20 and (left in right or right in left) else 0.0

    return min(
        1.0,
        sequence_ratio * 0.55
        + prefix_ratio * 0.15
        + overlap_ratio * 0.25
        + contains_bonus,
    )


def match_cost(group_similarity: float, friedberg_len: int, sefaria_len: int) -> float:
    return (
        (1.0 - group_similarity)
        + 0.03 * (friedberg_len + sefaria_len - 2)
        + 0.03 * abs(friedberg_len - sefaria_len)
    )


def pair_group_offsets(friedberg_len: int, sefaria_len: int) -> list[tuple[int, int]]:
    if friedberg_len == sefaria_len:
        return [(index, index) for index in range(friedberg_len)]
    if friedberg_len == 1:
        return [(0, index) for index in range(sefaria_len)]
    if sefaria_len == 1:
        return [(index, 0) for index in range(friedberg_len)]

    pairs: list[tuple[int, int]] = []
    if friedberg_len < sefaria_len:
        for sefaria_index in range(sefaria_len):
            friedberg_index = round(sefaria_index * (friedberg_len - 1) / (sefaria_len - 1))
            pairs.append((friedberg_index, sefaria_index))
        return pairs

    for friedberg_index in range(friedberg_len):
        sefaria_index = round(friedberg_index * (sefaria_len - 1) / (friedberg_len - 1))
        pairs.append((friedberg_index, sefaria_index))
    return pairs


def align_sequences(
    left_texts: list[str],
    right_texts: list[str],
    max_group_size: int = MAX_GROUP_SIZE,
    min_group_similarity: float = MIN_GROUP_SIMILARITY,
) -> tuple[list[GroupMatch], list[int], list[int]]:
    left_count = len(left_texts)
    right_count = len(right_texts)
    dp: list[list[float]] = [
        [float("inf")] * (right_count + 1) for _ in range(left_count + 1)
    ]
    choice: list[list[tuple[str, int, int, float] | None]] = [
        [None] * (right_count + 1) for _ in range(left_count + 1)
    ]
    dp[left_count][right_count] = 0.0

    for left_index in range(left_count, -1, -1):
        for right_index in range(right_count, -1, -1):
            if left_index == left_count and right_index == right_count:
                continue

            best_cost = float("inf")
            best_choice: tuple[str, int, int, float] | None = None

            if left_index < left_count:
                candidate = SKIP_COST + dp[left_index + 1][right_index]
                if candidate < best_cost:
                    best_cost = candidate
                    best_choice = ("skip_left", 1, 0, 0.0)

            if right_index < right_count:
                candidate = SKIP_COST + dp[left_index][right_index + 1]
                if candidate < best_cost:
                    best_cost = candidate
                    best_choice = ("skip_right", 0, 1, 0.0)

            for left_len in range(1, max_group_size + 1):
                if left_index + left_len > left_count:
                    continue
                grouped_left = group_text(left_texts, left_index, left_len)
                for right_len in range(1, max_group_size + 1):
                    if right_index + right_len > right_count:
                        continue
                    grouped_right = group_text(right_texts, right_index, right_len)
                    group_similarity = similarity(grouped_left, grouped_right)
                    if group_similarity < min_group_similarity:
                        continue
                    candidate = match_cost(group_similarity, left_len, right_len) + dp[
                        left_index + left_len
                    ][right_index + right_len]
                    if candidate < best_cost:
                        best_cost = candidate
                        best_choice = ("match", left_len, right_len, group_similarity)

            dp[left_index][right_index] = best_cost
            choice[left_index][right_index] = best_choice

    matches: list[GroupMatch] = []
    unmatched_left: list[int] = []
    unmatched_right: list[int] = []
    left_index = 0
    right_index = 0

    while left_index < left_count or right_index < right_count:
        selected = choice[left_index][right_index]
        if selected is None:
            if left_index < left_count:
                unmatched_left.extend(range(left_index, left_count))
            if right_index < right_count:
                unmatched_right.extend(range(right_index, right_count))
            break

        operation, left_len, right_len, group_similarity = selected
        if operation == "match":
            matches.append(
                GroupMatch(
                    left_start=left_index,
                    left_len=left_len,
                    right_start=right_index,
                    right_len=right_len,
                    similarity=group_similarity,
                )
            )
            left_index += left_len
            right_index += right_len
            continue

        if operation == "skip_left":
            unmatched_left.append(left_index)
            left_index += 1
            continue

        unmatched_right.append(right_index)
        right_index += 1

    return matches, unmatched_left, unmatched_right


def ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def discover_tractates(friedberg_dir: Path) -> list[str]:
    available: set[str] = set()
    for path in friedberg_dir.glob(f"{FRIEDBERG_FILE_PREFIX}*{FRIEDBERG_FILE_SUFFIX}"):
        tractate_name = path.name[
            len(FRIEDBERG_FILE_PREFIX) : -len(FRIEDBERG_FILE_SUFFIX)
        ]
        if tractate_name in TRACTATE_TO_SEFARIA:
            available.add(tractate_name)
        else:
            warn(f"Skipping unsupported Friedberg output file: {path.name}")

    tractates = [
        tractate_name
        for tractate_name in TRACTATE_TO_SEFARIA
        if tractate_name in available
    ]
    if not tractates:
        raise FileNotFoundError(
            f"No supported Friedberg output files found in: {friedberg_dir}"
        )
    return tractates


def load_friedberg_segments(
    tractate_name: str,
    friedberg_dir: Path,
) -> tuple[Path, list[AmudBlock]]:
    output_path = friedberg_dir / f"{FRIEDBERG_FILE_PREFIX}{tractate_name}{FRIEDBERG_FILE_SUFFIX}"
    if not output_path.exists():
        raise FileNotFoundError(f"Missing Friedberg output file: {output_path}")

    segments_by_amud: dict[str, list[Segment]] = {}
    amud_order: list[str] = []
    current_amud: str | None = None
    segment_counter_by_amud: dict[str, int] = {}

    for line_index, raw_line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), start=1):
        amud_label = extract_amud_label(raw_line)
        if amud_label is not None:
            current_amud = amud_label
            amud_order.append(amud_label)
            segments_by_amud.setdefault(amud_label, [])
            continue

        if current_amud is None or VILNA_WITNESS_PREFIX not in raw_line:
            continue

        text = clean_text(raw_line.split(":", 1)[1])
        normalized_text = normalize_compare_text(text)
        segment_counter_by_amud[current_amud] = segment_counter_by_amud.get(current_amud, 0) + 1
        segment_index = segment_counter_by_amud[current_amud]
        segments_by_amud[current_amud].append(
            Segment(
                tractate_name=tractate_name,
                book_name=f"הכי גרסינן {tractate_name}",
                amud_label=current_amud,
                segment_index=segment_index,
                line_index=line_index,
                he_ref=build_segment_ref(f"הכי גרסינן {tractate_name}", current_amud, segment_index),
                text=text,
                normalized_text=normalized_text,
            )
        )

    blocks: list[AmudBlock] = []
    for amud_label in ordered_unique(amud_order):
        amud_segments = segments_by_amud.get(amud_label, [])
        if not amud_segments:
            continue
        blocks.append(
            AmudBlock(
                label=amud_label,
                source_ref=f"הכי גרסינן {tractate_name} {amud_label}",
                normalized_text=" ".join(
                    segment.normalized_text for segment in amud_segments if segment.normalized_text
                ),
                alignment_text=build_alignment_signature(
                    " ".join(
                        segment.normalized_text for segment in amud_segments if segment.normalized_text
                    )
                ),
                segments=amud_segments,
            )
        )

    return output_path, blocks


def fetch_sefaria_amud(
    tref: str,
    retry_attempts: int,
    retry_backoff: float,
) -> ApiAmudResult:
    query = urlencode({"version": "source", "return_format": "text_only"})
    url = f"{SEFARIA_API_BASE}/{quote(tref)}?{query}"
    request = Request(url, headers={"User-Agent": "otzaria-sefaria-links/1.0"})

    last_error: Exception | None = None
    for attempt in range(1, retry_attempts + 1):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
            versions = payload.get("versions")
            if not isinstance(versions, list) or not versions:
                raise ValueError(f"No versions returned for {tref}")
            version = versions[0]
            text = version.get("text")
            if not isinstance(text, list):
                raise ValueError(f"Unexpected text payload for {tref}: {type(text).__name__}")
            segments = [clean_text(item) for item in text if clean_text(item)]
            if not segments:
                raise ValueError(f"No text segments returned for {tref}")
            return ApiAmudResult(
                segments=segments,
                version_title=str(version.get("versionTitle", "")),
                version_source=str(version.get("versionSource", "")),
                he_title=str(payload.get("heIndexTitle", "")),
                tref=str(payload.get("ref", tref)),
                next_tref=str(payload.get("next")) if payload.get("next") else None,
            )
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retry_attempts:
                break
            time.sleep(retry_backoff * (2 ** (attempt - 1)))

    if last_error is None:
        raise RuntimeError(f"Unknown Sefaria fetch failure for {tref}")
    raise RuntimeError(f"Failed to fetch {tref} from Sefaria after {retry_attempts} attempts: {last_error}") from last_error


def tref_to_amud_label(tref: str) -> str:
    match = re.search(r"(?P<daf>\d+)(?P<amud>[ab])$", tref)
    if not match:
        raise ValueError(f"Could not parse amud from tref: {tref!r}")
    daf_number = int(match.group("daf"))
    suffix = "." if match.group("amud") == "a" else ":"
    return f"{int_to_hebrew(daf_number)}{suffix}"


def build_virtual_sefaria_segments(
    tractate_name: str,
    sefaria_title: str,
    retry_attempts: int,
    retry_backoff: float,
) -> tuple[list[AmudBlock], int]:
    blocks: list[AmudBlock] = []
    line_index = 1  # <h1>
    next_tref = f"{sefaria_title} 2a"
    seen_trefs: set[str] = set()

    while next_tref:
        tref = next_tref
        if tref in seen_trefs:
            raise RuntimeError(f"Sefaria next-ref loop detected for {tref}")
        seen_trefs.add(tref)
        api_result = fetch_sefaria_amud(
            tref=tref,
            retry_attempts=retry_attempts,
            retry_backoff=retry_backoff,
        )
        amud_label = tref_to_amud_label(api_result.tref)
        line_index += 1  # <h2>
        book_name = api_result.he_title or tractate_name
        amud_segments: list[Segment] = []

        for segment_index, text in enumerate(api_result.segments, start=1):
            line_index += 1
            amud_segments.append(
                Segment(
                    tractate_name=tractate_name,
                    book_name=book_name,
                    amud_label=amud_label,
                    segment_index=segment_index,
                    line_index=line_index,
                    he_ref=build_segment_ref(book_name, amud_label, segment_index),
                    text=text,
                    normalized_text=normalize_compare_text(text),
                )
            )

        blocks.append(
            AmudBlock(
                label=amud_label,
                source_ref=api_result.tref,
                normalized_text=" ".join(
                    segment.normalized_text for segment in amud_segments if segment.normalized_text
                ),
                alignment_text=build_alignment_signature(
                    " ".join(
                        segment.normalized_text for segment in amud_segments if segment.normalized_text
                    )
                ),
                segments=amud_segments,
            )
        )
        next_tref = api_result.next_tref

    return blocks, line_index


def build_link_entry(source: Segment, target: Segment) -> dict[str, Any]:
    return {
        "line_index_1": source.line_index,
        "line_index_2": target.line_index,
        "heRef_2": target.he_ref,
        "path_2": f"{target.book_name}.txt",
        "Conection Type": "commentary",
    }


def deduplicate_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique_entries: list[dict[str, Any]] = []
    for entry in entries:
        key = (
            entry["line_index_1"],
            entry["line_index_2"],
            entry["heRef_2"],
            entry["path_2"],
            entry["Conection Type"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_entries.append(entry)
    return unique_entries


def generate_for_tractate(
    tractate_name: str,
    friedberg_dir: Path,
    output_dir: Path,
    retry_attempts: int,
    retry_backoff: float,
) -> tuple[Path, Path, Path]:
    try:
        sefaria_title = TRACTATE_TO_SEFARIA[tractate_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tractate for Sefaria link generation: {tractate_name}") from exc

    friedberg_path, friedberg_blocks = load_friedberg_segments(tractate_name, friedberg_dir)
    sefaria_blocks, virtual_line_count = build_virtual_sefaria_segments(
        tractate_name=tractate_name,
        sefaria_title=sefaria_title,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
    )

    amud_matches, unmatched_friedberg_blocks, unmatched_sefaria_blocks = align_sequences(
        [block.alignment_text for block in friedberg_blocks],
        [block.alignment_text for block in sefaria_blocks],
        max_group_size=2,
        min_group_similarity=0.35,
    )

    forward_entries: list[dict[str, Any]] = []
    reverse_entries: list[dict[str, Any]] = []
    report_amudim: list[dict[str, Any]] = []
    report_unmatched_amudim: dict[str, list[dict[str, str]]] = {
        "friedberg": [
            {"label": friedberg_blocks[index].label, "ref": friedberg_blocks[index].source_ref}
            for index in unmatched_friedberg_blocks
        ],
        "sefaria": [
            {"label": sefaria_blocks[index].label, "ref": sefaria_blocks[index].source_ref}
            for index in unmatched_sefaria_blocks
        ],
    }
    total_matches = 0
    total_unmatched_friedberg = 0
    total_unmatched_sefaria = 0
    total_skipped_unstable_pairs = 0

    for amud_match in amud_matches:
        friedberg_group = friedberg_blocks[
            amud_match.left_start : amud_match.left_start + amud_match.left_len
        ]
        sefaria_group = sefaria_blocks[
            amud_match.right_start : amud_match.right_start + amud_match.right_len
        ]
        friedberg_segments = [
            segment for block in friedberg_group for segment in block.segments
        ]
        sefaria_segments = [
            segment for block in sefaria_group for segment in block.segments
        ]
        matches, unmatched_friedberg, unmatched_sefaria = align_sequences(
            [segment.normalized_text for segment in friedberg_segments],
            [segment.normalized_text for segment in sefaria_segments],
        )

        total_unmatched_friedberg += len(unmatched_friedberg)
        total_unmatched_sefaria += len(unmatched_sefaria)
        candidate_pair_count = 0
        linked_pair_count = 0
        skipped_unstable_pairs = 0

        for match in matches:
            total_matches += 1
            match_friedberg_segments = friedberg_segments[
                match.left_start : match.left_start + match.left_len
            ]
            match_sefaria_segments = sefaria_segments[
                match.right_start : match.right_start + match.right_len
            ]
            for friedberg_offset, sefaria_offset in pair_group_offsets(
                match.left_len,
                match.right_len,
            ):
                candidate_pair_count += 1
                source = match_friedberg_segments[friedberg_offset]
                target = match_sefaria_segments[sefaria_offset]
                if not is_stable_link_text(source.normalized_text) or not is_stable_link_text(
                    target.normalized_text
                ):
                    skipped_unstable_pairs += 1
                    total_skipped_unstable_pairs += 1
                    continue
                forward_entries.append(build_link_entry(source, target))
                reverse_entries.append(build_link_entry(target, source))
                linked_pair_count += 1

        report_amudim.append(
            {
                "friedbergAmudLabels": [block.label for block in friedberg_group],
                "sefariaAmudRefs": [block.source_ref for block in sefaria_group],
                "amudGroupSimilarity": amud_match.similarity,
                "friedbergSegments": len(friedberg_segments),
                "sefariaSegments": len(sefaria_segments),
                "matchGroups": len(matches),
                "candidatePairs": candidate_pair_count,
                "linkedPairs": linked_pair_count,
                "skippedUnstablePairs": skipped_unstable_pairs,
                "unmatchedFriedbergSegments": [
                    {
                        "line_index": friedberg_segments[index].line_index,
                        "heRef": friedberg_segments[index].he_ref,
                        "text": friedberg_segments[index].text,
                    }
                    for index in unmatched_friedberg
                ],
                "unmatchedSefariaSegments": [
                    {
                        "line_index": sefaria_segments[index].line_index,
                        "heRef": sefaria_segments[index].he_ref,
                        "text": sefaria_segments[index].text,
                    }
                    for index in unmatched_sefaria
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    forward_entries = deduplicate_entries(
        sorted(forward_entries, key=lambda item: (item["line_index_1"], item["line_index_2"], item["heRef_2"]))
    )
    reverse_entries = deduplicate_entries(
        sorted(reverse_entries, key=lambda item: (item["line_index_1"], item["line_index_2"], item["heRef_2"]))
    )

    forward_path = output_dir / f"הכי גרסינן {tractate_name}_links.json"
    reverse_path = output_dir / f"{tractate_name}_links.json"
    report_path = reports_dir / f"{tractate_name}_alignment_report.json"

    forward_path.write_text(
        json.dumps(forward_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reverse_path.write_text(
        json.dumps(reverse_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "tractate": tractate_name,
        "friedbergFile": str(friedberg_path),
        "sefariaBook": sefaria_title,
        "lineIndexModel": {
            "friedberg": "1-based line number of the Vilna witness line inside הכי גרסינן output",
            "sefaria": "1-based virtual line number assuming <h1>, then <h2> per amud, then one line per Sefaria segment",
            "virtualLineCount": virtual_line_count,
        },
        "outputFiles": {
            "forward": str(forward_path),
            "reverse": str(reverse_path),
        },
        "stats": {
            "amudMatchGroups": len(amud_matches),
            "matchGroups": total_matches,
            "forwardLinks": len(forward_entries),
            "reverseLinks": len(reverse_entries),
            "skippedUnstablePairs": total_skipped_unstable_pairs,
            "unmatchedFriedbergSegments": total_unmatched_friedberg,
            "unmatchedSefariaSegments": total_unmatched_sefaria,
        },
        "unmatchedAmudGroups": report_unmatched_amudim,
        "amudim": report_amudim,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return forward_path, reverse_path, report_path


def main() -> int:
    args = parse_args()
    friedberg_dir = args.friedberg_dir.resolve()
    output_dir = args.output.resolve()

    if not friedberg_dir.exists():
        warn(f"Friedberg output directory does not exist: {friedberg_dir}")
        return 1

    try:
        tractates = args.tractate or discover_tractates(friedberg_dir)
    except FileNotFoundError as exc:
        warn(str(exc))
        return 1

    for tractate_name in tractates:
        print(f"Generating links for {tractate_name}...", file=sys.stderr)
        forward_path, reverse_path, report_path = generate_for_tractate(
            tractate_name=tractate_name,
            friedberg_dir=friedberg_dir,
            output_dir=output_dir,
            retry_attempts=args.retry_attempts,
            retry_backoff=args.retry_backoff,
        )
        print(f"Wrote {forward_path}", file=sys.stderr)
        print(f"Wrote {reverse_path}", file=sys.stderr)
        print(f"Wrote {report_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
