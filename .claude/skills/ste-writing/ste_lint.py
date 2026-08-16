# From woosal1337/blog videos/ep01-the-cure-for-ai-slop/ste-lint.py
# MIT License, Copyright (c) 2026 Ege Celebi. See LICENSE in this directory.

import glob
import json
import os
import re
import sys

MARKETING = [
    "seamless",
    "seamlessly",
    "robust",
    "powerful",
    "cutting-edge",
    "effortless",
    "effortlessly",
    "world-class",
    "next-generation",
    "revolutionary",
    "blazing",
    "lightning-fast",
    "elegant",
    "delightful",
    "turnkey",
    "best-in-class",
    "state-of-the-art",
    "game-changing",
    "first-class",
    "battle-tested",
    "enterprise-grade",
    "supercharge",
    "unlock",
    "unleash",
    "empower",
    "empowers",
]
BANNED = [
    "begin",
    "begins",
    "commence",
    "commences",
    "initiate",
    "initiates",
    "originate",
    "utilize",
    "utilizes",
    "utilizing",
    "leverage",
    "leverages",
    "leveraging",
    "facilitate",
    "facilitates",
    "ensure",
    "ensures",
    "ensuring",
    "prior to",
    "subsequent to",
    "obtain",
    "obtains",
    "acquire",
    "acquires",
    "demonstrate",
    "demonstrates",
    "additionally",
    "furthermore",
    "moreover",
    "comprehensive",
    "comprehensively",
    "utilization",
    "aforementioned",
    "henceforth",
    "therein",
    "whilst",
    "amongst",
    "numerous",
    "myriad",
    "plethora",
    "in order to",
    "a variety of",
    "in the event that",
    "due to the fact that",
    "it is important to note",
]
PHRASAL = [
    "spin up",
    "spin down",
    "reach out",
    "dive into",
    "dives into",
    "diving into",
    "kick off",
    "kicks off",
    "roll out",
    "rolls out",
    "tear down",
    "ramp up",
    "circle back",
    "drill down",
    "spun up",
    "reaching out",
]
MODAL_HEDGE = [
    "it is important to note",
    "it should be noted",
    "it is worth noting",
    "please note that",
    "as mentioned",
    "as noted above",
]
BE = r"(?:am|is|are|was|were|be|been|being)"
PP_IRREG = r"(?:done|made|sent|read|built|kept|held|set|put|run|written|shown|given|taken|found|got|gotten|seen|known|thrown|drawn)"


def strip_code(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    return re.sub(r"`[^`]*`", " ", t)


def sentences(text):
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^\s*#{1,6}\s*", "", s)
        s = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", s)
        if not s:
            continue
        for part in re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9\"'\-])", s):
            part = part.strip()
            if part:
                out.append(part)
    return out


def wc(s):
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-/]*", s))


def count_ci(text, phrases):
    hits = []
    low = text.lower()
    for phrase in phrases:
        hits.extend(phrase for _ in re.finditer(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", low))
    return len(hits), hits


def lint(text):
    raw = text
    text = strip_code(text)
    sents = sentences(text)
    words = sum(wc(sentence) for sentence in sents) or 1
    violations = {}
    longs = [(wc(sentence), sentence) for sentence in sents if wc(sentence) > 20]
    violations["long_sentence(>20w)"] = len(longs)
    violations["semicolon"] = text.count(";")
    violations["contraction"] = len(re.findall(r"\b\w+['\u2019](?:t|re|ve|ll|d|s|m)\b", text))
    violations["passive_voice"] = len(re.findall(rf"\b{BE}\s+(?:\w+ed|{PP_IRREG})\b", text, re.I))
    violations["ing_main_verb"] = len(re.findall(rf"\b{BE}\s+\w+ing\b", text, re.I))
    violations["nominalization"] = len(
        re.findall(
            r"\b(?:perform(?:s|ed)?|conduct(?:s|ed)?|provide(?:s|d)?|carry out|carries out|make use of|makes use of)\b",
            text,
            re.I,
        )
    ) + len(re.findall(r"\b\w{4,}(?:tion|ment|ance|ence)\s+of\b", text, re.I))
    violations["phrasal_verb"], _ = count_ci(text, PHRASAL)
    violations["banned_word"], banned_hits = count_ci(text, BANNED)
    violations["marketing_adjective"], marketing_hits = count_ci(text, MARKETING)
    violations["modal_hedge"], _ = count_ci(text, MODAL_HEDGE)
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", raw) if paragraph.strip()]
    violations["long_paragraph(>6s)"] = sum(
        1 for paragraph in paragraphs if len(sentences(strip_code(paragraph))) > 6
    )
    total = sum(violations.values())
    return {
        "words": words,
        "sentences": len(sents),
        "violations": violations,
        "total": total,
        "total_per100w": round(total * 100.0 / words, 2),
        "em_dash(slop-marker)": raw.count("\u2014") + raw.count("\u2013"),
        "longest_sentence_words": max(longs)[0] if longs else max((wc(sentence) for sentence in sents), default=0),
        "sample_marketing": list(dict.fromkeys(marketing_hits))[:6],
        "sample_banned": list(dict.fromkeys(banned_hits))[:6],
    }


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print(json.dumps(lint(sys.stdin.read()), indent=2))
        sys.exit(0)
    expanded = []
    for file in files:
        expanded += sorted(glob.glob(file)) if any(char in file for char in "*?[") else [file]
    for file in expanded:
        with open(file) as stream:
            result = lint(stream.read())
        print(
            f"{os.path.basename(file):32} words={result['words']:4d} total={result['total']:3d} "
            f"per100w={result['total_per100w']:6.2f} em_dash={result['em_dash(slop-marker)']:2d}"
        )
