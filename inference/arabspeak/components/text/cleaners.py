from typing import Dict, List
import re

abbreviations_ar = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("أ.د.م", "الأستاذ الدكتور المهندس"),
        ("أ.د", "الأستاذ الدكتور"),
        ("أ.م", "الأستاذ المهندس"),
        ("أ", "الأستاذ"),
        ("ا", "الأستاذ"),
        ("د", "الدكتور"),
        ("م", "المهندس"),
    ]
]

whitespace_regex = re.compile(r"\s+")
auxilary_symbols = re.compile(r"[\<\>\(\)\[\]\"]+")
symbols_mapping = {
    "،":",",
    "؛":",",
    ":":",",
    "؟":"?",
    "-":",",
    "_":",",
    "(":",",
    "(":",",
    "\"":",",
    "\'":",",
}

def expand_abbreviations(text):
    for regex, replacement in abbreviations_ar:
        text = re.sub(regex, replacement, text)
    return text

def remove_aux_symbols(text):
    text = auxilary_symbols.sub("", text)
    return text

def replace_symbols(text):
    for symbol, replacement in symbols_mapping.items():
        text = text.replace(symbol, replacement)
    return text

def collapse_whitespace(text):
    return whitespace_regex.sub(" ", text).strip()

#TODO: replace numbers, dates, and times.

def clean_text(text):
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text
