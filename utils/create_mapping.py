import os
import argparse

SEPARATOR = set("\u200b")

# list of constants needed for KCC and feature generation
# consonant and independent vowels
CONSTS = set(u"កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ")
VOWELS = set(u"឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8")

# subscript, diacritics
SUB = set(u"្")

# MUUSIKATOAN, TRIISAP, BANTOC,ROBAT,
DIAC = set(u"\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0")

SYMS = set("៕។៛ៗ៚៙៘,.?!@#%^&*()_+-=[]/\<>")  # _ is space
NUMBERS = set(u"០១២៣៤៥៦៧៨៩0123456789")  # remove 0123456789

characters = sorted(list(SEPARATOR) + list(CONSTS) + list(VOWELS) + list(SUB) + list(DIAC) + list(SYMS) + list(NUMBERS))

with open("char_map.txt", "w") as file:
    for i, char in enumerate(characters):
        file.write(f"{char}\t{i}\n")

with open("pos_map.txt", "w") as file:
    for i, tag in enumerate(["UNK", "AB", "AUX", "CC", "CD", "DT", "IN", "JJ", "VB", "NN", "PN", "PA", "PRO", "QT", "RB", "SYM", "NS", "PAD"]):
        file.write(f"{tag}\t{i}\n")
