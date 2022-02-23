import os
import argparse

parser = argparse.ArgumentParser(description='Create the char_map.txt and pos_map.txt file.')
parser.add_argument('--output_dir', type=str,
                    help='The directory to output the results.', default="output")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

SEPARATOR = set("\u200b")
CONSTS = set(u"កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ")
VOWELS = set(u"឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8")
SUB = set(u"្")
DIAC = set(u"\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0")
SYMS = set("៕។៛ៗ៚៙៘,.?!@#%^&*()_+-=[]/\<>")
NUMBERS = set(u"០១២៣៤៥៦៧៨៩0123456789")

characters = sorted(list(SEPARATOR) + list(CONSTS) + list(VOWELS) + list(SUB) + list(DIAC) + list(SYMS) + list(NUMBERS))

with open(os.path.join(args.output_dir, "char_map.txt"), "w") as file:
    for i, char in enumerate(characters):
        file.write(f"{char}\t{i}")
        if i != len(characters)-1:
            file.write("\n")

with open(os.path.join(args.output_dir, "pos_map.txt"), "w") as file:
    tags = ["AB", "AUX", "CC", "CD", "DT", "IN", "JJ", "VB", "NN", "PN", "PA", "PRO", "QT", "RB", "SYM", "NS"]
    for i, tag in enumerate(tags):
        file.write(f"{tag}")
        if i != len(tags)-1:
            file.write("\n")
