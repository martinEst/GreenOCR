import glob


chars = set()
# Scan all text files in the dataset folder
for filepath in glob.glob("data/ICDAR/targets.txt"):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            chars.update(line)  # add all characters from the line


print(chars)
