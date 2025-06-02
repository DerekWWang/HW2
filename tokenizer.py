import re

def tokenize(s: str):
    tokens = re.findall(r'<[se]>' + r'|.', s)
    return tokens


print(tokenize("<s>I love machine learning<e>"))