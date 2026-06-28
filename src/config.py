# src/config.py
import string
import torch
# from genomi:

# 1. Core characters found in 19th-century UK/Irish records
historical_chars = (
    "ſ"          # The long 's' (crucial for 1800s cursive/print)
    "£"          # Currency symbol for fees/tithes
    "æÆœŒ"       # Latin ligatures common in old baptismal names
    "“”„‘’—-"    # Historical punctuation and dashes used in registers
)

all_unique = set(historical_chars + string.ascii_letters + string.digits + string.punctuation + " ")


unique_chars_sorted = sorted(list(all_unique))


# This is going to be used for both training and execution script
VOCAB = ['<blank>'] + unique_chars_sorted

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#alternative but contains duplications, they need to be unique
#vocabSpecialC = ['<blank>'] + list("ÈĒĖēėěęĚĘëéèÉÊËðÐŊŋ") + list("£§êàâé£§⊥")+['£','ſ','—','“','„','’','ô','é']+ list(string.ascii_letters + string.digits + string.punctuation + " ") + ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß', 'å', 'Å', 'æ', 'Æ', 'ø', 'Ø']
""" 
vocabLessSpecial = ['<blank>'] +  punctuation_chars + list("abcdefghijklmnopqrstuvwxyz") + \
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + \
        list("0123456789") + \
        list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ")  # include space

 """





