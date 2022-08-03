import re

def decontract(phrase):
    # specific
    phrase = phrase.lower()
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def extract_tokens(tokens: list, punctuation: bool = False, lemmatize: bool = True):
    words = []
    for token in tokens:
        if token.is_digit or token.like_num:
            words.append("quantity")
            # TODO distinguish (using ent) between cardinal and ordinal
        elif token.is_currency:
            words.append("currency")
        elif token.like_url:
            words.append("url")
        elif token.like_email:
            words.append("email")
        elif token.text[0] == "@":
            words.append("mention")
        elif not token.is_stop and not token.is_space:
            if lemmatize:
                words.append(token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_)
            else:
                words.append(token.text.lower().strip() if token.lemma_ != "-PRON-" else token.lower_)
        else:
            pass
    return ' '.join(words)