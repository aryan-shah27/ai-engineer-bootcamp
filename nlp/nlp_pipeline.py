"""
NLP pipeline on AG News (120,000 news articles in 4 categories).

Each stage uses a part of the NLP module of "The AI Engineer Course 2026" (365 Careers):
  preprocess  lowercasing, stop words, regex, tokenization, stemming, lemmatization, n-grams
  pos_ner     part-of-speech tagging and named entity recognition        
  sentiment   rule-based vs pre-trained transformer sentiment            
  vectorize   bag of words vs TF-IDF                                     
  topics      LDA and LSA topic modelling                                
  classify    logistic regression, naive Bayes, linear SVM               
  report      case-study summary of every stage                          

Run everything:   python nlp/nlp_pipeline.py
Run one stage:    python nlp_pipeline.py --stage preprocess
Charts and tables are written to nlp/results/.
"""
import argparse
import html
import os
import re
from collections import Counter
from itertools import chain
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import datasets
import matplotlib

matplotlib.use("Agg")  # save charts as PNG files instead of opening windows
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

N = 4000  # articles sampled for the analysis stages (1,000 per category)
SEED = 42
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]
RESULTS = Path(__file__).parent / "results"

STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
EXTRA_STOP_WORDS = {"said", "says", "would", "could", "also"}


# --------------------------------------------------------------------------- helpers

def save_fig(fig, name):
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/{name}")


def save_table(df, name, title, append=False):
    """Write a DataFrame as a Markdown table so it renders on GitHub."""
    cols = [str(c) for c in df.columns]
    lines = [f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "a" if append else "w", encoding="utf-8") as f:
        f.write(("\n" if append else "") + "\n".join(lines) + "\n")
    if not append:
        print(f"  wrote results/{name}")


def ensure_nltk_data():
    """Download the NLTK resources this script needs, once."""
    for path, package in [("tokenizers/punkt_tab", "punkt_tab"),
                          ("corpora/stopwords", "stopwords"),
                          ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


# --------------------------------------------------------------------------- data

def load_data():
    """Download AG News from Hugging Face (cached after the first run) and take a stratified sample."""
    datasets.disable_progress_bars()
    ds = datasets.load_dataset("fancyzhx/ag_news")
    assert ds["train"].features["label"].names == CATEGORIES

    train, test = ds["train"].to_pandas(), ds["test"].to_pandas()
    for df in (train, test):
        df["category"] = df["label"].map(dict(enumerate(CATEGORIES)))

    sample = (train.groupby("label", group_keys=False)
                   .sample(n=N // len(CATEGORIES), random_state=SEED)
                   .reset_index(drop=True))
    return train, test, sample


# --------------------------------------------------------------------------- stage 1: preprocessing

# News agencies and feeds that AG News leaves inside the text, e.g. "(Reuters) Reuters - Stocks rose..."
SOURCES = (r"Reuters|AP|AFP|CP|Canadian Press|Sports Network|CBS\.MW|NewsFactor|PC World|SPACE\.com|"
           r"USATODAY\.com|washingtonpost\.com|Ziff Davis|newratings\.com|COMPUTERWORLD|CNN/Money|"
           r"InfoWorld|IDG NEWS SERVICE|MacCentral|Forbes\.com|Update\d|Quote, Chart|Ticker")
SOURCE_TAG = re.compile(rf"\((?:{SOURCES})\)")                          # "(Reuters)"
ATTRIBUTION = re.compile(rf"\b(?:{SOURCES})\s+-\s+")                     # "Reuters - "
TICKER_LINK = re.compile(r"\([^()]{0,30}:\s*Quote, Profile, Research\)")  # "(MLS.N: Quote, Profile, Research)"
# "NEW YORK - ", "KYIV, Ukraine - ", "NEW YORK, Aug 17 - ", "BURLINGAME, CALIF.--"
DATELINE = re.compile(r"\b[A-Z][A-Z.'/ ]{3,}?(?:,\s[\w. ]{1,25}?)?(?:\s+-{1,2}|--)\s*")
BROKEN_ENTITY = re.compile(r"#(\d+);")                                   # "#36;" is "&#36;" ($) with the & lost
MONEY = re.compile(r"\$\s?\d(?:[\d,]*\d)?(?:\.\d+)?(?:\s?(?:million|billion|trillion|bn|mn|m)\b)?", re.IGNORECASE)
URL = re.compile(r"https?://\S+|www\.\S+")
WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")                                 # "oil", "cease-fire"; not "-the" or "2004"
# used to measure the cleanup
ARTIFACTS = re.compile(rf"#\d+;|&(?:lt|gt|amp|quot);|\\|\((?:{SOURCES})\)|Quote, Profile, Research")


def _entity_to_char(match):
    code = int(match.group(1))
    if 128 <= code <= 159:  # Windows-1252 codes, e.g. #151; is an em dash
        return bytes([code]).decode("cp1252", errors="ignore")
    return chr(code) if code <= 0x10FFFF else ""


def fix_artifacts(text):
    """Undo the encoding damage in AG News so the text reads like the original article."""
    text = text.replace("\\", " ")                    # backslashes were line breaks
    text = re.sub(r"\s?#39;", "'", text)              # "Haiti #39;s" -> "Haiti's"
    text = BROKEN_ENTITY.sub(_entity_to_char, text)   # "#36;46" -> "$46"
    text = re.sub(r"\bquot;", '"', text)
    text = html.unescape(text)                        # "&lt;b&gt;" -> "<b>"
    text = re.sub(r"<[^>]+>", " ", text)              # then drop the tags
    return re.sub(r"\s+", " ", text).strip()


def strip_sources(text):
    """'NEW YORK (Reuters) - Stocks rose' -> 'Stocks rose'."""
    text = TICKER_LINK.sub(" ", text)
    text = SOURCE_TAG.sub(" ", text)
    text = ATTRIBUTION.sub(" ", text)
    text = DATELINE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_money(text):
    """'$46 a barrel ... $1.17 billion' -> ['$46', '$1.17 billion']"""
    return MONEY.findall(text)


def tokenize(text, stop_words):
    """lowercase -> drop URLs -> split into words -> keep real words (incl. 'cease-fire') that aren't stop words."""
    text = URL.sub(" ", text.lower())
    return [w for w in word_tokenize(text)
            if WORD.fullmatch(w) and len(w) > 2 and w not in stop_words]


def prepare(df):
    """Add the preprocessing columns every stage uses.

    text_fixed  regex-cleaned text, case kept (NER on day 2 needs capital letters)
    tokens      lowercased words without stop words
    stems       Porter stems of the tokens
    lemmas      WordNet lemmas of the tokens
    clean_text  lemmas joined back into one string (input for BoW, TF-IDF, topics, classifier)
    """
    ensure_nltk_data()
    stop_words = set(stopwords.words("english")) | EXTRA_STOP_WORDS

    df["text_fixed"] = df["text"].apply(lambda t: strip_sources(fix_artifacts(t)))
    df["money"] = df["text_fixed"].apply(extract_money)
    df["tokens"] = df["text_fixed"].apply(lambda t: tokenize(t, stop_words))
    df["stems"] = df["tokens"].apply(lambda toks: [STEMMER.stem(w) for w in toks])
    df["lemmas"] = df["tokens"].apply(lambda toks: [LEMMATIZER.lemmatize(w) for w in toks])
    df["clean_text"] = df["lemmas"].str.join(" ")
    return df


def stage_preprocess(df, **_):
    """Course: Text Preprocessing - lowercasing, stop words, regex, tokenization, stemming,
    lemmatization, n-grams, pandas apply() with lambdas."""
    # 1. How much did the regex cleanup fix?
    before = df["text"].str.contains(ARTIFACTS).sum()
    after = df["text_fixed"].str.contains(ARTIFACTS).sum()
    examples = df[df["text"].str.contains(ARTIFACTS)].sample(3, random_state=SEED)
    save_table(pd.DataFrame({"raw text": examples["text"].str[:150],
                             "after regex cleanup": examples["text_fixed"].str[:150],
                             "tokens (lemmatized)": examples["lemmas"].apply(lambda t: " ".join(t[:12]))}),
               "cleaning_examples.md", "Raw vs cleaned text")
    summary = pd.DataFrame({
        "metric": ["articles with encoding junk or source tags", "average words per article",
                   "average tokens after cleaning"],
        "value": [f"{before:,} before, {after:,} after",
                  f"{df['text_fixed'].str.split().str.len().mean():.1f}",
                  f"{df['tokens'].str.len().mean():.1f}"]})
    save_table(summary, "cleaning_examples.md", "Cleanup summary", append=True)
    print(f"  artifacts: {before:,} articles before cleanup -> {after:,} after")

    # 2. Stemming vs lemmatization
    words = ["studies", "companies", "officials", "economic", "prices",
             "running", "playing", "went", "wolves", "universities"]
    save_table(pd.DataFrame({"word": words,
                             "Porter stem": [STEMMER.stem(w) for w in words],
                             "lemma (as noun)": [LEMMATIZER.lemmatize(w) for w in words],
                             "lemma (as verb)": [LEMMATIZER.lemmatize(w, pos="v") for w in words]}),
               "stem_vs_lemma.md", "Stemming vs lemmatization")
    vocab = {col: len(set(chain.from_iterable(df[col]))) for col in ("tokens", "stems", "lemmas")}
    save_table(pd.DataFrame({"representation": list(vocab), "unique words": [f"{v:,}" for v in vocab.values()]}),
               "stem_vs_lemma.md", f"Vocabulary size across {len(df):,} articles", append=True)
    print(f"  vocabulary: {vocab['tokens']:,} words -> {vocab['stems']:,} stems / {vocab['lemmas']:,} lemmas")

    # 3. N-grams: top bigrams per category (chart) and top trigrams (table)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    trigram_rows = []
    for ax, cat in zip(axes.flat, CATEGORIES):
        lemmas = df.loc[df["category"] == cat, "lemmas"]
        bigrams = Counter(bg for toks in lemmas for bg in ngrams(toks, 2)).most_common(10)[::-1]
        ax.barh([" ".join(bg) for bg, _ in bigrams], [n for _, n in bigrams])
        ax.set_title(cat)
        trigrams = Counter(tg for toks in lemmas for tg in ngrams(toks, 3)).most_common(5)
        trigram_rows.append({"category": cat, "top trigrams": ", ".join(" ".join(tg) for tg, _ in trigrams)})
    fig.suptitle(f"Top 10 bigrams per category ({len(df):,} articles, after cleaning)")
    fig.tight_layout()
    save_fig(fig, "top_bigrams.png")
    save_table(pd.DataFrame(trigram_rows), "top_trigrams.md", "Top 5 trigrams per category")

    # 4. Regex extraction: dollar amounts
    has_money = df["money"].str.len() > 0
    money_rows = []
    for cat in CATEGORIES:
        amounts = chain.from_iterable(df.loc[(df["category"] == cat) & has_money, "money"])
        money_rows.append({"category": cat,
                           "articles with a $ amount": f"{has_money[df['category'] == cat].mean():.1%}",
                           "examples": ", ".join(list(dict.fromkeys(amounts))[:6])})
    save_table(pd.DataFrame(money_rows), "money_mentions.md", "Dollar amounts found by regex")


# --------------------------------------------------------------------------- pipeline

def main():
    parser = argparse.ArgumentParser(description="One-file NLP pipeline on AG News")
    parser.add_argument("--stage", choices=["all", "preprocess"], default="all",
                        help="run a single stage instead of all of them")
    args = parser.parse_args()

    train, test, df = load_data()
    print(f"AG News: {len(train):,} train / {len(test):,} test articles, analysing a sample of {len(df):,}")
    df = prepare(df)

    if args.stage in ("all", "preprocess"):
        print("\n[preprocess]")
        stage_preprocess(df)


if __name__ == "__main__":
    main()
