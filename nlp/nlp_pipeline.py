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
from matplotlib.patches import Patch
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

N = 4000  # articles sampled for the analysis stages (1,000 per category)
SENTIMENT_N = 1000  # the transformer manages ~20 articles/second on a CPU, so sentiment uses a smaller sample
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
                          ("corpora/wordnet", "wordnet"),
                          ("sentiment/vader_lexicon.zip", "vader_lexicon")]:
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


# --------------------------------------------------------------------------- stage 2: POS tagging and NER

POS_TAGS = ["PROPN", "NOUN", "VERB", "ADJ", "NUM"]
ENTITY_LABELS = ["PERSON", "ORG", "GPE", "NORP", "DATE", "MONEY", "CARDINAL"]


def stage_pos_ner(df):
    """Course: Identifying Parts of Speech and Named Entities - POS tagging and NER with spaCy.

    Runs on text_fixed, not the stage 1 tokens: spaCy needs capital letters and full sentences
    to tell "Apple" the company from "apple" the fruit.
    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])  # only the tagger and NER are needed
    docs = list(nlp.pipe(df["text_fixed"], batch_size=200))

    # 1. POS tagging: what share of each category's words are names, nouns, verbs, adjectives, numbers?
    pos_counts = pd.DataFrame([Counter(t.pos_ for t in doc if not (t.is_punct or t.is_space)) for doc in docs])
    pos_share = pos_counts.fillna(0).groupby(df["category"]).sum()
    pos_share = pos_share.div(pos_share.sum(axis=1), axis=0).reindex(CATEGORIES)[POS_TAGS] * 100
    ax = pos_share.plot.bar(figsize=(10, 5), rot=0)
    ax.set_ylabel("% of words")
    ax.set_xlabel("")
    ax.set_title("Part-of-speech mix per category")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    save_fig(ax.figure, "pos_by_category.png")
    print("  POS (% of words): " + ", ".join(f"{cat} {pos_share.loc[cat, 'PROPN']:.0f}% PROPN / "
                                              f"{pos_share.loc[cat, 'NUM']:.0f}% NUM" for cat in CATEGORIES))

    # 2. NER: which kinds of entities does each category talk about?
    ents = pd.DataFrame([(cat, ent.text, ent.label_) for cat, doc in zip(df["category"], docs) for ent in doc.ents],
                        columns=["category", "text", "label"])
    per_article = (ents[ents["label"].isin(ENTITY_LABELS)].groupby(["category", "label"]).size()
                   .unstack(fill_value=0).reindex(index=CATEGORIES, columns=ENTITY_LABELS, fill_value=0)
                   .div(df["category"].value_counts(), axis=0))
    ax = per_article.plot.bar(figsize=(11, 5), rot=0)
    ax.set_ylabel("entities per article")
    ax.set_xlabel("")
    ax.set_title("Named entities per article, by type")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    save_fig(ax.figure, "entities_by_category.png")

    rows = []
    for cat in CATEGORIES:
        row = {"category": cat}
        for label in ["PERSON", "ORG", "GPE"]:
            top = ents.loc[(ents["category"] == cat) & (ents["label"] == label), "text"].value_counts().head(5)
            row[label] = ", ".join(f"{name} ({n})" for name, n in top.items())
        rows.append(row)
    save_table(pd.DataFrame(rows), "top_entities.md", "Top 5 people (PERSON), organisations (ORG) and places (GPE)")

    # 3. Multi-word names: what stage 1's word-by-word tokens broke apart ("united state")
    names = ents[ents["label"].isin(["PERSON", "ORG", "GPE"])]
    multi_word = names["text"].str.contains(" ")
    save_table(pd.DataFrame({
        "metric": ["entities found", "entities per article", "people/organisations/places spanning 2+ words",
                   "most common multi-word names"],
        "value": [f"{len(ents):,}", f"{len(ents) / len(df):.1f}", f"{multi_word.mean():.0%}",
                  ", ".join(names.loc[multi_word, "text"].value_counts().head(6).index)]}),
        "top_entities.md", "NER summary", append=True)
    print(f"  NER: {len(ents):,} entities ({len(ents) / len(df):.1f} per article), "
          f"{multi_word.mean():.0%} of people/organisations/places span 2+ words")

    # 4. Where NER gets confused: the same name given different labels in different articles
    #    (plain numbers are skipped: "50" really can be money in one article and a count in another)
    labels_per_name = ents[ents["text"].str.contains("[A-Za-z]")].groupby("text")["label"].agg(Counter)
    mixed = [(name, c) for name, c in labels_per_name.items() if sum(c.values()) >= 5 and len(c) > 1]
    mixed.sort(key=lambda item: sum(item[1].values()) - item[1].most_common(1)[0][1], reverse=True)
    save_table(pd.DataFrame([{"name": name, "labels spaCy gave it": ", ".join(f"{lab} x{n}" for lab, n in c.most_common())}
                             for name, c in mixed[:10]]),
               "top_entities.md", "Same name, different labels (where NER gets confused)", append=True)


# --------------------------------------------------------------------------- stage 3: sentiment analysis

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def vader_label(compound):
    """VADER's usual cut-offs: compound >= 0.05 is positive, <= -0.05 negative, anything between is neutral."""
    return "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"


def stage_sentiment(df):
    """Course: Sentiment Analysis - rule-based (VADER) vs a pre-trained transformer.

    Both run on text_fixed: VADER scores capitals and "!" as intensity, and the transformer
    was trained on normal sentences, so neither should see the stage 1 tokens.
    """
    from transformers import pipeline, logging as hf_logging  # imported here: loading torch is slow
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    df = (df.groupby("category", group_keys=False)  # both models score the same 1,000 articles
            .sample(n=SENTIMENT_N // len(CATEGORIES), random_state=SEED).copy())

    # 1. Rule-based: VADER adds up word scores from a hand-made lexicon ("crisis" -, "win" +)
    vader = SentimentIntensityAnalyzer()
    df["vader"] = df["text_fixed"].apply(lambda t: vader.polarity_scores(t)["compound"])
    df["vader_label"] = df["vader"].apply(vader_label)

    # 2. Pre-trained transformer: DistilBERT fine-tuned on movie-review sentences (SST-2), positive/negative only
    classifier = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    preds = classifier(df["text_fixed"].tolist(), batch_size=32, truncation=True)
    df["bert_label"] = [p["label"].lower() for p in preds]
    df["bert"] = [p["score"] if p["label"] == "POSITIVE" else -p["score"] for p in preds]

    # 3. Does sentiment differ by news type? Label shares per category for both models
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    colors = {"negative": "#d62728", "neutral": "#9e9e9e", "positive": "#2ca02c"}
    for ax, col, title in [(axes[0], "vader_label", "VADER (rule-based)"),
                           (axes[1], "bert_label", "DistilBERT (transformer)")]:
        shares = (pd.crosstab(df["category"], df[col], normalize="index") * 100).reindex(CATEGORIES)
        shares = shares[[c for c in colors if c in shares.columns]]
        shares.plot.barh(stacked=True, ax=ax, color=[colors[c] for c in shares.columns], legend=False)
        ax.set_title(title)
        ax.set_xlabel("% of articles")
        ax.set_ylabel("")
        ax.invert_yaxis()
    fig.legend(handles=[Patch(color=c, label=label) for label, c in colors.items()],
               loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"Sentiment of {len(df):,} news articles by category")
    fig.tight_layout()
    save_fig(fig, "sentiment_by_category.png")

    # 4. How often do they agree? (only where VADER picked a side - the transformer has no "neutral")
    decided = df["vader_label"] != "neutral"
    agree = (df.loc[decided, "vader_label"] == df.loc[decided, "bert_label"]).mean()
    summary = pd.DataFrame({
        "metric": ["articles scored", "articles VADER calls neutral", "articles the transformer calls negative",
                   "agreement where VADER isn't neutral"],
        "value": [f"{len(df):,}", f"{(~decided).mean():.0%}", f"{(df['bert_label'] == 'negative').mean():.0%}",
                  f"{agree:.0%}"]})
    save_table(summary, "sentiment.md", "Rule-based vs transformer sentiment")
    by_cat = df.groupby("category").agg(vader_mean=("vader", "mean"),
                                        vader_neg=("vader_label", lambda s: (s == "negative").mean()),
                                        bert_neg=("bert_label", lambda s: (s == "negative").mean())).reindex(CATEGORIES)
    save_table(pd.DataFrame({"category": CATEGORIES,
                             "VADER mean score (-1 to +1)": [f"{v:+.2f}" for v in by_cat["vader_mean"]],
                             "negative (VADER)": [f"{v:.0%}" for v in by_cat["vader_neg"]],
                             "negative (transformer)": [f"{v:.0%}" for v in by_cat["bert_neg"]]}),
               "sentiment.md", "Sentiment by category", append=True)
    print(f"  VADER neutral on {(~decided).mean():.0%} of articles; models agree on {agree:.0%} of the rest")

    # 5. The biggest disagreements: confident opposite answers
    clash = df[decided & (df["vader_label"] != df["bert_label"])].copy()
    clash["strength"] = clash["vader"].abs() * clash["bert"].abs()
    rows = []
    for vader_side in ["positive", "negative"]:
        for _, r in clash[clash["vader_label"] == vader_side].nlargest(3, "strength").iterrows():
            rows.append({"category": r["category"], "article": r["text_fixed"][:140],
                         "VADER": f"{r['vader']:+.2f}", "transformer": f"{r['bert_label']} ({abs(r['bert']):.2f})"})
    save_table(pd.DataFrame(rows), "sentiment.md", "Biggest disagreements", append=True)


# --------------------------------------------------------------------------- stage 4: vectorizing

def similarity_by_category(sim, categories):
    """Average cosine similarity between the articles of every pair of categories."""
    return pd.DataFrame([[np.nanmean(sim[np.ix_(categories == a, categories == b)]) for b in CATEGORIES]
                         for a in CATEGORIES], index=CATEGORIES, columns=CATEGORIES)


def most_similar(sim, i, k=3):
    """Indexes of the k articles closest to article i (sim has NaN on its diagonal, so i never matches itself)."""
    return np.argsort(-np.nan_to_num(sim[i], nan=-1), kind="stable")[:k]


def stage_vectorize(df):
    """Course: Vectorizing Text - bag of words (CountVectorizer) vs TF-IDF (TfidfVectorizer), cosine similarity.

    Runs on clean_text from stage 1 (lowercased lemmas, no stop words). min_df=2 drops words that appear
    in only one article: typos and one-off names can't link two articles together anyway.
    """
    categories = df["category"].to_numpy()
    vectorizers = {"bag of words": CountVectorizer(min_df=2),
                   "TF-IDF": TfidfVectorizer(min_df=2),
                   "TF-IDF + bigrams": TfidfVectorizer(min_df=2, ngram_range=(1, 2))}

    # 1. Text -> vectors. Compare size, sparsity, and whether similar vectors mean similar topics
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    heat_axes = iter(axes)
    matrices, stats = {}, []
    for name, vectorizer in vectorizers.items():
        X = matrices[name] = vectorizer.fit_transform(df["clean_text"])
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, np.nan)  # every article is identical to itself - ignore that
        nearest = np.nanargmax(sim, axis=1)
        by_cat = similarity_by_category(sim, categories)
        within = np.diag(by_cat).mean()
        across = by_cat.to_numpy()[~np.eye(len(CATEGORIES), dtype=bool)].mean()
        stats.append({"representation": name,
                      "vocabulary": f"{X.shape[1]:,}",
                      "non-zero cells": f"{X.nnz / (X.shape[0] * X.shape[1]):.2%}",
                      "own-category vs other-category similarity": f"{within / across:.2f}x",
                      "nearest neighbour in the same category": f"{(categories[nearest] == categories).mean():.1%}"})
        if name == "TF-IDF":
            tfidf_sim = sim
        if name in ("bag of words", "TF-IDF"):
            ax, values = next(heat_axes), by_cat.to_numpy()
            ax.imshow(values, cmap="Blues")
            ax.set_xticks(range(len(CATEGORIES)), CATEGORIES)
            ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
            for r in range(len(CATEGORIES)):
                for c in range(len(CATEGORIES)):
                    ax.text(c, r, f"{values[r, c]:.3f}", ha="center", va="center",
                            color="white" if values[r, c] > 0.6 * values.max() else "black")
            ax.set_title(f"{name}: own category {within / across:.1f}x more similar")
    fig.suptitle("Average cosine similarity between articles of each category")
    fig.tight_layout()
    save_fig(fig, "category_similarity.png")
    save_table(pd.DataFrame(stats), "bow_vs_tfidf.md", "Bag of words vs TF-IDF")
    print("  nearest neighbour in the same category: " +
          ", ".join(f"{s['representation']} {s['nearest neighbour in the same category']}" for s in stats))

    # 2. What TF-IDF changes: top words per category by raw count vs by TF-IDF weight
    vocab = vectorizers["bag of words"].get_feature_names_out()  # same tokens and min_df, so same vocabulary as TF-IDF
    rows = []
    for cat in CATEGORIES:
        in_cat = categories == cat
        counts = np.asarray(matrices["bag of words"][in_cat].sum(axis=0)).ravel()
        weights = np.asarray(matrices["TF-IDF"][in_cat].mean(axis=0)).ravel()
        rows.append({"category": cat,
                     "top 8 by raw count": ", ".join(vocab[np.argsort(-counts, kind="stable")[:8]]),
                     "top 8 by TF-IDF": ", ".join(vocab[np.argsort(-weights, kind="stable")[:8]])})
    save_table(pd.DataFrame(rows), "bow_vs_tfidf.md", "Top words per category", append=True)

    # 3. Why: IDF shrinks words that appear everywhere
    doc_share = matrices["bag of words"].getnnz(axis=0) / len(df)
    idf = vectorizers["TF-IDF"].idf_
    idf_rows = [{"word": vocab[i], "in % of articles": f"{doc_share[i]:.1%}", "IDF weight": f"{idf[i]:.2f}"}
                for i in np.argsort(-doc_share, kind="stable")[:8]]
    idf_rows.append({"word": "(any word in only 2 articles)", "in % of articles": f"{2 / len(df):.2%}",
                     "IDF weight": f"{idf.max():.2f}"})
    save_table(pd.DataFrame(idf_rows), "bow_vs_tfidf.md", "The most common words get the lowest IDF", append=True)

    # 4. "Find similar articles" with TF-IDF + cosine similarity
    rows = []
    for cat in CATEGORIES:
        i = np.flatnonzero(categories == cat)[0]
        rows.append({"article": f"**{df.at[i, 'text_fixed'][:110]}**", "category": cat, "cosine similarity": "query"})
        for j in most_similar(tfidf_sim, i):
            rows.append({"article": "↳ " + df.at[j, "text_fixed"][:110], "category": categories[j],
                         "cosine similarity": f"{tfidf_sim[i, j]:.2f}"})
    save_table(pd.DataFrame(rows), "similar_articles.md", "Most similar articles (TF-IDF + cosine similarity)")


# --------------------------------------------------------------------------- pipeline

def main():
    parser = argparse.ArgumentParser(description="One-file NLP pipeline on AG News")
    parser.add_argument("--stage", choices=["all", "preprocess", "pos_ner", "sentiment", "vectorize"], default="all",
                        help="run a single stage instead of all of them")
    args = parser.parse_args()

    train, test, df = load_data()
    print(f"AG News: {len(train):,} train / {len(test):,} test articles, analysing a sample of {len(df):,}")
    df = prepare(df)

    if args.stage in ("all", "preprocess"):
        print("\n[preprocess]")
        stage_preprocess(df)
    if args.stage in ("all", "pos_ner"):
        print("\n[pos_ner]")
        stage_pos_ner(df)
    if args.stage in ("all", "sentiment"):
        print("\n[sentiment]")
        stage_sentiment(df)
    if args.stage in ("all", "vectorize"):
        print("\n[vectorize]")
        stage_vectorize(df)


if __name__ == "__main__":
    main()
