import pandas as pd
from jsonargparse import CLI
from pathlib import Path
import json
import logging

from spacy.lang.xx import Language
from spacy.pipeline.sentencizer import Sentencizer

from .utils import clean_text
from .encoder import Encoder
from .aligner import Bertalign


def main(
        src_path: Path, 
        tgt_path: Path, 
        tgt_lang: str,
        model_name: str = "sentence-transformers/LaBSE", 
        input_key: str = "text", 
        src_lang: str = "en",
        max_align: int = 5,
        top_k: int = 3,
        win: int = 5,
        skip: float = -0.1,
        len_slack: float = 0.,
        verbose: int = logging.INFO
    ):
    """Multi-parallel alignment from multiple bilingual alignments using Bertalign"""
    sentencizer = Language()
    sentencizer.add_pipe("sentencizer", config={"punct_chars": ["\n","\n\n"] + Sentencizer.default_punct_chars})
    logging.basicConfig(level=verbose)
    src = pd.read_csv(src_path)
    with open(tgt_path, 'rt') as file:
        tgt = json.load(file)
    assert len(tgt) == 1
    model = Encoder(model_name)

    src_sentence_path = src_path.parent/f"{src_lang}_sentences.json"
    if src_sentence_path.exists():
        with open(src_sentence_path, 'rt') as file:
            src_sentences = json.load(file)
    else:
        src_sentences = []
        for doc in sentencizer.pipe(src[input_key]):
            for sent in doc.sents:
                src_sentences.append(clean_text(sent.text))
        with open(src_sentence_path, 'wt') as file:
            json.dump(src_sentences, file)

    tgt_sentences = []
    tgt_sentences = [clean_text(sent.text) for sent in sentencizer(tgt[0]).sents]
    aligner = Bertalign(
        model, 
        src_sentences, 
        tgt_sentences, 
        max_align=max_align,
        top_k=top_k,
        win=win,
        skip=skip,
        len_slack=len_slack,
        src_lang=src_lang, 
        tgt_lang=tgt_lang
    )
    aligner.align_sents()
    src_is, tgt_is = [], []
    for src_i, tgt_i in aligner.result:
        src_is.append([int(i) for i in src_i])
        tgt_is.append([int(i) for i in tgt_i])
    alignments = {"src": src_is, "tgt": tgt_is}
    alignment_scores = aligner.scores["cos"]

    with open(tgt_path.parent/f"{tgt_lang}_sentences.json", "wt") as file:
        json.dump(tgt_sentences, file)
    with open(tgt_path.parent/f"{tgt_lang}_alignments.json", "wt") as file:
        json.dump(alignments, file)
    with open(tgt_path.parent/f"{tgt_lang}_alignment_scores.json", "wt") as file:
        json.dump(alignment_scores, file)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
