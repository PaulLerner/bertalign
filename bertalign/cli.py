import pandas as pd
from jsonargparse import CLI
from pathlib import Path
import json
import logging

import trankit

from .utils import clean_text, detect_lang
from .encoder import Encoder
from .aligner import Bertalign


def main(
        data_path: Path, 
        tgt_lang: str,
        model_name: str = "sentence-transformers/LaBSE", 
        trankit_cache_dir: Path = None, 
        src_lang: str = "en",
        verbose: int = logging.INFO
    ):
    """Multi-parallel alignment from multiple bilingual alignments using Bertalign"""
    logging.basicConfig(level=verbose)
    data = pd.read_csv(data_path, index_col="Unnamed: 0")
    model = Encoder(model_name)

    src_sentence_path = data_path.parent/f"{src_lang}_sentences.json"
    if src_sentence_path.exists():
        with open(src_sentence_path, 'rt') as file:
            src_sentences = json.load(file)
    else:
        src_sentences = {key: {} for key in data.index}
        sentence_seg = trankit.Pipeline(lang=trankit.utils.code2lang[src_lang], cache_dir=trankit_cache_dir, gpu=True)
        for key, text in zip(data.index, data[src_lang]):
            text = clean_text(text)
            src_sentences[key] = [s["text"] for s in sentence_seg.ssplit(text)['sentences']]
        with open(src_sentence_path, 'wt') as file:
            json.dump(src_sentences, file)

    tgt_sentences = {key: {} for key in data.index}
    alignments = {key: {} for key in data.index}
    alignment_scores = {key: {} for key in data.index}
    sentence_seg = trankit.Pipeline(lang=trankit.utils.code2lang[tgt_lang], cache_dir=trankit_cache_dir, gpu=True)
    for key, text in zip(data.index, data[tgt_lang]):
        text = clean_text(text)
        tgt_sentences[key] = [s["text"] for s in sentence_seg.ssplit(text)['sentences']]
        aligner = Bertalign(model, src_sentences[key], tgt_sentences[key], src_lang=src_lang, tgt_lang=tgt_lang)
        aligner.align_sents()
        src_is, tgt_is = [], []
        for src_i, tgt_i in aligner.result:
            src_is.append([int(i) for i in src_i])
            tgt_is.append([int(i) for i in tgt_i])
        alignments[key] = {"src": src_is, "tgt": tgt_is}
        alignment_scores[key] = aligner.scores["cos"]

    with open(data_path.parent/f"{tgt_lang}_sentences.json", "wt") as file:
        json.dump(tgt_sentences, file)
    with open(data_path.parent/f"{tgt_lang}_alignments.json", "wt") as file:
        json.dump(alignments, file)
    with open(data_path.parent/f"{tgt_lang}_alignment_scores.json", "wt") as file:
        json.dump(alignment_scores, file)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
