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
        model_name: str = "sentence-transformers/LaBSE", 
        trankit_cache_dir: Path = None, 
        src_lang: str = "en",
        verbose: int = logging.INFO
    ):
    """Multi-parallel alignment from multiple bilingual alignments using Bertalign"""
    logging.basicConfig(level=verbose)
    data = pd.read_csv(data_path, index_col="Unnamed: 0")
    model = Encoder(model_name)

    sentences = {key: {} for key in data.index}
    alignments = {key: {src_lang: {}} for key in data.index}

    sentence_seg = trankit.Pipeline(lang=trankit.utils.code2lang[src_lang], cache_dir=trankit_cache_dir, gpu=True)
    for key, text in zip(data.index, data[src_lang]):
        text = clean_text(text)
        sentences[key][src_lang] = [s["text"] for s in sentence_seg.ssplit(text)['sentences']]

    for tgt_lang in {c for c in data.columns if len(c)==2 and c not in {src_lang, 'mt', 'hr'}}:
        sentence_seg = trankit.Pipeline(lang=trankit.utils.code2lang[tgt_lang], cache_dir=trankit_cache_dir, gpu=True)
        for key, text in zip(data.index, data[tgt_lang]):
            text = clean_text(text)
            sentences[key][tgt_lang] = [s["text"] for s in sentence_seg.ssplit(text)['sentences']]
            aligner = Bertalign(model, sentences[key][src_lang], sentences[key][tgt_lang], src_lang=src_lang, tgt_lang=tgt_lang)
            aligner.align_sents()
            src_lines, tgt_lines = aligner.get_sents()
            alignments[key][src_lang][tgt_lang] = {"src": src_lines, "tgt": tgt_lines}

    with open(data_path.parent/"sentences.json", "wt") as file:
        json.dump(sentences, file)

    with open(data_path.parent/"alignments.json", "wt") as file:
        json.dump(alignments, file)


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
