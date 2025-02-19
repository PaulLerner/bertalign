#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import pandas as pd
import json
from pathlib import Path

# load data
# FIXME customize path
root = Path("/home/paul/code/iznogoud/data/linkedep/")

sentences = {}
for path in root.glob("*_sentences.json"):
    lang = path.name.split("_")[0]
    with open(path, "rt") as file:
        sentences[lang] = json.load(file)

bials = {}
for path in root.glob("*_alignments.json"):
    lang = path.name.split("_")[0]
    with open(path, "rt") as file:
        bials[lang] = json.load(file)

scores = {}
for path in root.glob("*scores.json"):
    lang = path.name.split("_")[0]
    with open(path, "rt") as file:
        scores[lang] = json.load(file)


def get_multi_alignment(G, languages):
    """Multi-parallel alignments are simply connected components of the alignment graph."""
    multi_as = {tgt_lang: [] for tgt_lang in languages}
    multi_scores = []
    for component in nx.connected_components(G):
        multi_a = {tgt_lang: [] for tgt_lang in languages}
        for lang, index in component:
            multi_a[lang].append(index)
        for lang in multi_as.keys():
            multi_as[lang].append(multi_a[lang])
    
        # keep minimum score as multi-parallel score
        score = float("inf")
        for u, v, attr in G.edges(component, data=True):
            if attr["score"] < score:
                score = attr["score"]
        multi_scores.append(score)
    return multi_as, multi_scores


# compute multi-parallel alignments
src_lang="en"
all_multi_alignments = {}
all_multi_scores= {}
for speech in sentences[src_lang].keys():
    # each sentence in a given language is a node in a graph
    # and is aligned to other sentences in other languages: 
    # edge of the graph weighted by the alignment score
    G = nx.Graph()
    for tgt_lang, bial in bials.items():
        for i, (src_range, tgt_range) in enumerate(zip(bial[speech]["src"], bial[speech]["tgt"])):
            score = scores[tgt_lang][speech][i]
            for src_i in src_range:
                for tgt_i in tgt_range:
                    G.add_node((src_lang,src_i))
                    G.add_node((tgt_lang, tgt_i))
                    G.add_edge((src_lang,src_i),(tgt_lang, tgt_i),score=score)

    multi_as, multi_scores = get_multi_alignment(G, sentences.keys())
    all_multi_alignments[speech] = multi_as
    all_multi_scores[speech] = multi_scores


with open(root/"multi-alignments.json","wt") as file:
    json.dump(all_multi_alignments, file)
with open(root/"multi-scores.json","wt") as file:
    json.dump(all_multi_scores, file)


# format as the original dataset
data = pd.read_csv(root/"21-multi-europarl.csv", index_col="Unnamed: 0")
metadata = set(c for c in data.columns if len(c)!=2)
sentence_data = []
for key, row in data.iterrows():
    for i, score in enumerate(all_multi_scores[key]):
        # FIXME customize threshold
        if score < 0.8:
            continue
        sentence_row = {"speech": key}
        ok = True
        for lang in all_multi_alignments[key].keys():
            sent_range = all_multi_alignments[key][lang][i]
            # beware sent_range is not sorted (ok because we keep only 1-1 alignments)
            if len(sent_range) != 1:
                # FIXME customize criterion for keeping 1-1 or 1-many alignments
                ok = False
                break
            j = sent_range[0] #for j in sent_range
            sentence_row[lang] = sentences[lang][key][j]
        if not ok:
            continue
        for field in metadata:
            sentence_row[field] = row[field]
        sentence_data.append(sentence_row)
                
sentence_data=pd.DataFrame(sentence_data)
sentence_data.to_csv(root/"21-multi-europarl-sent-1-1-8.csv")
