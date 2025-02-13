from typing import List
import logging

import numpy as np

from bertalign.corelib import *
from bertalign.utils import *


logger = logging.getLogger(__name__)


class Bertalign:
    def __init__(self,
                 model,
                 src_sents: List[str],
                 tgt_sents: List[str],
                 max_align=5,
                 top_k=3,
                 win=5,
                 skip=-0.1,
                 margin=True,
                 len_penalty=True,
                 src_lang = None,
                 tgt_lang = None,
                 cos_similarity = True,
               ):
        
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        
        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        
        src_lang = LANG.ISO[src_lang]
        tgt_lang = LANG.ISO[tgt_lang]
        
        logger.debug("Source language: {}, Number of sentences: {}".format(src_lang, src_num))
        logger.debug("Target language: {}, Number of sentences: {}".format(tgt_lang, tgt_num))

        logger.debug("Embedding source and target text using {} ...".format(model.model_name))
        src_vecs, src_lens = model.transform(src_sents, max_align - 1)
        tgt_vecs, tgt_lens = model.transform(tgt_sents, max_align - 1)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs
        self.scores = None
        self.cos_similarity = cos_similarity
        
    def align_sents(self):

        logger.debug("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.src_vecs[0,:], self.tgt_vecs[0,:], k=self.top_k)
        first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I)
        first_alignment = first_back_track(self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types)
        
        logger.debug("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        second_pointers, cost = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                            second_w, second_path, second_alignment_types,
                                            self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty)
        second_alignment = second_back_track(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types)
        # record alignment scores
        scores, length_ratio = second_back_track_score(self.src_num, self.tgt_num, second_pointers, cost, 
                                                       second_path, second_alignment_types, self.src_lens, self.tgt_lens)

        self.scores = {'bertalign':  scores, 'length_ratio': length_ratio}
        if self.cos_similarity:
            self.scores['cos']  = calculate_cos_similarity(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types,
                                                  self.src_vecs, self.tgt_vecs)
        
        logger.debug("Finished! Successfully aligning {} {} sentences to {} {} sentences\n".format(self.src_num, self.src_lang, self.tgt_num, self.tgt_lang))
        self.result = second_alignment

        logger.debug(second_pointers)
        logger.debug(cost)
        logger.debug(second_alignment)
    
    def print_sents(self, print_scores = True):
        # print(f"#SCORES: bertalign= | cos= | src/tgt=")
        i = 0
        for bead in (self.result):
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            to_print = src_line + "\n" + tgt_line + "\n"
            if print_scores:
                print_cos = "" if self.scores.get('cos', None) is None else f"cos= {self.scores['cos'][i]}| "
                to_print = to_print + f"#SCORES: bertalign= {self.scores['bertalign'][i]}| " + print_cos + f"src/tgt= {self.scores['length_ratio'][i]}" + "\n"
                i+=1
            logger.debug(to_print)
            
    def get_sents(self):
        src_lines = []
        tgt_lines = []
        for bead in (self.result):
            src_lines.append( self._get_line(bead[0], self.src_sents))
            tgt_lines.append( self._get_line(bead[1], self.tgt_sents))
        return src_lines, tgt_lines

    def store_sents(self, src_store_path, tgt_store_path):
        src_lines, tgt_lines = self.get_sents()
        with open(src_store_path, 'w', encoding = 'utf-8') as f:
            f.write('\n'.join(src_lines))

        with open(tgt_store_path, 'w', encoding = 'utf-8') as f:
            f.write('\n'.join(tgt_lines))
    
    def get_align_score(self):
        """alignment score"""
        return self.scores

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1]+1])
        return line
