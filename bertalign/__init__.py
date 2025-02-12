"""
Bertalign initialization
"""

__author__ = "Jason (bfsujason@163.com)"
__version__ = "1.1.0"

from bertalign.encoder import Encoder

# See other cross-lingual embedding models at
# https://www.sbert.net/docs/pretrained_models.html

# model_name = "LaBSE"
model_name = "/gpfsdswork/dataset/HuggingFace_Models/sentence-transformers/LaBSE"
model = Encoder(model_name)

from bertalign.aligner import Bertalign
