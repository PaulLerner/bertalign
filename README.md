# Bertalign

An automatic mulitlingual sentence aligner.

Bertalign is designed to facilitate the construction of multilingual parallel corpora and translation memories, which have a wide range of applications in translation-related research such as corpus-based translation studies, contrastive linguistics, computer-assisted translation, translator education and machine translation.

## MaTOS fork
In the context of the [MaTOS project](https://anr-matos.github.io/), we have made a few changes to the original implementation of https://github.com/bfsujason/bertalign/

### Language identification
We replaced `googletrans` by `fasttext` to remain open-source.

### Sentence segmentation
We replaced `sentence-splitter` by `trankit` after a few trial and errors.

## Democratic Commons fork
In the context of the [Democratic Commons project](https://about.make.org/democratic-commons/landing-page) we built upon the MaTOS fork and added multi-parallel alignments.

We first compute every bi-parallel alignments from English then merge them.

```py
from bertalign.cli import main
# align one language first to cache english sentence segmentations
main("../data/linkedep/21-multi-europarl.csv", "fr")
languages = {
 'nl',
 'es',
 'lv',
 'pt',
 'it',
 'de',
 'da',
 'sv',
 'fi',
 'el',
 'pl',
 'et',
 'hu',
 'sk',
 'lt',
 'cs',
 'sl',
 'ro',
 'bg'}
# the rest can be done in parallel (e.g. using multiple sbatch calls)
for lang in languages:
    main("../data/linkedep/21-multi-europarl.csv", lang)
```

Once you get bi-parallel alignments, merge them with `python -m bertalign.multi`


## Installation

```sh
mamba create --name trankit python=3.10
mamba install -c conda-forge sentencepiece=0.2.0
mamba activate trankit
pip install -e .
```


## Approach

Bertalign uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers) to represent source and target sentences so that semantically similar sentences in different languages are mapped onto similar vector spaces. Then a two-step algorithm based on dynamic programming is performed: 1) Step 1 finds the 1-1 alignments for approximate anchor points; 2) Step 2 limits the search path to the anchor points and extracts all the valid alignments with 1-many, many-1 or many-to-many relations between the source and target sentences.

## Performance

According to our experiments, Bertalign achieves more accurate results on [Text+Berg](./text+berg), a publicly available German-French parallel corpus, than the traditional length-, dictionary-, or MT-based alignment methods as reported in [Thompson & Koehn (2019)](https://aclanthology.org/D19-1136/)

## Languages Supported

Alignment between 25 languages: Catalan (ca), Chinese (zh), Czech (cs), Danish (da), Dutch (nl), English(en), Finnish (fi), French (fr), German (de), Greek (el), Hungarian (hu), Icelandic (is), Italian (it), Lithuanian (lt), Latvain (lv), Norwegian (no), Polish (pl), Portuguese (pt), Romanian (ro), Russian (ru), Slovak (sk), Slovenian (sl), Spanish (es), Swedish (sv), and Turkish (tr).

FIXME: why not more given the languages supported by LaBSE?


## Citation

Lei Liu & Min Zhu. 2022. Bertalign: Improved word embedding-based sentence alignment for Chinese–English parallel corpora of literary texts, *Digital Scholarship in the Humanities*. [https://doi.org/10.1093/llc/fqac089](https://doi.org/10.1093/llc/fqac089).

## Funding

The work is supported by the MOE Foundation of Humanities and Social Sciences (Grant No. 17YJC740055).

## Licence

Bertalign is released under the [GNU General Public License v3.0](./LICENCE)

## Credits

##### Main Libraries

* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

* [faiss](https://github.com/facebookresearch/faiss)

* [sentence-splitter](https://github.com/mediacloud/sentence-splitter)

##### Other Sentence Aligners

* [Hunalign](http://mokk.bme.hu/en/resources/hunalign/)

* [Bleualign](https://github.com/rsennrich/Bleualign)

* [Vecalign](https://github.com/thompsonb/vecalign)

## Todo List

- Try the [CNN model](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) for sentence embeddings
* Develop a GUI for Windows users
