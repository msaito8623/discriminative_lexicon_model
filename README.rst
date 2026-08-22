=====================================================================
discriminative_lexicon_model - Discriminative Lexicon Model in Python
=====================================================================

.. image:: https://zenodo.org/badge/428379187.svg
  :target: https://zenodo.org/doi/10.5281/zenodo.13771599

*discriminative_lexicon_model* is a Python implementation of the Discriminative Lexicon Model (DLM; Baayen et al., 2019), a unified computational model of the mental lexicon that handles both comprehension (form-to-meaning mapping) and production (meaning-to-form mapping) using linear discriminative learning.


Overview
========

DLM consists of six core matrices:

- **C** (word-form matrix): cue representations of words (e.g., triphones)
- **S** (word-meaning matrix): semantic vectors for each word
- **F** (comprehension weights): maps forms to meanings, estimated so that **CF** approximates **S**
- **G** (production weights): maps meanings to forms, estimated so that **SG** approximates **C**
- **S-hat**: predicted meanings (**CF**), the model's comprehension output
- **C-hat**: predicted forms (**SG**), the model's production output


Installation
============

*discriminative_lexicon_model* is available on `PyPI <https://pypi.org/project/discriminative-lexicon-model/>`_:

.. code:: bash

    pip install discriminative_lexicon_model

Note that the PyPI version may be outdated. For the latest version, install directly from GitHub:

.. code:: bash

    pip install git+https://github.com/quantling/discriminative_lexicon_model

Or clone the repository first and install locally:

.. code:: bash

    git clone https://github.com/quantling/discriminative_lexicon_model
    cd discriminative_lexicon_model
    pip install .


Quick start
===========

.. code-block:: python

    import discriminative_lexicon_model as dlm
    import pandas as pd

    # Define words and semantic vectors
    words = ['walk', 'walked', 'walks']
    sems = pd.DataFrame(
        {'WALK': [1, 1, 1], 'Present': [1, 0, 1],
         'Past': [0, 1, 0], 'ThirdPerson': [0, 0, 1]},
        index=words
    )

    # Build the full model in one step
    mdl = dlm.LDL(words, sems, allmatrices=True)

    # Check comprehension and production accuracy
    mdl.accuracy(print_output=True)

    # Examine predictions for individual words
    dlm.predict_df(pred=mdl.chat, gold=mdl.cmat)
    #       Word    Pred  Correct
    # 0     walk    walk     True
    # 1   walked  walked     True
    # 2    walks   walks     True

    # Compute linguistic measures
    round(dlm.semantic_support('walked', 'ed#', mdl.chat), 3)   # 1.0
    round(dlm.functional_load('ed#', mdl.fmat, 'walked', mdl.smat), 3)  # 0.853

You can also build the model step by step:

.. code-block:: python

    mdl = dlm.LDL()
    mdl.gen_cmat(words)                # C-matrix from trigrams
    mdl.gen_smat(sems)                 # S-matrix from semantic DataFrame
    mdl.gen_fmat()                     # F = pinv(t(C)C) t(C)S
    mdl.gen_gmat()                     # G = pinv(t(S)S) t(S)C
    mdl.gen_shat()                     # S-hat = CF
    mdl.gen_chat()                     # C-hat = SG

Matrices can be saved to and loaded from compressed CSV files:

.. code-block:: python

    # Save all matrices to a directory
    mdl.save_matrices('path/to/directory')

    # Load matrices back into a new model
    mdl2 = dlm.LDL()
    mdl2.load_matrices('path/to/directory')

Individual matrices can also be saved and loaded directly:

.. code-block:: python

    dlm.save_mat(mdl.cmat, 'cmat.csv.gz')
    cmat = dlm.load_mat('cmat.csv.gz')


Features
========

- Full DLM pipeline: cue extraction, matrix estimation, prediction, and evaluation
- Endstate-learning, incremental learning, frequency-weighted learning
- Incremental production via the ``produce`` algorithm (iterative cue selection with validity constraints)
- Path weaving algorithm: selecting and ordering predicted form cues into concatenated word forms by synthesis-by-analysis
- Linguistic measures: semantic support, functional load, production accuracy, uncertainty, vector length
- Optional GPU acceleration via PyTorch for production and incremental learning
- Semantic vectors from fastText embeddings or custom DataFrames
- All matrices stored as ``xarray.DataArray`` with labeled dimensions
- Matrix I/O (save/load as CSV)


Modules
=======

- ``ldl``: ``LDL`` class that bundles all matrices and methods into a single model object
- ``mapping``: core functions for cue extraction, matrix generation, production, and incremental learning
- ``measures``: linguistic measures (semantic support, functional load, uncertainty, etc.)
- ``path_weaving``: positional learning, path finding, and synthesis-by-analysis for production
- ``performance``: prediction accuracy and evaluation utilities


Dependencies
============

- Python >= 3.11
- numpy, scipy, pandas, xarray, netCDF4, tqdm, fasttext

Optional:

- PyTorch (for GPU-accelerated production and incremental learning)


Documentation
=============

Full documentation is hosted on `Read the Docs <https://discriminative-lexicon-model.readthedocs.io/en/latest/>`_.


Citation
========

If you use this package in your research, please cite:

    Baayen, R. H., Chuang, Y.-Y., Shafaei-Bajestan, E., & Blevins, J. P. (2019). The discriminative lexicon: A unified computational model for the lexicon and lexical processing in comprehension and production grounded not in (de)composition but in linear discriminative learning. *Complexity*, 2019, 1--39.


Authors and Contributors
========================

*discriminative_lexicon_model* is being developed and maintained by `Motoki Saito <https://github.com/msaito8623>`_.


License
=======

MIT
