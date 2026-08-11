"""
discriminative_lexicon_model - Discriminative Lexicon Model in Python
=====================================================================

*discriminative_lexicon_model* provides some matrix-multiplications and accuracy-calculations in the framework of Discriminative Lexicon Model (Baayen et al., 2019).

"""

import os
import sys
import multiprocessing as mp
try:
    from importlib.metadata import requires
except ModuleNotFoundError:
    requires = None
try:
    from packaging.requirements import Requirement
except ModuleNotFoundError:
    Requirement = None

try:
    from . import mapping
    from . import measures
    from . import performance
    from . import ldl
    from . import embedding
    from .mapping import *
    from .measures import *
    from .performance import *
    from .ldl import *
    from .embedding import *
except ModuleNotFoundError:
    pass

__all__ = ["mapping", "measures", "performance", "ldl", "embedding"]
for _mod in (mapping, measures, performance, ldl, embedding):
    __all__ += getattr(_mod, "__all__", [])
del _mod

