#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Config File
# Written by bdemirel
# --------------------------------------------------------

from easydict import EasyDict as edict

__C = edict()

cfg = __C

# Remove all words that occurs less than 3 times.
__C.VOCAB_THRESHOLD = 3

# Local word seach window. [ 1x5 + current_word + 1x5 ] = 1x11 word vector
__C.SEARCH_WINDOW = 5

# Path for data and stopword files
__C.DATA_PATH = '/data/'

# Name of vocabulary object
__C.VOCAB_OBJECT = 'vocab.p'

# Name of vocabulary indices' object
__C.VOCAB_INDICES_OBJECT = 'vocabIndices.p'

# Name of raw data file
__C.RAW_DATA_FILE = 'raw_data.txt'

# Name of data file
__C.DATA_FILE = 'data.txt'

# Name of the file of the stopwords
__C.STOPWORDS_FILE = 'stopwords.txt'

# Start flag of the context
__C.CONTEXT_SEPARATOR_S = '<article>'

# End flag of the context
__C.CONTEXT_SEPARATOR_F = '</article>'

#name of the co-occurence object
__C.CO_OCCURRENCE_OBJECT = 'coOccurrence.p'

#name of the co-occurence PPMI
__C.CO_OCCURRENCE_PPMI_OBJECT = 'coOccurrencePPMI.p'

#name of the post-processed co-occurence PPMI
__C.CO_OCCURRENCE_PPMI_SVD_OBJECT = 'coOccurrencePPMISVD.p'

#name of the test data
__C.TEST_DATA_FILE = 'toefl.txt'

#number of factors in SVD
__C.NUMBER_OF_FACTOR = 800

#determine assesment type(Use SVD or not)
__C.ASSESS_WITH_SVD = True
