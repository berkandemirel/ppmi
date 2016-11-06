#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Training Part
# Written by bdemirel
# --------------------------------------------------------

import config
import cPickle, os, re, math
import numpy as np
from scipy import sparse
from sparsesvd import sparsesvd

def prepareData( rawDataPath, dataPath ):

    """ This function separates articles and clear data
    :param rawDataPath: path of the raw data
    :param dataPath: path of the clean data
    :return: none
    """

    if os.path.isfile(dataPath):
        print 'data is available...'
    else:
        fi = open( rawDataFile, 'r' )
        fo = open(dataPath, 'w')

        rawData = fi.readlines()

        for rawDataLine in rawData:

            if config.cfg.CONTEXT_SEPARATOR_F in rawDataLine:
                fo.write('\n')
            elif config.cfg.CONTEXT_SEPARATOR_S in rawDataLine:
                pass
            else:
                fo.write( re.sub(r'[^\w=]', ' ',rawDataLine).lower() )

        print 'data is cleaned...'

        fi.close()
        fo.close()

def getStopWordList(stopWordFile):

    """
    This function creates list of stop words
    :param stopWordFile: path of the file of the stop words
    :return: list of stop words.
    """

    lines = [line.rstrip('\n') for line in open(stopWordFile)]

    stopWords = {}
    for line in lines:
        stopWords[line] = 1

    print 'stopwords are generated...'
    return stopWords


def createVocabulary(dataFile, stopWords, vocabPath):

    """This function create a vocabulary that is needed to find word relatedness
    :param dataFile: file of the data
    :param stopWords: stop word list
    :param vocabPath: path to save vocabulary list
    :return: vocabulary list
    """

    try:
        vocab = cPickle.load(open(vocabPath, "rb"))
        print 'vocabulary is loaded...'
        return vocab

    except:

        vocab = {}

        '''
        Read raw data, merge upper and lower cases, sort them, apply unique operation
        and also apply threshold to skip words that occur less than 5 times
        '''
        process = os.popen("tr 'A-Z' 'a-z' < " + dataFile + \
                           "| tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -n -r | awk -v threshold=" + \
                           str(config.cfg.VOCAB_THRESHOLD) + " '$1 > threshold' ").readlines()

        # clean data, remove stopwords and create patricie-trie as a dictionary.
        totalWordCount = 0
        for word in process:
            # clean whitespaces and separate count and words
            wordAndCount = word.split('\n')[0].lstrip().split(' ')
            processedWord = wordAndCount[1]
            relatedCount = int(wordAndCount[0])

            if stopWords.get(processedWord) != 1:
                vocab.setdefault(processedWord, relatedCount)
                totalWordCount = totalWordCount + relatedCount
        vocab['totalWordCount'] = totalWordCount
        cPickle.dump(vocab, open(vocabPath, "wb"))
    print 'vocabulary is generated...'
    return vocab


def createCoOccurrenceMatrix(dataFile, vocab, vocabIndices, coOccurenceMatrixPath, coOccurrencePPMIPath):

    """This function creates word-word cooccurrence matrix

    :param dataFile: path of the data file
    :param vocab: vocabulary list
    :param vocabIndices: path of the indice list of the words that are located in the vocabulary
    :param coOccurenceMatrixPath: path of the raw coocurrence matrix( w/o PPMI)
    :param coOccurrencePPMIPath: path of the cooccurrence matrix that contains PPMI operations on each word-word relation
    :return: none
    """

    try:
        vocabIndices = cPickle.load(open(vocabIndicesPath, "rb"))
    except:
        vocabIndices = {}
        indiceCounter = 0
        for word,_ in vocab.items():
            vocabIndices[word] = indiceCounter
            indiceCounter += 1
        cPickle.dump(vocabIndices, open(vocabIndicesPath, "wb"))

    try:
        cooccurrences = cPickle.load(open(coOccurenceMatrixPath, "rb"))
        print 'co-occurrence matrix is loaded...'
    except:
        cooccurrences = sparse.lil_matrix((len(vocab), len(vocab)),dtype=np.float64)

        lines = [line.rstrip('\n') for line in open(dataFile)]

        lineCounter = 1
        for rawLine in lines:
            tokens = rawLine.strip().split()
            token_ids =[vocabIndices.get(word) for word in tokens]

            for currrInd, currId in enumerate(token_ids):

                if currId is not None:
                    searchWindow = token_ids[ max( currrInd -  config.cfg.SEARCH_WINDOW, 0 ): \
                        min( currrInd +  config.cfg.SEARCH_WINDOW +1, len(token_ids) ) ]

                    searchWindow = list(filter(None,searchWindow))

                    for _, validId in enumerate(searchWindow):
                        if int(currId) != int(validId):
                            cooccurrences[currId, validId] += 1

            print 'Completed: '+ str(lineCounter) + " / "+str(len(lines))
            lineCounter += 1
        cPickle.dump(cooccurrences, open(coOccurenceMatrixPath, "wb"))
        print 'co-occurrence matrix is generated...'

    #apply PPMI on co-occurrence matrix
    try:
        cooccurrences = cPickle.load(open(coOccurrencePPMIPath, "rb"))
        print 'co-occurrence matrix is loaded...'
    except:
        #sum of each row
        wordVectorSums = cooccurrences.sum(axis=1)
        wordVectorSums = np.array(wordVectorSums)

        lineCounter = 1
        for fW,_ in vocab.items():
            currVector = cooccurrences[vocabIndices.get(fW),:]
            nonZeroElemInCurrVector = currVector.nonzero() # find non-zero elements in related vector
            cPickle.dump(nonZeroElemInCurrVector, open('test2.p', "wb"))
            for sW in nonZeroElemInCurrVector[1]:
                currVector[0,sW] = max(math.log( (currVector[0,sW]/vocab['totalWordCount']) \
                        / ( (wordVectorSums[vocabIndices.get(fW)][0]/vocab['totalWordCount'])* \
                                    (  wordVectorSums[sW][0] /vocab['totalWordCount']) ), 2 ), 0 )
            cooccurrences[vocabIndices.get(fW),:] = currVector
            print 'PPMI values are generated ... '+ str(lineCounter) +'/'+ str(len(vocab))
            lineCounter += 1
        cPickle.dump(cooccurrences, open(coOccurrencePPMIPath, "wb"))

        del cooccurrences

def SVDOnCoOccurrenceMatrix( coOccurrencePPMIPath, coOccurrenceSVDPath ):

    """ This function apply Singular Value Decomposition operation on the cooccurrence matrix
    :param coOccurrencePPMIPath: path of the coocurrence matrix
    :param coOccurrenceSVDPath: path of the SVD components
    :return: none
    """

    cooccurrences = cPickle.load(open(coOccurrencePPMIPath, "rb"))
    smat = sparse.csc_matrix(cooccurrences)
    ut, s, vt = sparsesvd(smat, config.cfg.NUMBER_OF_FACTOR)
    print 'SVD is applied...'
    cPickle.dump(ut, open(coOccurrenceSVDPath, "wb"))


if __name__ == "__main__":

    stopWordFile = config.cfg.DATA_PATH + config.cfg.STOPWORDS_FILE
    rawDataFile = config.cfg.DATA_PATH + config.cfg.RAW_DATA_FILE
    dataFile = config.cfg.DATA_PATH + config.cfg.DATA_FILE

    vocabPath = config.cfg.DATA_PATH + config.cfg.VOCAB_OBJECT
    vocabIndicesPath = config.cfg.DATA_PATH + config.cfg.VOCAB_INDICES_OBJECT
    coOccurrenceMatrixPath = config.cfg.DATA_PATH + config.cfg.CO_OCCURRENCE_OBJECT
    coOccurrencePPMIPath = config.cfg.DATA_PATH + config.cfg.CO_OCCURRENCE_PPMI_OBJECT
    coOccurrenceSVDPath = config.cfg.DATA_PATH + config.cfg.CO_OCCURRENCE_PPMI_SVD_OBJECT

    prepareData( rawDataFile, dataFile ) # clean data

    stopWords = getStopWordList(stopWordFile)  # get stopword list

    vocab = createVocabulary(dataFile, stopWords, vocabPath)  # create dictionary

    createCoOccurrenceMatrix(dataFile, vocab, vocabIndicesPath, coOccurrenceMatrixPath, coOccurrencePPMIPath)  # create word-word co-occurrence matrix.

    #SVDOnCoOccurrenceMatrix(coOccurrenceMatrixPath, coOccurrenceSVDPath)
