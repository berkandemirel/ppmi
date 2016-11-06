#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Test Part
# Written by bdemirel
# --------------------------------------------------------

import config
import cPickle, os, re
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cosine
from scipy import linalg, mat, dot

def testModel( dataFile, vocabIndicesPath, coOccurrencePPMIPath, assessmentType ):

    """Calculate accuracy of the prepared model
    :param dataFile: path of the test words
    :param vocabIndicesPath: path of the indice list of the words that are located in the vocabulary
    :param coOccurrencePPMIPath: path of the cooccurrence matrix that contains PPMI operations on each word-word relations
    :param assessmentType: type of the assessment( with SVD(true) or pure PPMI matrix(False) )
    :return: none
    """

    vocab = cPickle.load(open(vocabIndicesPath, "rb"))
    cooccurrence = cPickle.load(open(coOccurrencePPMIPath, "rb"))

    fi = open( dataFile, 'r' )

    correctAnswerCount = 0
    numberOfQuestion = 0

    for question in fi:
        words = question.split('\r\n')[0].split('|')

        givenWord = words[0].rsplit()[0]
        correctAnswer = words[1].rsplit()[0]
        wrongAnswers = words[2:]

        if assessmentType:
            costValueCA = cosDistanceBetweenSVDVectors( givenWord, correctAnswer , vocab, cooccurrence)
        else:
            costValueCA = cosDistanceBetweenVectors( givenWord, correctAnswer , vocab, cooccurrence)

        costValueWA = 0
        for wrAnswer in wrongAnswers:
            wrAnswer = wrAnswer.rsplit()[0]
            if assessmentType:
                costValueWA_ = cosDistanceBetweenSVDVectors( givenWord, wrAnswer , vocab, cooccurrence)
            else:
                costValueWA_ = cosDistanceBetweenVectors( givenWord, wrAnswer , vocab, cooccurrence)

            costValueWA = max( costValueWA, costValueWA_)

        if costValueCA > costValueWA:
            correctAnswerCount += 1
            print 'Correct Answer! Question Number: '+ str(numberOfQuestion+1)
        numberOfQuestion += 1
    print 'total accuracy: %'+str((float(correctAnswerCount)/numberOfQuestion)*100)

def cosDistanceBetweenVectors( wordA, wordB, vocab, cooccurrence  ):

    """ Calculate cosine distance between two vectors.
    Note: These vectors have huge dimensions, so we applied some pruning operations before calculation

    :param wordA: name of the word A
    :param wordB: name of the word B
    :param vocab: list of the vocabulary words
    :param cooccurrence: occurrence matrix ( with SVD )
    :return:
    """

    try:
        vectorA = cooccurrence[vocab.get(wordA),:]
        vectorB = cooccurrence[vocab.get(wordB),:]
        intersectionBetweenVectors = set(np.array(vectorA.nonzero()[1])).intersection(np.array(vectorB.nonzero()[1]))

        #dot product of vectors / L2-Norms of vectors
        dotProduct = 0
        for point in intersectionBetweenVectors:
            dotProduct += vectorA[0,point]*vectorB[0,point]

        return dotProduct / (np.sqrt(sum(np.square(vectorA.data[0])))+np.sqrt(sum(np.square(vectorB.data[0]))))
    except:
        return 0

def cosDistanceBetweenSVDVectors( wordA, wordB, vocab, cooccurrence  ):

    """ Calculate cosine distance between two vectors
    :param wordA: name of the word A
    :param wordB: name of the word B
    :param vocab: list of the vocabulary words
    :param cooccurrence: occurrence matrix ( with SVD )
    :return:
    """

    try:
        vectorA = cooccurrence[:,vocab.get(wordA)]
        vectorB = cooccurrence[:,vocab.get(wordB)]

        return dot(vectorA,vectorB)/(np.sqrt(sum(np.square(np.transpose(vectorA))))*np.sqrt(sum(np.square(np.transpose(vectorB)))))
    except:
        return 0

if __name__ == "__main__":
    dataFile = config.cfg.DATA_PATH + config.cfg.TEST_DATA_FILE
    vocabIndicesPath = config.cfg.DATA_PATH + config.cfg.VOCAB_INDICES_OBJECT
    assessmentType = config.cfg.ASSESS_WITH_SVD

    if assessmentType:
        coOccurrencePath = config.cfg.DATA_PATH + config.cfg.CO_OCCURRENCE_PPMI_SVD_OBJECT
    else:
        coOccurrencePath = config.cfg.DATA_PATH + config.cfg.CO_OCCURRENCE_PPMI_OBJECT

    testModel( dataFile, vocabIndicesPath, coOccurrencePath, assessmentType )
