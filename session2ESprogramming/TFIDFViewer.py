"""
.. module:: TFIDFViewer

TFIDFViewer
******

:Description: TFIDFViewer

    Receives two paths of files to compare (the paths have to be the ones used when indexing the files)

:Authors:
    bejar

:Version: 

:Date:  05/07/2017
"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import argparse

import numpy as np

__author__ = 'bejar'

def search_file_by_path(client, index, path):
    """
    Search for a file using its path

    :param path:
    :return:
    """
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id


def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())


def fill_lists(l1_tv, l2_tv, l1_df, l2_df):
    l1aux_tv = l1_tv
    l2aux_tv = l2_tv

    for i in range(len(l1aux_tv)):
        found = False
        for j in range(len(l2aux_tv)):
            if l1aux_tv[i][0] == l2aux_tv[j][0]:
                found = True

        if not found:
            l2_tv.append((l1_tv[i][0], 0))

    for i in range(len(l2aux_tv)):
        found = False
        for j in range(len(l1aux_tv)):
            if l2aux_tv[i][0] == l1aux_tv[j][0]:
                found = True

        if not found:
            l1_tv.append((l2_tv[i][0], 0))

    l1_tv.sort()
    l2_tv.sort()

    l1aux_df = l1_df
    l2aux_df = l2_df

    for i in range(len(l1aux_df)):
        found = False
        for j in range(len(l2aux_df)):
            if l1aux_df[i][0] == l2aux_df[j][0]:
                found = True

        if not found:
            l2_df.append((l1_df[i][0], l1_df[i][1]))

    # tanto l2_df como l1_df seran iguales asi que solo hace falta 1

    l2_df.sort()

    return l1_tv, l2_tv, l2_df

def toTFIDF(client, index, file_id1, file_id2):
    """
    Returns the term weights of a document  
    :param file:
    :return:
    """ 
    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv1, file_df1 = document_term_vector(client, index, file_id1) 
    file_tv2, file_df2 = document_term_vector(client, index, file_id2)

    file_tv1, file_tv2, file_df = fill_lists(file_tv1, file_tv2, file_df1, file_df2)

    print("d3 -> ")
    print(file_tv1)
    print("d4 -> ")
    print(file_tv2)
    print("df -> ")
    print(file_df1)
    
    max_freq1 = max([f for _, f in file_tv1]) 
    dcount = doc_count(client, index)   
    tfidfw1 = []
    
    print("frequencia maxima d3 = %d" % max_freq1)
    print("numero documentos = %d" % dcount)

    for (t, w),(_, df) in zip(file_tv1, file_df):
        #
        # Something happens here
        #
        # print("frequency " + str(w))
        tfdi = w / max_freq1
        idfi = np.log2(dcount/df)
        tfidfw1.append(tfdi*idfi)    
    
    max_freq2 = max([f for _, f in file_tv2]) 
    dcount = doc_count(client, index)   
    tfidfw2 = []

    print("frequencia maxima d4 = %d" % max_freq2)
    print("numero documentos = %d" % dcount)

    for (t, w),(_, df) in zip(file_tv2, file_df):
        #
        # Something happens here
        #
        # print("frequency " + str(w))
        # print("df word " + str(df))
        tfdi = w / max_freq2
        idfi = np.log2(dcount/df)
        print("tfdi = " + str(tfdi) + " idfi = " + str(idfi) + " idfi*tfdi = " + str(tfdi*idfi))
        tfidfw2.append(tfdi*idfi)  

    print("d3 weights -> ")
    print(tfidfw1)
    print("d4 weights -> ")
    print(tfidfw2)
    
    return tfidfw1, tfidfw2


def print_term_weigth_vector(twv):
    """
    Prints the term vector and the correspondig weights
    :param twv:
    :return:
    """
    #
    # Program something here
    #
    print(twv)
    
def normal(v):
    """
    Returns the normal of a vector
    :param v:
    :return:
    """

    print("normal -> ")
    print(np.sqrt(np.dot(v, v)))
    return np.sqrt(np.dot(v, v))

def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    #
    # Program something here
    #
    #return None
    return np.divide(tw, normal(tw))


def cosine_similarity(tw1, tw2):
    """
    Computes the cosine similarity between two weight vectors, terms are alphabetically ordered
    :param tw1:
    :param tw2:
    :return:
    """
    #
    # Program something here
    #
    
    return np.dot(normalize(tw1), normalize(tw2))

def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, required=True, help='Index to search')
    parser.add_argument('--files', default=None, required=True, nargs=2, help='Paths of the files to compare')
    parser.add_argument('--print', default=False, action='store_true', help='Print TFIDF vectors')

    args = parser.parse_args()


    index = args.index

    file1 = args.files[0]
    file2 = args.files[1]

    client = Elasticsearch(timeout=1000)

    try:

        # Get the files ids
        file1_id = search_file_by_path(client, index, file1)
        file2_id = search_file_by_path(client, index, file2)


        # Compute the TF-IDF vectors
        # file1_tw = toTFIDF(client, index, file1_id)
        # file2_tw = toTFIDF(client, index, file2_id)

        file1_tw, file2_tw = toTFIDF(client, index, file1_id, file2_id)

        if args.print:
            print(f'TFIDF FILE {file1}')
            print_term_weigth_vector(file1_tw)
            print ('---------------------')
            print(f'TFIDF FILE {file2}')
            print_term_weigth_vector(file2_tw)
            print ('---------------------')

        print(f"Similarity = {cosine_similarity(file1_tw, file2_tw):3.5f}")

    except NotFoundError:
        print(f'Index {index} does not exists')

