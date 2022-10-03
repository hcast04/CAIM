import numpy as np


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
    # return None
    return np.divide(tw, np.sqrt(np.dot(tw, tw)))


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

    return np.dot(tw1, tw2) / (normalize(tw1) * normalize(tw2))


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

    # seguramente el siguiente for sea equivalente a l2_df = l1_df

    for i in range(len(l2aux_df)):
        found = False
        for j in range(len(l1aux_df)):
            if l2aux_df[i][0] == l1aux_df[j][0]:
                found = True

        if not found:
            l1_df.append((l2_df[i][0], l2_df[i][1]))

    l1_df.sort()
    l2_df.sort()

    return l1_tv, l2_tv, l1_df, l2_df


if __name__ == '__main__':
    tw1 = [1.81, 0.41, 0.41, 0, 0.07, 0]
    tw2 = [0, 0, 0.61, 1.22, 0.11, 3.61]

    l1_tv = [('five', 3), ('four', 1), ('one', 1), ('three', 1)]
    l2_tv = [('one', 1), ('six', 2), ('three', 1), ('two', 4)]
    l1_df = [('five', 2), ('four', 3), ('one', 3), ('three', 6)]
    l2_df = [('one', 3), ('six', 3), ('three', 6), ('two', 2)]

    l1_tv, l2_tv, l1_df, l2_df = fill_lists(l1_tv, l2_tv, l1_df, l2_df)

    print(l1_tv)
    print(l2_tv)
    print(l1_df)
    print(l2_df)
    
    print(normalize([3, 4]))
