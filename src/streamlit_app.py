import numpy as np
import streamlit as st
import pandas as pd
import os
from pyECLAT import ECLAT


def eclat(df, minsupport=0):
    eclat_instance = ECLAT(df)
    indexes, support = eclat_instance.fit(min_support=minsupport,
                                          min_combination=1,
                                          max_combination=5,
                                          separator=",",
                                          verbose=False)

    return indexes, support


def obtain_rules(frequent_itemlists):
    rules = list()
    for itemlists in frequent_itemlists:
        if len(itemlists) > 1:
            for item in itemlists:
                # appending tupel (itemlists/item, item)
                dif = itemlists.copy()
                dif.remove(item)
                dif.sort()
                rules.append((dif, [item]))

    return rules


def get_dict_index(rule):
    x, y = rule
    rule_index = [*x, *y]
    return get_string(rule_index)


def get_string(x: list):
    string = ""
    x.sort()
    for item in x:
        string += item + ','
    string = string.rstrip(',')
    return string


def evaluate(indexes, frequent_itemsets, num_itemsets):
    # indexes: dict: {"item1,item2": [indexes]}
    rules = obtain_rules(frequent_itemsets)

    results = []
    for rule in rules:
        dict_index = get_dict_index(rule)

        xandy = len(indexes[dict_index])
        x = len(indexes[get_string(rule[0])])
        y = len(indexes[get_string(rule[1])])
        # confidence
        conf = xandy / x

        # lift
        lift = conf / (y / num_itemsets)

        # support
        avg_support = (x + y) / num_itemsets

        # conviction
        conviction = np.Inf
        if conf != 1:
            conviction = (1 - avg_support) / (1 - conf)

        results.append([rule_to_string(rule), avg_support, conf, lift, conviction])

    return results


def rule_to_string(rule):
    x, y = rule
    return get_string(x) + " -> " + get_string(y)


if __name__ == '__main__':
    DATA = "../data/"
    path_norm = os.path.join(DATA, "norm_data.csv")

    min_support = st.slider("Min Support", 0, 100, 1)
    min_support /= 100
    df = pd.read_csv(path_norm, header=None)

    indexes, support = eclat(df, min_support)

    itemlists = []
    new_indexes = dict()
    for index in indexes.keys():
        itemlist = index.split(',')
        itemlists.append(itemlist)

        itemlist.sort()
        key = get_string(itemlist)
        new_indexes.update({key: indexes[index]})



    res = evaluate(new_indexes, itemlists, len(df.index))
    res_df = pd.DataFrame(res, columns=["Rules", "Avg Support", "Confidence", "Lift", "Conviction"])


    st.text(f"Itemsets: {len(df.index)}\tRules:{len(res)}")
    res_df