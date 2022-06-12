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
                if len(dif) > 1:
                    rules.append(([item], dif))

    return rules


def check_for_duplicates(rules):
    check_dict = {}
    for rule in rules:
        x = get_string(rule[0])
        y = get_string(rule[1])
        if x in check_dict.keys():
            if y in check_dict[x]:
                raise Exception("Duplicate found!")
            else:
                check_dict[x].append(y)
        else:
            check_dict.update({x: [y]})

def get_string(x: list):
    string = ""
    x.sort()
    for item in x:
        string += item + ','
    string = string.rstrip(',')
    return string

def get_indexes_and_itemsets(indexes):
    itemlists = []
    new_indexes = dict()
    for index in indexes.keys():
        itemlist = index.split(',')
        itemlists.append(itemlist)

        itemlist.sort()
        key = get_string(itemlist)
        new_indexes.update({key: indexes[index]})

    return new_indexes, itemlists

def evaluate(indexes, frequent_itemsets, num_itemsets):
    # indexes: dict: {"item1,item2": [indexes]}
    rules = obtain_rules(frequent_itemsets)

    results = []
    for rule in rules:
        x_set = set(indexes[get_string(rule[0])])
        y_set = set(indexes[get_string(rule[1])])
        x = len(x_set)
        y = len(y_set)
        xandy = len(x_set.intersection(y_set))

        # support
        avg_support = xandy / num_itemsets

        # confidence
        conf = xandy / x

        # lift
        lift = avg_support / ((y / num_itemsets) * (x / num_itemsets))

        # conviction
        conviction = np.Inf
        if conf != 1:
            conviction = (1 - (y / num_itemsets)) / (1 - conf)

        results.append([rule_to_string(rule), avg_support, conf, lift, conviction])

    return results

def rule_to_string(rule):
    x, y = rule
    return get_string(x) + " -> " + get_string(y)


if __name__ == '__main__':
    DATA = "../data/"
    path_norm = os.path.join(DATA, "norm_data.csv")
    path_schiz = os.path.join(DATA, "schiz_data.csv")

    st.markdown("## Choose minimum support")
    min_support = st.slider("Minimum Support", 0, 100, 1)
    min_support /= 100

    # Normal data
    df = pd.read_csv(path_norm, header=None)
    indexes, _ = eclat(df, min_support)
    new_indexes, itemlists = get_indexes_and_itemsets(indexes)

    res = evaluate(new_indexes, itemlists, len(df.index))
    res_df_norm = pd.DataFrame(res, columns=["Rules", "Avg Support", "Confidence", "Lift", "Conviction"])

    st.markdown("## Normal Adolesence")
    st.text(f"Itemsets: {len(df.index)}\tRules:{len(res)}")
    res_df_norm

    # Schizo data
    df = pd.read_csv(path_schiz, header=None)
    indexes, _ = eclat(df, min_support)
    new_indexes, itemlists = get_indexes_and_itemsets(indexes)

    res = evaluate(new_indexes, itemlists, len(df.index))
    res_df_schiz = pd.DataFrame(res, columns=["Rules", "Avg Support", "Confidence", "Lift", "Conviction"])

    st.markdown("## Schizophrenia Adolesence")
    st.text(f"Itemsets: {len(df.index)}\tRules:{len(res)}")
    res_df_schiz

    st.markdown("## Rules Appearing only in Schizophrenia Data")
    view_only_schiz = res_df_schiz[~res_df_schiz["Rules"].isin(res_df_norm["Rules"])]
    st.text(f"Rules:{len(view_only_schiz)}")
    view_only_schiz