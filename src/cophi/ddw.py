import pandas as pd



# ausgabe eines token anhand seiner id
def get_token(df, tokenid):
    return df[df['TokenId'] == int(tokenid)]



# ausgabe aller dependency-tree-children, tokenid = id of parent
def get_children(df, tokenid):
    return df[df['DependencyHead'] == str(tokenid)]



# nur children mit bestimmtem postag ausgeben, tokenid = id of parent
def get_children_pos(df, tokenid, postag):
    df = df[df['DependencyHead'] == str(tokenid)]
    return df[df['CPOS'] == postag]



# checkt ob tokenid (= id of parent) -> je child mit postag1 -> child mit postag2
def walk_tree_pos2(df, tokenid, postag1, postag2):
    result = {}

    ch1 = get_children_pos(df, tokenid, postag1)                 # given tokenid, get children matching postag1

    for index, child1 in ch1.iterrows():                         # for each of those
        ch2 = get_children_pos(df, child1['TokenId'], postag2)   # get children matching postag2

        if not ch2.empty:
            for index, child2 in ch2.iterrows():
                result[child1['TokenId']] = child2      # also return id of corresponding head noun

            return result
    return None



# checkt ob tokenid -> je child mit postag1 -> je child mit postag2 -> child mit postag3
def walk_tree_pos3(df, tokenid, postag1, postag2, postag3):
    result = {}

    ch1 = get_children_pos(df, tokenid, postag1)                 # given tokenid, get children matching postag1

    for index, child1 in ch1.iterrows():                         # for each of those
        ch2 = get_children_pos(df, child1['TokenId'], postag2)   # get children matching postag2

        if not ch2.empty:
            for index, child2 in ch2.iterrows():
                ch3 = get_children_pos(df, child2['TokenId'], postag3)

                if not ch3.empty:
                    for index, child3 in ch3.iterrows():
                        result[child1['TokenId']] = child3      # also return id of corresponding head noun

                    return result
    return None

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

"""
aus den dependency relations einen graph bauen
- satzweise, für jedes token: ein edge aus tok_id:DependencyHead
- labels: tokens für nodes, DependencyRelation für edges
- edges: vgl. http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/kanten.html

http://guitarpenguin.is-programmer.com/posts/44818.html
https://networkx.github.io/documentation/latest/reference/drawing.html
https://networkx.github.io/documentation/latest/reference/functions.html
http://networkx.lanl.gov/reference/classes.digraph.html
"""

df = pd.read_csv("EffiBriestKurz.txt.csv", sep="\t")
sent_max = df["SentenceId"].max()
trees = []


### READER

# construct dependency trees

for sent_id in range(sent_max+1):                           # iterate through sentences
    sentence = df[df['SentenceId'] == sent_id]              # return rows corresponding to sent_id

    dg = nx.DiGraph()                                       # a new directed graph

    for row in sentence.iterrows():
        tok_id = str(row[0])                                # current token id
        tok = row[1].get("Token")                           # current token
        head_id = row[1].get("DependencyHead")              # token head id
        rel = row[1].get("DependencyRelation")              # dependency relation

        if head_id.isdigit() == True:
            head = df.iloc[int(head_id), 6]                 # get head token
        else:
            head = "ROOT"                                   # or mark as root

        dg.add_node(tok, id=tok_id)                         # save token id as node attribute
        dg.add_node(head, id=head_id)
        dg.add_edge(head, tok, rel=rel)                     # add edge to graph

    trees.append(dg)                                        # save digraph


### QUERIES

# query for relation type

query1 = 'NK'

print("All edges of type", query1, ":\n")

for tree in trees:
    rel = dict([((u, v), d['rel']) for u, v, d in tree.edges(data=True)])

    for nodes, r in rel.items():
        if r == query1:
            print(r, nodes)

print("\n")


# query for token id

query2 = 921

for tree in trees:
    id = dict([(u, d['id']) for u, d in tree.nodes(data=True)])

    for token, i in id.items():
        if i == str(query2):
            print("Token", i, ":", token)
            print(nx.info(tree, token))

print("\n")


# query for token

query2 = "von"

for tree in trees:
    id = dict([(u, d['id']) for u, d in tree.nodes(data=True)])

    for token, i in id.items():
        if token == query2:
            print("Token", i, ":", token)
            print(nx.info(tree, token))

print("\n")


# visualise a sentence

query3 = 33

dg = trees[query3]
print("Dependency Tree for Sentence", query3, ":")
print(nx.info(dg))
print("Token IDs:", nx.get_node_attributes(dg, 'id'))       # mapping back into dataframe

pos = nx.spectral_layout(dg)
edge_labels = dict([((u, v), d['rel']) for u, v, d in dg.edges(data=True)])
nx.draw_networkx_edge_labels(dg, pos, edge_labels=edge_labels)
nx.draw_networkx(dg, pos, with_labels=True)

plt.axis('off')
plt.show()

import pandas as pd
from nltk.tree import ParentedTree

"""
from: http://conll.cemantix.org/2012/data.html
    "This is the bracketed structure broken before the first open parenthesis in the parse,
    and the word/part-of-speech leaf replaced with a *. The full parse can be created by
    substituting the asterix with the "([pos] [word])" string (or leaf) and concatenating
    the items in the rows of that column."
"""
"""
alle noun phrases inkl. pos-tags auslesen
-> mit nltk.tree.Tree und mapping zurück ins dataframe: satzweise die bäume wieder zusammensetzen als (LABEL TOKEN#TOKID)
https://stackoverflow.com/questions/25815002/nltk-tree-data-structure-finding-a-node-its-parent-or-children
https://stackoverflow.com/questions/14841997/how-to-navigate-a-nltk-tree-tree
http://www.nltk.org/howto/tree.html
http://nbviewer.ipython.org/github/gmonce/nltk_parsing/blob/master/1.%20NLTK%20Syntax%20Trees.ipynb
http://www.mit.edu/~6.863/spring2011/labs/nltk-tree-pages.pdf
"""

sign = '#'          # token#token_id delimiter for use inside tree objects

df = pd.read_csv("EffiBriestKurz.txt.csv", sep="\t")
sent_max = df["SentenceId"].max()
trees = []


### READER

# construct syntax trees

for sent_id in range(sent_max+1):                                       # iterate through sentences
    sentence = df[df['SentenceId'] == sent_id]                          # return rows corresponding to sent_id
    sent_string = tmp_string = tmp_string2 = ""

    for row in sentence.iterrows():
        tok_id = str(row[0])                                            # current token id
        tok = row[1].get("Token")                                       # current token
        tree_frag = row[1].get("SyntaxTree").strip("*")                 # current syntax tree fragment

        """
        if "*)" in sent_string:                                         # TODO: possible bug in csv tree writer
            if tmp_string:
                tmp_string2 = sent_string.replace("*)", "")             # we ran into "*)" a second time
                sent_string = tmp_string2
            else:
                tmp_string = sent_string.replace("*)", "")
                sent_string = tmp_string
        """

        if tree_frag.startswith("("):                                   # reconstruct tree + save token id
            sent_string += tree_frag + " " + tok + sign + tok_id + " "  # beginning of fragment
        elif not ")" in tree_frag:
            sent_string += tree_frag + " " + tok + sign + tok_id + " "  # middle
        else:                                                           # end
            """
            if tmp_string:                                              # TODO: possible bug in csv tree writer
                sent_string += " " + tok + sign + tok_id + tree_frag + ") "
                tmp_string = ""
            elif tmp_string2:
                sent_string += " " + tok + sign + tok_id + tree_frag + ")) "
                tmp_string2 = ""
            else:
            """
            sent_string += " " + tok + sign + tok_id + tree_frag + " "

    trees.append(sent_string)                                           # save reconstruction


### QUERIES

# query for constituent type

query1 = 'NP'

print("All constituents of type", query1, "+ POS-Tags:\n")

for string in trees:
    tree = ParentedTree.fromstring(string)                              # read string into tree object

    for subtree in tree.subtrees(filter=lambda t: t.label() == query1): # query subtrees
        print(subtree, "\n")

        for leaf in subtree.leaves():
            t, i = leaf.split(sign)                                     # map back to dataframe using token id
            pos = df.iloc[int(i), 9]                                    # and get a pos-tag
            print("{0}: {1} {2}".format(i, t, pos))
        print("\n")


# query for token or token id

query2 = 'Effi'     # '#921'

print("Token", query2, "found in:\n")

for string in trees:
    tree = ParentedTree.fromstring(string)                              # read string into tree object

    for subtree in tree.subtrees():
        for leaf in subtree.leaves():
            if query2 in leaf: print(subtree)                           # subtrees containing query


# visualise a sentence

query3 = 33

tree = ParentedTree.fromstring(trees[query3])
tree.draw()

