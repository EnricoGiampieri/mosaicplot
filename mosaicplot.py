# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 13:23:49 2016

@author: enrico
"""

# %%
import pandas as pd
import pylab as plt
import numpy as np
import itertools as it
from collections import namedtuple
from functools import partial
import operator as op
import seaborn
import toolz
# %%
"""
how the data is expected:
Series with hierachical index and the counts for each category.
The lists of categories should be complete.

Other columns should be used to include informations about colors and labels.

l'ordine delle colonne degli indici è quello che determina
l'ordine di divisione.
"""

"""
SANE DEFAULTS FOR THE MOSAIC

first division: space
second: hue
third: saturation
fourth: value
fifth: texture

the first and second dimension will have tick labels.
from the third on they should go with legend labels
"""

def _get_fig(ax, **kwargs):
    if ax is None:
        return plt.subplots(1, 1, **kwargs)
    else:
        return ax.get_figure(), ax

# %%
from io import StringIO
import pandas as pd
data = """
PersonID  Married  No_of_Children    Sex
1         yes      0                 male
2         no       0                 female
3         no       1                 male
4         yes      1                 male
5         no       1                 female
6         no       2                 female
7         no       1                 male
8         no       2                 male
9         no       2                 male
10        no       1                 male
11        no       0                 female
"""
data = StringIO(data)
data.seek(0)

data = pd.read_csv(data, sep=' ', skipinitialspace=True, index_col='PersonID')
data.info()
# %%
"""
genero la tabella dei conteggi
"""
data_ = data.copy()
data_['Intercept'] = 1
info_df_ = data_.groupby(['Sex', 'No_of_Children', 'Married'])['Intercept'].count()
print(info_df_)

# %%
url = "http://facweb.cs.depaul.edu/mobasher/classes/csc478/Data/titanic-trimmed.csv"
titanic = pd.read_csv(url)
titanic.head(10)
titanic['Intercept'] = 1
titanic['survived'] = titanic['survived'].replace({0:'No', 1:'Yes'})
info_df_ = titanic.groupby(['pclass', 'sex', 'survived'])['Intercept'].count()


# %%
"""
faccio in modo che tutte le ctegorie siano piene.
in teoria i conteggi vanno
"""
info_df = info_df_.copy()
p = pd.MultiIndex.from_product(info_df.index.levels, names=info_df.index.names)
info_df = info_df.reindex(p, fill_value=0)
prior = 0.5*0
info_df += prior
levels = info_df.index.names
labels = list(map(list, info_df.index.levels))

# this check is to ensure the proper behavior later one, where we need this trick
assert info_df.equals(info_df[tuple()])
# pick the sublist fromt he list given of elements that share the same
# root with the given one
sameRoot = lambda i, seq: [j for j in seq if (j[0][:-1]==i[0][:-1])]

# join the list of labels into a sequence of sequence of lists
# from [['a', 'b'], [1, 0]]
# to   [[['a', 'b']], [['a', 'b'], [1, 0]]]
hierachical_labels = it.accumulate(zip(labels))
# now use those list and create the hierarchical combinations
# [['a'], ['b'], ['a', 0], ['a', 1], ['b', 0], 'b', 1]
hierachical_products = [it.product(*i) for i in hierachical_labels]
hierachical_chain = it.chain.from_iterable(hierachical_products)
# now for each one I get the counts for the category and for the root
# using the trick that an empty tuple as index gives the whole series
# if the element is empty, just drop it
hierachical_counts = [(i, info_df[i[:-1]].sum(), info_df[i].sum())
                      for i in hierachical_chain if info_df[i].sum()]
#hierachical_counts = [i for i in hierachical_counts if i[2]]
# add to the info how many other non null groups are in the same root group
hierachical_counts_2 = [(*i, len(sameRoot(i, hierachical_counts)))
                        for i in hierachical_counts]
# now evaluate the ratio, but on empty root groups leave it as 0
# instead of rising error
hierachical_ratio = [(i[0], i[2]/i[1] if i[1] else 0.0, i[3])
                     for i in hierachical_counts_2]
# now add the position in the root group
hierachical_index = [(*i, sameRoot(i, hierachical_ratio).index(i))
                     for i in hierachical_ratio]

# now pick from all the previous groups in the same root the sum of their lenght
# (0 if it is the first) and append it as the starting point of the interval lenght
hierachical_extent = [(*i, sum(j[1] for j in sameRoot(i, hierachical_index)[:i[3]]))
                      for i in hierachical_index]
# now just put the in order:
# group, starting position of the interval, lenght of the interval, size of the group, order in the group
hierachical_extent_2 = [(i[0], i[4], i[1], i[2], i[3]) for i in hierachical_extent]
# now the spacing. I scale starting points and lenght based on the number of elements
# and the size of the scaling
spacing = [0.05, 0.01] + [0]*(len(labels)-2)
hierachical_spacing = [(i[0],
                        i[1]/(1+spacing[len(i[0])-1]*(i[3]-1)),
                        i[2]/(1+spacing[len(i[0])-1]*(i[3]-1)),
                        i[3],
                        i[4])
                       for i in hierachical_extent_2]
# now I have to move the starting point accordingly to put everything back in place
hierachical_spacing_2 = [(i[0],
                          i[1]+i[4]*spacing[len(i[0])-1]/(1+spacing[len(i[0])-1]*(i[3]-1)),
                          i[2],
                          i[3],
                          i[4])
                         for i in hierachical_spacing]
# group, x, width, y, height, size, rank, parity of the group (horizontal or vertical)
hierachical_spacing_3 = [(*i, ((-1)**len(i[0]))==-1) for i in hierachical_spacing_2]

# we add a first element as a basis for the others, to be removed soon after
final_division = [(tuple(), 0, 1, 0, 1)]
for i in hierachical_spacing_3:
    a = [j for j in final_division if (j[0]==i[0][:-1])]
    assert len(a)==1
    _, x, w, y, h = a[0]
    code, ix, iw, _, _, parity = i
    if parity: # divide on the horizontal axis
        new_x = x + ix*w
        new_w = iw*w
        new_y = y
        new_h = h
    else: # divide on the vertical axis
        new_x = x
        new_w = w
        new_y = y + ix*h
        new_h = iw*h
    new_i = code, new_x, new_w, new_y, new_h
    final_division.append(new_i)
final_division = final_division[1:]
assert len(final_division) == len(hierachical_spacing_3)

rectStruct = namedtuple('RectStruct', ['code', 'x', 'width', 'y', 'height'])
final_division_named = [rectStruct(*i) for i in final_division]

# %%
current_palette = seaborn.color_palette()*5
#seaborn.palplot(current_palette)

colori = list(it.chain.from_iterable(seaborn.dark_palette(current_palette[i], len(labels[2])+1)[::-1][:-1]
                                for i in range(len(labels[1]))))
#seaborn.palplot(colori)
etichette = list(it.product(*labels[1:]))
etichette2color = dict(zip(etichette, colori))

max_len = max(len(k.code) for k in final_division_named)
with seaborn.axes_style("white"):
    fig, ax = plt.subplots(1, 1)
    ax.grid(False)
    for (k, x, w, y, h) in final_division_named:
        if len(k)<max_len:
            continue
        text = " ".join(str(i) for i in k)
        r = plt.Rectangle((x, y), w, h,
                          facecolor=etichette2color[k[1:3]],
                          edgecolor='gray',
                          label=text)
        ax.add_patch(r)
        if w==0 or h==0:
            continue

    ax.set_xticks([k.x+k.width/2 for k in final_division_named if len(k.code)==1])
    ax.set_xticklabels([k.code[0] for k in final_division_named if len(k.code)==1])
    ax.set_xlabel(levels[0])

    y_ticks = [(k.code[1], k.y+k.height/2) for k in final_division_named if len(k.code)==2]
    y_ticks = [(k, np.mean([i[1] for i in v])) for k, v in toolz.groupby(lambda i:i[0], y_ticks).items()]

    ax.set_yticks([i[1] for i in y_ticks])
    ax.set_yticklabels([i[0] for i in y_ticks])
    ax.set_ylabel(levels[1])

    ax_clone = ax.twiny()
    ax_clone.grid(False)
    z_ticks = [((k.code[0], k.code[2]), k.x+k.width/2) for k in final_division_named if len(k.code)==3]
    z_ticks = [(k, np.mean([i[1] for i in v])) for k, v in toolz.groupby(lambda i: i[0], z_ticks).items()]
    ax_clone.set_xticks([i[1] for i in z_ticks])
    ax_clone.set_xticklabels([i[0][1] for i in z_ticks], rotation=90)
    ax_clone.set_xlabel(levels[2])

    handles = {n:plt.Rectangle((0,0),1,1, color=c) for n, c in etichette2color.items()}
    obj, clr = zip(*sorted(handles.items()))

    ax.legend(clr,
              [" - ".join(map(str, i)) for i in obj],
              bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# %%

# %%
