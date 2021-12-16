#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import shit
from search import *
from notebook import psource, heatmap, gaussian_kernel, show_map, final_path_colors, display_visual, plot_NQueens

# Needed to hide warnings in the matplotlib sections
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import lines

from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display
import time


# In[6]:


# Example 2.1 Graph Search

EX_1_tree = UndirectedGraph(dict(
    A = dict(B=1, C=2 , D=4),
    B = dict(E=2),
    D = dict(F=1),

))

EX_1_tree.locations = dict(
    A = (35, 40), 
    B = (15, 20), C = (35, 20), D = (55, 20),
    E = (15, 10), F = (55, 10)
)

# node colors, node positions and node label positions
node_colors = {node: 'white' for node in EX_1_tree.locations.keys()}
node_positions = EX_1_tree.locations
node_label_pos = { k:[v[0]-3,v[1]]  for k,v in EX_1_tree.locations.items() }
edge_weights = {(k, k2) : v2 for k, v in EX_1_tree.graph_dict.items() for k2, v2 in v.items()}

EX_1_tree_data = {  'graph_dict' : EX_1_tree.graph_dict,
                      'node_colors': node_colors,
                      'node_positions': node_positions,
                      'node_label_positions': node_label_pos,
                      'edge_weights': edge_weights
                     }


# In[7]:


# Solve using breadth first graph search
all_node_colors = []
BFS_problem = GraphProblem('A', 'F', EX_1_tree)

a, b, c = breadth_first_graph_search(BFS_problem)
display_steps(EX_1_tree_data, user_input=False, 
              algorithm=breadth_first_graph_search, 
              problem=BFS_problem)


# In[ ]:




