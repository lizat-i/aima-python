#!/usr/bin/env python
# coding: utf-8

# # Informed Search
# 
# This notebook serves as supporting material for topics covered in **Lecture 4 - Informed Search** from the lecture *Grundlagen der Künstlichen Intelligenz(IN2062)* at the Technical University of Munich. This notebook uses implementations from [search.py](https://github.com/aimacode/aima-python/blob/master/search.py) module. Let's start by importing everything from the search module.
# 
# <div class="alert alert-info">
#     <h3>Note</h3>
#     <p>In this notebook, we show graph search while the examples in the lecture slides show tree search.</p> 
# </div>

# ## How to use this notebook
# Clone the aima-python repository to your local machine, and add this notebook as well as `search_helpers.py` **directly** to the root directory of aima-python in order to make the following imports work.
# 
# There is also a bigger notebook, *search.ipynb*, in the same root directory, which contains more examples from the book *Artificial Intelligence: A Modern Approach*. Unfortunately, some modules there are not working at the moment. That is why it is important to import the PriorityQueue from our helpers in order to make it work.

# In[1]:


from search import *
from notebook import psource, heatmap, gaussian_kernel, show_map, final_path_colors, display_visual, plot_NQueens

from search_helpers import PriorityQueue

# Needed to hide warnings in the matplotlib sections
import warnings
warnings.filterwarnings("ignore")


# ## CONTENTS
# 
# * Visualization and Problem Setup
# * Search algorithms
#   1. Greedy Best-First Search
#   2. A* Search
# * What's next

# # Visualization and Problem Setup
# For visualisations, we use networkx and matplotlib to show the map in the notebook and we use ipywidgets to interact with the map to see how the searching algorithm works. These are imported as required in `notebook.py`.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import lines

from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display
import time


# # 1. Greedy Best-First Search
# ***Lecture 4, Slide 6-9***
# 
# In the last notebook, we constructed a simple tree. Here, we use the same method to construct a romania_map.

# In[3]:


romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))

romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))


# Specify the start location and destination here:

# In[4]:


romania_problem = GraphProblem('Arad', 'Bucharest', romania_map)


# In[5]:


# node colors, node positions and node label positions
node_colors = {node: 'white' for node in romania_map.locations.keys()}
node_positions = romania_map.locations
node_label_pos = { k:[v[0],v[1]-10]  for k,v in romania_map.locations.items() }
edge_weights = {(k, k2) : v2 for k, v in romania_map.graph_dict.items() for k2, v2 in v.items()}

romania_graph_data = {  'graph_dict' : romania_map.graph_dict,
                        'node_colors': node_colors,
                        'node_positions': node_positions,
                        'node_label_positions': node_label_pos,
                         'edge_weights': edge_weights
                     }


# In[6]:


show_map(romania_graph_data)


# An easy way to model the Greedy Best-First Search algorithm is to construct it on the base of Best-First search.
# 
# A reminder of how we based ***Uniform Cost Search*** on ***Best First Graph Search***:
# ```python
# def uniform_cost_search(problem):
#     return best_first_graph_search_for_vis(problem, lambda node: node.path_cost)
# ```

# In[7]:


def best_first_graph_search_for_vis(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    
    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k : 'white' for k in problem.graph.nodes()}
    
    f = memoize(f, 'f')
    node = Node(problem.initial)
    
    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return(iterations, all_node_colors, node)
    
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    explored = set()
    while frontier:
        node = frontier.pop()
        
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        
        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return(iterations, all_node_colors, node)
        
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None


# In[8]:


def greedy_best_first_search(problem, h=None):
    """Greedy Best-first graph search is an informative searching algorithm with f(n) = h(n).
    You need to specify the h function when you call best_first_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    iterations, all_node_colors, node = best_first_graph_search_for_vis(problem, lambda n: h(n))
    return(iterations, all_node_colors, node)


# As in the notebook *UninformedSearch.ipynb*:
# 
# Execute the following code block, you will see a "visualize" button directly beneath the "iteration" slider if you are running this notebook for the first time. Otherwise you can always find the button on the bottom.
# 
# Click the "visualize" button and let the search algorithm runs through, once it's completed, you can use the slider to see all the iterations step by step.

# In[9]:


all_node_colors = []
romania_problem = GraphProblem('Arad', 'Bucharest', romania_map)
display_visual(romania_graph_data, user_input=False, 
               algorithm=greedy_best_first_search, 
               problem=romania_problem)


# # 2. A* Search
# ***Lecture 4, Slide 17-22***
# 
# Here we use the romania map example again to illustrate A* Search.
# 
# Again, we base this on the  Best-First search by changing the cost evaluation function.

# In[10]:


def astar_search_graph(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    iterations, all_node_colors, node = best_first_graph_search_for_vis(problem, 
                                                                lambda n: n.path_cost + h(n))
    return(iterations, all_node_colors, node)


# In[11]:


all_node_colors = []
romania_problem = GraphProblem('Arad', 'Bucharest', romania_map)
display_visual(romania_graph_data, user_input=False, 
               algorithm=astar_search_graph, 
               problem=romania_problem)


# # What's next
# 
# Congratulation on completing this notebook! Now you have a better understanding of different search algorithms. 
# 
# The next notebook is about ***Lecture 5: Constraint Satisfaction Problems***
# 

# In[ ]:




