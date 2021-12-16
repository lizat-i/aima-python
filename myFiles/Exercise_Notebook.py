#!/usr/bin/env python
# coding: utf-8

# In[1]:


from search import *
from ypstruct import struct
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

from search_helpers import show_tree, display_steps


# In[2]:


# Example 2.1 Graph Search

simple_treeEX_1 = UndirectedGraph(dict(
    A = dict(B=1, C=2 , D=4),
    B = dict(E=2),
    D = dict(F=1),

))

simple_treeEX_1.locations = dict(
    A = (35, 40), 
    B = (15, 20), C = (35, 20), D = (55, 20),
    E = (15, 10), F = (55, 10)
)

# node colors, node positions and node label positions
node_colors = {node: 'white' for node in simple_treeEX_1.locations.keys()}
node_positions = simple_treeEX_1.locations
node_label_pos = { k:[v[0]-3,v[1]]  for k,v in simple_treeEX_1.locations.items() }
edge_weights = {(k, k2) : v2 for k, v in simple_treeEX_1.graph_dict.items() for k2, v2 in v.items()}

simple_treeEX_1_data = {  'graph_dict' : simple_treeEX_1.graph_dict,
                      'node_colors': node_colors,
                      'node_positions': node_positions,
                      'node_label_positions': node_label_pos,
                      'edge_weights': edge_weights
                     }


# In[3]:


def dataSetCollector(problem,iterations,state,frontier,explored):
    problem.iterations.append(iterations)
    problem.state.append(state)
    problem.frontier.append(frontier)
    problem.explored.append(explored)
    print("isCalled")
    print(str(iterations),str(state),str(frontier),str(explored))
    return problem


# In[4]:


def breadth_first_search_graph_vis_u(problem):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue."""
    
    # we use these two variables at the time of visualisation
    iterations = 0  
    all_node_colors = []
    node_colors = {k : 'white' for k in problem.graph.nodes()}
    node = Node(problem.initial)   
    node_colors[node.state] = "red"
 
    
    iterations += 1
    all_node_colors.append(dict(node_colors))
      
    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        problem.solution.iterations = append
        return(iterations, all_node_colors, node)
    
    frontier = deque([node])   
    # modify the color of frontier nodes to blue
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))
        
    explored = set()

    #Datacollector
    problem = dataSetCollector(problem,iterations,node.state,frontier,explored)
    
    while frontier:
        
        node = frontier.popleft()
        node_colors[node.state] = "red"
        #Datacollector
        problem = dataSetCollector(problem,iterations,node.state,frontier,explored)
        iterations += 1
        all_node_colors.append(dict(node_colors))        
        explored.add(node.state)
        
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    node_colors[child.state] = "green"
                    #Datacollector
                    problem = dataSetCollector(problem,iterations,node.state,frontier,explored)
                    iterations += 1
                    all_node_colors.append(dict(node_colors))
 
                    return(iterations, all_node_colors, child)
                frontier.append(child)
                node_colors[child.state] = "orange"
                #Datacollector
                problem = dataSetCollector(problem,iterations,node.state,frontier,explored)
                iterations += 1
                all_node_colors.append(dict(node_colors))
                
        node_colors[node.state] = "gray"
        #Datacollector
        problem = dataSetCollector(problem,iterations,node.state,frontier,explored)
        iterations += 1 
        all_node_colors.append(dict(node_colors))
    return None

def breadth_first_graph_search_mod(problem):
    "Search the shallowest nodes in the search tree first."
    iterations, all_node_colors, node = breadth_first_search_graph_vis_u(problem)
    return(iterations, all_node_colors, node)


# In[5]:


all_node_colors = []
BFS_problem = GraphProblem('A', 'F', simple_treeEX_1)
a, b, c = breadth_first_graph_search_mod(BFS_problem)
print(*BFS_problem.iterations)
display_steps(simple_treeEX_1_data, user_input=False, 
              algorithm=breadth_first_graph_search_mod, 
              problem=BFS_problem)


# In[ ]:


#Solve using depth first graph search
all_node_colors = []
EX1 = GraphProblem('A', 'F', simple_treeEX_1)
a, b, c = breadth_first_graph_search_mod(EX1)
display_steps(simple_treeEX_1_data, user_input=False, 
               algorithm=depth_first_graph_search, 
              problem=EX1)


# In[ ]:


'''
def breadth_first_search_graph_vis(problem):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue."""
    
    # we use these two variables at the time of visualisation
    iterations = 0
    all_node_colors = []
    node_colors = {k : 'white' for k in problem.graph.nodes()}
    
    node = Node(problem.initial)
    
    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))
      
    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return(iterations, all_node_colors, node)
    
    frontier = deque([node])
    
    # modify the color of frontier nodes to blue
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))
        
    explored = set()
    while frontier:
        node = frontier.popleft()
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        
        explored.add(node.state) 
        
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    node_colors[child.state] = "green"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))
                    return(iterations, all_node_colors, child)
                frontier.append(child)

                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
                
        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None
 '''

