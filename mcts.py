import math
import random as rd
import collections
from unicodedata import name
import numpy as np
import os, shutil
import os.path as osp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from LLMQueryEnv import LLMQueryEnv

# Exploration constant
c_PUCT = 1.38
# Dirichlet noise alpha parameter.
D_NOISE_ALPHA = 0.03
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly
TEMP_THRESHOLD = 5
CHILD_TYPE = 'robust'


class DummyNode:
    """
    Special node that is used as the node above the initial root node to
    prevent having to deal with special cases when traversing the tree.
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)
        self.child_M = collections.defaultdict(float)

    def revert_visits(self, up_to=None): pass

    def backup_value(self, predValue,actualValue,up_to=None): pass


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    """

    def __init__(self,state,n_actions, TreeEnv, action=None, parent=None,childType='robust'):
        """
        :param state: State that the node should hold.
        :param n_actions: Number of actions that can be performed in each
        state. Equal to the number of outgoing edges of the node.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param action: Index of the action that led from the parent node to
        this node.
        :param parent: Parent node.
        """
        self.TreeEnv = TreeEnv
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth+1
        self.parent = parent
        self.action = action
        self.state = state
        self.n_actions = n_actions
        self.is_expanded = False
        self.childType = childType

        self.n_vlosses = 0  # Number of virtual losses on this node
        self.child_N = np.zeros([n_actions], dtype=np.float32)
        self.child_W = np.zeros([n_actions], dtype=np.float32)
        #self.child_W = np.ones([n_actions], dtype=np.float32)*(-1.0)
        self.child_M = np.zeros([n_actions], dtype=np.float32)
        #self.child_M = np.ones([n_actions], dtype=np.float32)*(-1.0)
        # Save copy of original prior before it gets mutated by dirichlet noise
        self.original_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_visited = np.zeros([n_actions], dtype=np.int32)
        self.child_ids = np.zeros([n_actions], dtype=np.int32)
        self.children = {}

    @property
    def N(self):
        """
        Returns the current visit count of the node.
        """
        return self.parent.child_N[self.action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        """
        Returns the current total value of the node.
        """
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value
        
    @property
    def M(self):
        """
        Returns the current total value of the node.
        """
        return self.parent.child_M[self.action]

    @M.setter
    def M(self, value):
        self.parent.child_M[self.action] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)
    
    @property
    def averagedMonteCarlo(self):
        """
        Returns the averaged montecarlo value of the node.
        """
        return self.M / (1 + self.N)
    
    @property
    def child_averagedMonteCarlo(self):
        return self.child_M / (1 + self.child_N)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        #print("Child N = ", len(self.child_N))
        #print("Self N = ", len(self.child_prior))
        return (c_PUCT * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))
        #Child_prior - purely LLM predictions
        #averge

    @property
    def child_action_score(self):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        #return self.child_Q + self.child_U
        return self.child_averagedMonteCarlo + self.child_U
    
    
    def select_leaf(self):
        """
        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        child_N and child_W values but not as a dedicated node in the parent's
        children-mapping). Nodes are selected according to child_action_score.
        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.
        :return: Expanded leaf MCTSNode.
        """
        current = self
        while True:
            print("Leaf selection - depth: ", current.depth)
            #current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                break
            print("Leaf selection - action scores: ", current.child_action_score, " taking action: ", np.argmax(current.child_action_score))
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def select_leaf_during_evaluation(self,childType='robust'):
        current = self
        while True:
            if not current.is_expanded:
                break
            if self.childType == 'max':
                best_move = np.argmax(current.child_averagedMonteCarlo)
            else:
                best_move = np.argmax(current.child_N)
            current = current.maybe_add_child(best_move)
        return current


    def maybe_add_child(self, action):
        """
        Adds a child node for the given action if it does not yet exists, and
        returns it.
        :param action: Action to take in current state which leads to desired
        child node.
        :return: Child MCTSNode.
        """
        if action not in self.children:
            # Obtain state following given action.
            print("Adding child.")
            new_state = self.TreeEnv.next_state(self.state,self.child_ids[action])
            #print("new state: ", new_state)
            self.children[action] = MCTSNode(new_state,self.n_actions,
                                             self.TreeEnv,
                                             action=action, parent=self,childType=self.childType)
            self.child_visited[action] = 1
        return self.children[action]
        
    def incorporate_estimates(self,action_probs,predValue,actualValue,up_to):
        """
        Call if the node has just been expanded via `select_leaf` to
        incorporate the prior action probabilities and state value estimated
        by the neural network.
        :param action_probs: Action probabilities for the current node's state
        predicted by the neural network.
        :param value: Value of the current node's state predicted by the neural
        network.
        :param up_to: The node to propagate until.
        """
        # A done node (i.e. episode end) should not go through this code path.
        # Rather it should directly call `backup_value` on the final node.
        # TODO: Add assert here
        # Another thread already expanded this node in the meantime.
        # Ignore wasted computation but correct visit counts.
        self.is_expanded = True
        self.original_prior = self.child_prior = action_probs
        # This is a deviation from the paper that led to better results in
        # practice (following the MiniGo implementation).
        self.child_W = np.zeros([self.n_actions], dtype=np.float32)
        self.child_M = np.zeros([self.n_actions], dtype=np.float32)
        self.backup_value(predValue,actualValue,up_to=up_to)

    def backup_value(self, predValue,actualValue, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.W += predValue
        self.M += actualValue
        self.N += 1
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(predValue,actualValue,up_to)

    def is_done(self):
        return self.TreeEnv.is_done_state(self.state,self.depth)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
        #dirch = np.random.dirichlet([D_NOISE_ALPHA] * self.child_prior)
        #self.child_prior = self.child_prior * 0.75 + dirch * 0.25

        #print("Shape ", self.parent, len(self.child_prior), " ", len(dirch))
        self.child_prior = self.child_prior * 0.85 + dirch * 0.15

    def visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.
        :param squash: If True, exponentiate the probabilities by a temperature
        slightly large than 1 to encourage diversity in early steps.
        :return: Numpy array of shape (n_actions).
        """
        if self.childType == 'max':
            #probs = self.child_Q # MCTS max child
            probs = self.child_averagedMonteCarlo # MCTS max child
        else:
            probs = self.child_N # MCTS robust child
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)
    
    def print_bfs_tree(self, level=0):
        self.printNodeInfo(level)
        listOfNodesToPrint = []
        for _, child in sorted(self.children.items()):
            listOfNodesToPrint.append((child,level+1))
        while len(listOfNodesToPrint)>0:
            childNode,depth = listOfNodesToPrint.pop(0)
            childNode.printNodeInfo(depth)
            for _, child in sorted(childNode.children.items()):
                listOfNodesToPrint.append((child,depth+1))
            
    def printNodeInfo(self,level):
        node_string = "----"
        node_string += "\n Tree depth: {}".format(level)
        node_string += "\n Node: action={}".format(self.action)
        node_string += "\n• state:{}".format(self.state)
        node_string += "\n• Child Action scores:{}".format(self.child_action_score)
        node_string += "\n• Child averaged monte carlo:{}".format(self.averagedMonteCarlo)
        node_string += "\n• Child probablities:{}".format(self.child_prior)
        node_string += "\n• Child visitation:{}".format(self.child_visited)
        node_string += "\n• N={},Q={},M={}".format(self.N,self.Q,self.averagedMonteCarlo)
        print(node_string)

    def print_tree(self, level=0):
        node_string = "\033[94m|" + "----"*level
        node_string += "\n Node: action={}\033[0m".format(self.action)
        node_string += "\n• state:\n{}".format(self.state)
        node_string += "\n• Child Action scores:\n{}".format(self.child_action_score)
        node_string += "\n• Child visitation:\n{}".format(self.child_visited)
        node_string += "\n• N={},Q={},M={}".format(self.N,self.Q,self.averagedMonteCarlo)
        print(node_string)
        for _, child in sorted(self.children.items()):
            child.print_tree(level+1)


class MCTS:
    """
    Represents a Monte-Carlo search tree and provides methods for performing
    the tree search.
    """

    def __init__(self, TreeEnv, childType='robust',
                 simulations_per_move=800, num_parallel=4):
        """
        :param agent_netw: Network for predicting action probabilities and
        state value estimate.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param seconds_per_move: Currently unused.
        :param simulations_per_move: Number of traversals through the tree
        before performing a step.
        :param num_parallel: Number of leaf nodes to collect before evaluating
        them in conjunction.
        """
        self.TreeEnv = TreeEnv
        self.childType = childType
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None        # Overwritten in initialize_search
        self.isFirstExploration = True
        self.root = None
        self.num_simulations = 0

    def initialize_search(self, state=None):
        init_state = self.TreeEnv.get_initial_state()
        n_actions = self.TreeEnv.n_actions
        print("Initialize search (creating root node)", end='\n\n')
        self.root = MCTSNode(init_state,n_actions, self.TreeEnv,childType=self.childType)
        # Number of steps into the episode after which we always select the
        # action with highest action probability rather than selecting randomly
        self.temp_threshold = TEMP_THRESHOLD

    def tree_search(self):    
        print("MCTS Stage 1 - Selection: finding leaf node.", end='\n\n')
        leaf = self.root.select_leaf()
        if leaf.is_done():
            print("Leaf is terminal - getting return value.")
            value = self.TreeEnv.get_return(leaf.state,leaf.depth)
            print("MCTS Stage 4 - Backpropogation: incorporating estimates.", end='\n\n')
            leaf.backup_value(value,value,up_to=self.root)
        else:
            print("Getting LLM token estimates (probs/ids).")
            probs, ids = self.TreeEnv.getLLMestimates(leaf.state)
            leaf.child_ids = ids
            startingAction = leaf.child_ids[np.argmax(probs)]
            print("MCTS Stage 2 - Expansion: next action: ", np.argmax(probs), " corresponding to state: ", startingAction, end='\n\n')
            next_state = self.TreeEnv.next_state(leaf.state,startingAction)
            print("MCTS Stage 3 - Rollout: Getting rollout return of leaf.", end='\n\n')
            rolloutReturn = self.TreeEnv.get_montecarlo_return(next_state,leaf.depth+1)
            print("MCTS Stage 4 - Backpropogation: incorporating estimates.", end='\n\n')
            leaf.incorporate_estimates(probs,rolloutReturn,rolloutReturn,up_to=self.root)
            
    def test_tree_search(self,cType):
        ## This should not backup value since we are only traversing to the leaf node based
        ## on robust-child or max-child and return the value
        
        #print("Post-MCTS ideal tree search: ")
        leaf = self.root.select_leaf_during_evaluation(cType)
        if leaf.is_done():
            #print("MCTS tree nodes reached end of module.")
            value = self.TreeEnv.get_return(leaf.state,leaf.depth)
        else:
            #print("MCTS tree didnt reach end nodes - geting MC return for rest of prediction:")
            
            probs, ids = self.TreeEnv.getLLMestimates(leaf.state)
            leaf.child_ids = ids
            startingAction = leaf.child_ids[np.argmax(probs)]
            next_state = self.TreeEnv.next_state(leaf.state,startingAction)
            value = self.TreeEnv.get_montecarlo_return(next_state,leaf.depth+1)
            
        return value


def initialize_MCTS_tree(TreeEnv):
    print("Initializing MCTS tree.")
    mcts = MCTS(TreeEnv,childType=CHILD_TYPE)
    mcts.initialize_search()
    print("MCTS Stage 1 - Selection: finding leaf node.", end='\n\n')
    first_node = mcts.root.select_leaf()
    print("Getting LLM token estimates (probs/ids).")
    probs, ids = TreeEnv.getLLMestimates(first_node.state)

    ## Compute montecarlo return using the policy's first best move followed by random rather than entirely random policy
    first_node.child_ids = ids
    startingAction = first_node.child_ids[np.argmax(probs)]
    print("MCTS Stage 2 - Expansion: next action: ", np.argmax(probs), " corresponding to state: ", startingAction, end='\n\n')
    next_state = TreeEnv.next_state(first_node.state,startingAction)
    print("MCTS Stage 3 - Rollout: Getting rollout return of leaf.", end='\n\n')
    first_node_rolloutReturn = TreeEnv.get_montecarlo_return(next_state,first_node.depth+1) #resyn2 output will lead to DRAW or 0.
    print("MCTS Stage 4 - Backpropogation: incorporating estimates.", end='\n\n')
    first_node.incorporate_estimates(probs, first_node_rolloutReturn, first_node_rolloutReturn,first_node)
    mcts.root.inject_noise()
    return mcts

def initialize_thread_tree(index, prompt_str, problem_name, file_dir, model_name, tokenizer, model):
    # Create a partially applied function with the required parameters
    return initialize_MCTS_tree(
        LLMQueryEnv(orig_prompt=prompt_str, orig_module=problem_name, file_path=file_dir,
                    model_name=model_name, tokenizer=tokenizer, model=model)
    )

def execute_episode(mctsTree,simulation_budget):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mctsTree.num_simulations += 1
    current_runs = mctsTree.root.N
    #print("Current runs: ", current_runs)
    #print("Simulation budget", simulation_budget)
    while mctsTree.root.N < current_runs+simulation_budget:
        if mctsTree.root.N > 0 and mctsTree.root.N % 100 == 0:
            print("ROBUST FINAL VALUE, ITERATION: ", current_runs)
            evalMctsRobustValue, evalMctsMaxValue = test_episode(mctsTree)

        mctsTree.tree_search()
        print("-------------------------------------------------------")
        print("MCTS Iteration: ", mctsTree.root.N)

        mctsTree.TreeEnv.row_data['episode'] = mctsTree.num_simulations
        mctsTree.TreeEnv.row_data['currentRun'] = mctsTree.root.N
        
        mctsTree.TreeEnv.csv_logger.log(mctsTree.TreeEnv.row_data)

    mctsTree.root.print_bfs_tree()
    #print("execute episode finished")
    return mctsTree

def test_episode(mctsTree):
    mctsEvalMaxValue = mctsTree.test_tree_search(cType='max')
    mctsEvalRobustValue = mctsTree.test_tree_search(cType='robust')
    return mctsEvalRobustValue, mctsEvalMaxValue

def merge_nodes(merged_tree_node, thread_tree_node):
    #for actions in current merged tree node
    for action in merged_tree_node.children:
        #if action (child) also exists in the thread tree
        if action in thread_tree_node.children:
            merge_nodes(merged_tree_node.children[action], thread_tree_node.children[action])
        else:
            #If the child does not exist
            merged_tree_node.children[action] = thread_tree_node.children[action]

    for action in thread_tree_node.children:
        if action not in merged_tree_node.children:
            merged_tree_node.children[action] = thread_tree_node.children[action]


def update_root_node(merged_root, thread_root):
    merged_root.parent.child_N += thread_root.parent.child_N
    merged_root.parent.child_W += thread_root.parent.child_W
    merged_root.parent.child_M += thread_root.parent.child_M

def update_node_children(merged_node, thread_node):
    #add child_n, child_m, child_w, child_visited
    #If child dictionary doesnt contain node, add it here.
    merged_node.child_N += thread_node.child_N
    merged_node.child_W += thread_node.child_W
    merged_node.child_M += thread_node.child_M
    for action in thread_node.children:
        if action not in merged_node.children:
            merged_node.children[action] = thread_node.children[action]

def merge_nodes(merged_tree_node, thread_tree_node):
    if (merged_tree_node.parent == DummyNode() and thread_tree_node.parent == DummyNode()):
        #If root node
        update_root_node(merged_tree_node, thread_tree_node)

    if thread_tree_node.children:
        update_node_children(merged_tree_node, thread_tree_node)

        for action in thread_tree_node.children:
            thread_node_child = thread_tree_node.children[action]
            merged_node_child = merged_tree_node.children[action]            
            merge_nodes(merged_node_child, thread_node_child)
    
def merge_trees(merge_tree, thread_trees):
    for thread_tree in thread_trees:
        merge_nodes(merge_tree.root, thread_tree.root)

        