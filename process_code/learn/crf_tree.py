'''
Created on Mar 6, 2017

@author: Tuan
'''
from collections import deque
import copy

import numpy as np
import tensorflow as tf
from crf_utils import gather_2d, gather_2d_to_shape, expand, expand_first

class CRFTree(object):
    '''
    '''
    
    def __init__(self, node_types, dictionaries, edges):
        '''
        Parameters:
        ----------
        node_types:    list of String
        dictionaries:  dictionary of dictionaries, for each node_type there is a dictioanry from node value to id
        edges:         dictionary of list representation of edges
        '''
        self.node_types = node_types
        self.node_type_indices = dict((node_type, i) for i, node_type in enumerate(self.node_types))
        self.dictionaries = dictionaries
        self.edges = edges

        for node_type in node_types:
            if node_type not in dictionaries or len(dictionaries[node_type] ) == 0:
                raise Exception('There should be at least one label for each node_type')

    def initiate_crf(self):
        self.crf = {}

        with tf.variable_scope("crf"):
            for node_1 in self.edges:
                for node_2 in self.edges[node_1]:
                    edge = (node_1, node_2)
                    sorted_edge = tuple(sorted(edge) )
                    if not sorted_edge in self.crf:
                        source, target = sorted_edge
                        if source in self.dictionaries and target in self.dictionaries:
                            self.crf[sorted_edge] = tf.get_variable("A_" + source + '_' + target, 
                                                        [len(self.dictionaries[source]), len(self.dictionaries[target])])
                            print sorted_edge
                            print self.crf[sorted_edge].get_shape()

    def is_tree(self):
        '''
        Check to see if the input graph is actually a tree
        
        Return:
        ------
        True If is a tree graph
        '''
        """
        Just start from any node, BFS through the tree, if it visited all nodes, and doesn't come back to any node
        """
        visited = dict([ (node, False) for node in self.node_types])
        print 'visited' , visited
        if len(self.node_types) == 0:
            return False
        
        start = self.node_types[0]
        
        q = deque([(None, start)])
        
        while len(q) != 0:
            parent, visit = q.popleft()

            visited[visit] = True
            for t in self.edges[visit]:
                if t != parent:
                    if visited[t]:
                        # Detect a circular
                        return False

                    q.append( (visit, t) )
        
        return all(visited.values())
    
    @staticmethod
    def look_for_collapsing_node(edges):
        '''
        Find a node that has the most edges connected to leaf
        '''
        leaves = set([node for node in edges if len(edges[node]) == 1])
        
        count_leaf = sorted( [ (node, set(edges[node]) & leaves)  for node in edges], key = lambda x: len(x[1]) )
        
        selected_node, collapsed_nodes = count_leaf[-1]
        
        return (selected_node, collapsed_nodes)
        
        
    @staticmethod
    def empty(edges):
        for source in edges:
            if edges[source] != None and len(edges[source]) != 0:
                return False
            
        return True
        
    def sum_over(self, crf_weight, logits):
        '''
        Sum over all exponential term for all combination of values 
        
        Parameters:
        -----------
        crf_weight:     Weight for CRF 
        logits:         for each node_type, an np array of ( batch_size, #node_type_targets )
        
        
        Return:
        -------
        log_sum =  numpy array of size = batch_size
        '''
        
        # Remove edges on the cloned edges, not from self.edges
        # We will remove until there are no edges left
        cloned_edges = copy.deepcopy( self.edges )
        
        
        cloned_logits = {}
        
        for node_type in logits:
            cloned_logits[node_type] = logits[node_type]
        
        def recursive_sum_over( edges, logits ):
            '''
            Recursively sum over the exponential components, given the current state of edges and logits
            
            Parameters:
            -----------
            edges:          Current state of edges (some nodes and edges might have been collapsed) 
            logits:         logit for each slot
            
            
            Return:
            -------
            log_sum =  numpy array of size = batch_size
            '''
            if not CRFTree.empty(edges):
                # All nodes in collapsed_nodes will be collapsed into selected_node
                selected_node, collapsed_nodes = CRFTree.look_for_collapsing_node(edges)
                
                size_source = len(self.dictionaries[selected_node])
                
                for collapsed_node in collapsed_nodes:
                    sorted_edge = tuple(sorted((selected_node, collapsed_node)))
                    A = self.crf[sorted_edge]
                    logit = logits[collapsed_node]
                    
                    size_target = len(self.dictionaries[collapsed_node])
                    
                    print 'selected_node', selected_node
                    print sorted_edge
                    if selected_node == sorted_edge[0]:
                        # Same order
                        # A will have size (size_source, size_target)
                        log_edge = tf.reduce_min(crf_weight * tf.transpose(A) + expand(logit, size_source, axis = 2), 1)
                        
                        log_edge += tf.log(tf.reduce_sum(tf.exp(crf_weight * tf.transpose(A) +\
                                                                expand(logit, size_source, axis = 2) -\
                                                                expand(log_edge, size_target, axis = 1) ), 1))
                    else:
                        # Reverse order
                        # A will have size (size_target, size_source)
                        log_edge = tf.reduce_min(crf_weight * A + expand(logit, size_source, axis = 2), 1)
                        
                        log_edge += tf.log(tf.reduce_sum(tf.exp(crf_weight * A +\
                                                                expand(logit, size_source, axis = 2) -\
                                                                expand(log_edge, size_target, axis = 1) ), 1))
                        
                    logits[selected_node] += log_edge
            
                for collapsed_node in collapsed_nodes:
                    del edges[collapsed_node]
                    del logits[collapsed_node]
                
                # Remaining nodes that connected to collapsed_selected_node
                remaining_nodes = list(set(edges[selected_node]) - collapsed_nodes)
                
                edges[selected_node] = remaining_nodes
                
                return recursive_sum_over ( edges, logits )
            else:
                #There should be only one key in logits, otherwise throw an Error
                if len(logits) == 1:
                    remaining_node = logits.keys()[0]
                    
                    # ( #remaining_node, batch_size)
                    logit = tf.transpose(logits[remaining_node])
                    
                    # ( batch_size )
                    log_sum = tf.reduce_min(logit, 0)
                    log_sum += tf.log(tf.reduce_sum(tf.exp(logit - log_sum), 0))
                    
                    return log_sum
                else:
                    raise Exception("At this state, there should be only one logit")
        
        return recursive_sum_over (cloned_edges, cloned_logits)
    
    def calculate_logit_correct(self, crf_weight, batch_size, logits, targets):
        '''
        Sum over all exponential term for all combination of values 
        
        Parameters:
        -----------
        crf_weight:     Weight for CRF 
        batch_size:     add a batch_size so that we don't have to recalculate
        logits:         for each node_type, an np array of ( batch_size, #node_type_targets )
        targets:        Correct labels for training, of size np.array ( batch_size, #self.node_types)
        
        Return:
        -------
        logit_correct =  numpy array of size = batch_size
        '''
        
        logit_correct = tf.zeros(batch_size)

        
        for source in self.edges:
            source_id = self.node_type_indices[source]
            target_src = targets[:, source_id]

            for target in self.edges[source]:
                sorted_edge = tuple(sorted((source, target)))
                # Only count 1 for each edge
                if (source, target) == sorted_edge:
                    target_id = self.node_type_indices[target]
                    target_target = targets[:, target_id]
                    logit_correct += crf_weight * gather_2d ( self.crf[sorted_edge], tf.transpose(tf.stack([target_src, target_target])))
            
            logit_correct += gather_2d(logits[source], tf.transpose(tf.stack([tf.range(batch_size), target_src])))

        return logit_correct
    
    def predict(self, crf_weight, batch_size, logits ):
        '''
        Argmax over all exponential combinations of values.
        This is analogous to the process in sum_over
        
        Parameters:
        -----------
        crf_weight:     Weight for CRF 
        batch_size:     add a batch_size so that we don't have to recalculate
        logits:         for each node_type, an np array of ( batch_size, #node_type_targets )
        
        
        Return:
        -------
        out:            numpy array of size = (batch_size, len(self.node_types) )
        
        Here I use a kind of collapsing algorithm, each time, looking for a node to collapse on. All leaf nodes connected to this node 
        are collapsed into this center node.
        '''
        
        
        def recursive_predict(edges, logits, best_combinations = {}, collapse_list = deque() ):
            '''
            Parameters:
            -----------
            edges:          Current state of edges (some nodes and edges might have been collapsed) 
            logits:         add a batch_size so that we don't have to recalculate
            best_combinations: Current best combination states, map from each node of collapsed_nodes to a tensorflow of size (batch_size, # of values of selected_node)
            collapse_list:  Dict from selected_node to collapsed_nodes that has been collected
            
            Return:
            -------
            out:            numpy array of size = (batch_size, len(self.node_types) )
            '''
            if not CRFTree.empty(edges):
                # All nodes in collapsed_nodes will be collapsed into selected_node
                selected_node, collapsed_nodes = CRFTree.look_for_collapsing_node(edges)
                
                collapse_list.append((selected_node, collapsed_nodes))
                
                size_source = len(self.dictionaries[selected_node])
                
                for collapsed_node in collapsed_nodes:
                    sorted_edge = tuple(sorted((selected_node, collapsed_node)))
                    A = self.crf[sorted_edge]
                    logit = logits[collapsed_node]
                    
                    if selected_node == sorted_edge[0]:
                        # Same order
                        # (batch_size, size_target, size_source)
                        log_edge = crf_weight * tf.transpose(A) + expand(logit, size_source, axis = 2)
                    else:
                        # Reverse order
                        # (batch_size, size_target, size_source)
                        log_edge = crf_weight * A + expand(logit, size_source, axis = 2)
                    
                    # (batch_size, size_source)
                    best_combinations[collapsed_node] = tf.cast(tf.argmax(log_edge, axis = 1), np.int32)
                    
                    logits[selected_node] += tf.reduce_max(log_edge, 1)
            
                for collapsed_node in collapsed_nodes:
                    del edges[collapsed_node]
                    del logits[collapsed_node]
                
                # Remaining nodes that connected to collapsed_selected_node
                remaining_nodes = list(set(edges[selected_node]) - collapsed_nodes)
                
                edges[selected_node] = remaining_nodes
                
                return recursive_predict(edges, logits)
            else:
                if len(logits) == 1:
                    remaining_node = logits.keys()[0]
                    
                    size_remaining = len(self.dictionaries[remaining_node])
                    
                    """
                    Recalculate best_combinations according to the last node
                    """
                    recalculated_best_combinations = [tf.zeros((batch_size, size_remaining), dtype=np.int32) for _ in xrange(len(self.node_types))]
                    recalculated_best_combinations[self.node_type_indices[remaining_node]] = expand_first(range(size_remaining), batch_size)
                    
                    while len(collapse_list) > 0:
                        selected_node, collapsed_nodes = collapse_list.pop()
                        selected_node_index = self.node_type_indices[selected_node]
                        
                        """
                        Propagate from selected_node to collapsed_nodes
                        """
                        # (batch_size, #Subject)
                        q = np.array([[i for _ in xrange(size_remaining)] for i in xrange(batch_size)])
                        
                        # (batch_size x #Subject, 2)
                        indices = tf.reshape( tf.transpose( tf.stack ( [q, recalculated_best_combinations[selected_node_index] ]), [1, 2, 0] ), [-1, 2]) 
                        
                        for collapsed_node in collapsed_nodes:
                            collapsed_index = self.node_type_indices[collapsed_node]
                            recalculated_best_combinations[collapsed_index] = gather_2d_to_shape(best_combinations[collapsed_node], 
                                                                 indices, (batch_size, size_remaining))
            
                    # batch_size
                    best_best_values = tf.argmax(logits[remaining_node], axis = 1)
                    
                    indices = tf.transpose( tf.stack([range(batch_size), best_best_values]))
                    
                    out = tf.transpose(tf.stack([gather_2d( recalculated_best_combinations[t], indices ) for t in xrange(len(self.node_types))]))
                    
                    return out
                else:
                    raise Exception("At this state, there should be only one logit")
        
        # Remove edges on the cloned edges, not from self.edges
        # We will remove until there are no edges left
        cloned_edges = copy.deepcopy( self.edges )
        cloned_logits = {}
        
        for node_type in logits:
            cloned_logits[node_type] = logits[node_type]
            
        return recursive_predict (cloned_edges , cloned_logits )