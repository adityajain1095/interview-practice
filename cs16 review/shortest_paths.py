'''

shortest paths in a graph

two main algos for finding shortest paths:
breadth first search --> works when all unit edges
	
djikstra's for non negative paths
bellman-ford for graphs with negative

use priority queue in shortest path instead of queue like breadth first search
priority queue for djikstra should be binary heap so runtime of removing is log(V)
and not v

with heap:
	runtime is o(V)log(v)+ o(e)log(v)
	without heap is o(v)(v+e) --> o(v^2)

#bellman ford algorithm, repeat v-1 times, go through each edge in the loop
and update accordingly


'''

class T(object):

	def __init__(self):
		self.v = 4

from queue import PriorityQueue
import random
i = 0
a = T()
b = T()
c = T()
d = T()
graph = [a,b,c,d]

def djikstra(graph, start, end):
	pq = PriorityQueue()
	for vertex in graph.vertices():   #O(v)
		vertex.visited = False
		vertex.prev = None
		vertex.cost = float('inf')
		pq.put(vertex, vertex.cost)
	start.visited = True
	start.cost = 0
	pq.replace(start, start.cost)
	while not pq.empty():         #O(v)
		node = pq.get()				#O(logv)
		if node == end:
			break
		for neighbor in node.neighbors():     #O(e)
			if edge(neighbor, node) + node.cost < neighbor.cost
				neighbor.cost = edge(neighbor, node) + node.cost
				neighbor.prev = node
				pq.replace(neighbor, neighbor.cost)    #o(logv)
	node = end
	path = []
	while node != None
		results.append(node)
		node = node.prev
	return path[::-1]

	def BellmanFord(graph, start, end):
		for vertex in graph.vertices():
			vertex.dist = float('inf')
			vertex.prev = None
		start.dist = 0
		for i in range(len(graph.vertices()) - 1):
			for edge in graph.edges():
				if edge[0].dist + cost(edge[0],edge[1]) > edge[1].dist:
					edge[1].dist = edge[0].dist + cost(edge[0],edge[1])
					edge[1].prev = edge[0]
		node = end
		path = []
		while node != None
			results.append(node)
			node = node.prev
		return path[::-1]