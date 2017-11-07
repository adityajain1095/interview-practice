'''

graphs


spanning tree --> it is a subgraph that contains all the graph's
vertices in a single tree and enough edges to connect each vertex
without cycles

spanning forest --> is a subgraph that consists of a spanning
tree in each connected component of a graph. it doesn't have 
a cycle.

to be a TREE:

	G has V-1 edges, no cycles
	G have V-1 edges and is connected
	G is connected but removing an edge disconnects
	tree
	G is acyclic, but adding an edge creates a cycle
	only one path to go from one vertex to another

	graph --> directed and undirected graphs


HOW TO REPRESENT GRAPHS:

edge List(or Set)
	list of tuple of edges

Adjacency Lists(or Sets)
	list where each index is the vertex and the bucket
	value is the bucket of connected vertices

Adjacency Matrices

2d array, top row and first column is the vertices
similar to edit distance 2d array, except inside it 
is T, F if there is a connection

BIG-O PERFORMANCE

	EDGE SET 
	vertices              O(1), return set of vertices
	edges                 O(1), return set of edges
	incidentEdges(v)      O(E), go through set of edges and check if v in it
	areAdjacent(v1, v2)   O(1), check if tuple is in set
	insertVertex(v)       O(1), add to set of vertices
	insertEdge(v1,v2) 	  O(1), add to set of edges
	removeVertex(v) 	  O(E), remove from set of vertices, then check in edges and remove if they have v
	removeEdge(v1,v2)	  O(1), remove form set of edges

Adjacency Lists
	vertices              O(1), return set of vertices
	edges                 O(E), go through each vertex, and create edges
	incidentEdges(v)      O(1), look up set at key v
	areAdjacent(v1, v2)   O(1), check if v1 in v2.value
	insertVertex(v)       O(1), add to set of vertices
	insertEdge(v1,v2) 	  O(1), add v2 into set of v1. vice versa
	removeVertex(v) 	  O(V), check each key in set, and if v in value, then remove
	removeEdge(v1,v2)	  O(1), remove form set of edges

Adjacency Matrix
	vertices              O(1), return set of vertices
	edges                 O(V^2), go through 2d list of vertices/edges
	incidentEdges(v)      O(V), look at 2d list column 
	areAdjacent(v1, v2)   O(1), look up 2d[v1,v2]
	insertVertex(v)       O(V)*, add to vertex matrix, update, amoritized if you create big enough matrix
	insertEdge(v1,v2) 	  O(1), update 2d[v1,v2] vice versa
	removeVertex(v) 	  O(V), then remove edges in that column
	removeEdge(v1,v2)	  O(1
	), remove 2d[v1,v2] vice versa

'''

from queue import Queue

def graphBFT(graph, start, end):
	visited = {}
	prev = {}
	for v in graph.vertices:
		visited[v] = False
		prev[v] = None
	q = Queue()
	visited[start] = True
	q.put(start)
	while not q.empty():
		vertex = q.get()
		visited[vertex] = True
		if vertex == end:
			break
		for neighbor in vertex.edges():
			if not visited[neighbor]:
				prev[neighbor] = vertex
				q.put(neighbor)
	vertex = end
	path = []
	while prev[vertex] != None:
		path.append(vertex)
		vertex = prev[vertex]
	return path[::-1]















