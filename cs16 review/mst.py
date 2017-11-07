'''

minimum spanning trees
prim-jarnik's algorithm and kruskal algorithm

a mst is a spanning tree of a weighted gprah with minimum total edge weight

runtime of prim's O((e+V)logV)
'''
from queue import PriorityQueue

def prim(g):

	mst = []
	pq = PriorityQueue()

	for all v in g.vertices():
		v.cost = float('inf')
		v.prev = None
		pq.put((v.cost, v))
	start = pq.get()[1]
	start.cost = 0
	pq.put((start.cost, start))

	while not pq.empty():
		node = pq.get()
		if node.prev:
			mst.append((node, node.prev))
		for neighbor in node.neighbors():
			if neighbor.cost > g.weight(neighbor, node) + node.cost:
				neighbor.cost = g.weight(neighbor, node) + node.cost
				neighbor.prev = node
				pq.update(neighbor, neighbor.cost)

	return mst
