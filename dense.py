import numpy as np
import random
import heapq
import time
import sys

# Number of nodes

num_nodes = 5000

# Generate a dense graph adjacency matrix with random edge weights between 0 and 9
dense_graph = np.random.randint(10, size=(num_nodes, num_nodes))

# Ensure the graph is undirected (symmetric adjacency matrix)
dense_graph = np.triu(dense_graph) + np.triu(dense_graph, k=1).T

# Set the diagonal to 0 (no self-loops)
np.fill_diagonal(dense_graph, 0)

# Create both adjacency matrix and adjacency list representations


def adjacency_matrix_and_list(graph_matrix):
    adjacency_matrix = graph_matrix.copy()

    adjacency_list = {}
    num_nodes = len(graph_matrix)
    for node in range(num_nodes):
        neighbors = []
        for neighbor, weight in enumerate(graph_matrix[node]):
            if weight > 0:
                neighbors.append((neighbor, weight))
        adjacency_list[node] = neighbors

    return adjacency_matrix, adjacency_list


# Get both representations of the graph
adj_matrix, adj_list = adjacency_matrix_and_list(dense_graph)


def lazy_prim_matrix(adj_matrix):
    num_nodes = len(adj_matrix)
    mst = []  # Minimum Spanning Tree edges
    visited = [False] * num_nodes

    # Create a list to store edges with their weights
    edges = [(float('inf'), None)] * num_nodes

    # Initialize with an arbitrary node (e.g., 0)
    start_node = 0

    while True:
        visited[start_node] = True

        # Add eligible edges to the priority queue
        for end_node, weight in enumerate(adj_matrix[start_node]):
            if not visited[end_node] and weight < edges[end_node][0]:
                edges[end_node] = (weight, start_node)

        # Find the minimum-weight edge to add to MST
        min_weight, min_edge = float('inf'), None
        for node, (weight, parent) in enumerate(edges):
            if not visited[node] and weight < min_weight:
                min_weight, min_edge = weight, node

        if min_edge is None:
            # No more eligible edges to add
            break

        mst.append((min_edge, edges[min_edge][1]))
        start_node = min_edge

    # Calculate the total weight of MST
    total_weight = sum(edge[0] for edge in mst)

    return mst, total_weight


def lazy_prim_list(adj_list):
    num_nodes = len(adj_list)
    mst = []  # Minimum Spanning Tree edges
    visited = [False] * num_nodes

    # Create a priority queue to store edges with their weights
    edge_heap = []

    # Initialize with an arbitrary node (e.g., 0)
    start_node = 0

    def visit(node):
        visited[node] = True
        for neighbor, weight in adj_list[node]:
            if not visited[neighbor]:
                heapq.heappush(edge_heap, (weight, node, neighbor))

    visit(start_node)

    while edge_heap:
        weight, src, dest = heapq.heappop(edge_heap)
        if visited[dest]:
            continue
        mst.append((src, dest, weight))
        visit(dest)

    # Calculate the total weight of MST
    total_weight = sum(edge[2] for edge in mst)

    return mst, total_weight


def prim_mst_eager_matrix(adj_matrix):
    num_nodes = len(adj_matrix)
    # Initialize data structures to keep track of the MST
    mst = []  # Stores the edges of the MST
    key = [sys.maxsize] * num_nodes  # Key values to track minimum edge weights
    parent = [-1] * num_nodes  # Parent array to construct MST
    in_mst = [False] * num_nodes  # Track nodes in MST
    heap = [(0, 0)]  # Priority queue to store candidate edges (weight, node)

    while heap:
        weight, node = heapq.heappop(heap)

        # If the node is already in the MST, skip it
        if in_mst[node]:
            continue

        in_mst[node] = True
        if parent[node] != -1:
            mst.append((parent[node], node, weight))

        # Update key values and push adjacent nodes to the heap
        for v in range(num_nodes):
            if not in_mst[v] and adj_matrix[node][v] < key[v]:
                key[v] = adj_matrix[node][v]
                parent[v] = node
                heapq.heappush(heap, (key[v], v))

    return mst


def prim_mst_eager_list(adj_list):
    num_nodes = len(adj_list)
    # Initialize data structures to keep track of the MST
    mst = []  # Stores the edges of the MST
    key = [sys.maxsize] * num_nodes  # Key values to track minimum edge weights
    parent = [-1] * num_nodes  # Parent array to construct MST
    in_mst = [False] * num_nodes  # Track nodes in MST
    heap = [(0, 0)]  # Priority queue to store candidate edges (weight, node)

    while heap:
        weight, node = heapq.heappop(heap)

        # If the node is already in the MST, skip it
        if in_mst[node]:
            continue

        in_mst[node] = True
        if parent[node] != -1:
            mst.append((parent[node], node, weight))

        # Update key values and push adjacent nodes to the heap
        for neighbor, edge_weight in adj_list[node]:
            if not in_mst[neighbor] and edge_weight < key[neighbor]:
                key[neighbor] = edge_weight
                parent[neighbor] = node
                heapq.heappush(heap, (key[neighbor], neighbor))

    return mst


def kruskal_matrix(adj_matrix):
    num_nodes = len(adj_matrix)
    edges = []

    # Step 1: Create a list of edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] > 0:
                edges.append((adj_matrix[i][j], i, j))

    # Step 2: Sort edges by weight
    edges.sort()

    # Step 3: Initialize connected components
    components = [i for i in range(num_nodes)]

    # Step 4: Initialize MST
    minimum_spanning_tree = []

    # Step 5: Iterate through sorted edges
    for edge in edges:
        weight, start_node, end_node = edge

        # Check if adding the edge creates a cycle
        if components[start_node] != components[end_node]:
            minimum_spanning_tree.append(edge)

            # Merge the connected components
            old_component, new_component = components[start_node], components[end_node]
            for i in range(num_nodes):
                if components[i] == old_component:
                    components[i] = new_component

            # Check if MST is complete (V-1 edges)
            if len(minimum_spanning_tree) == num_nodes - 1:
                break

    return minimum_spanning_tree


def kruskal_list(adj_list):
    def find_parent(parent, node):
        while parent[node] != -1:
            node = parent[node]
        return node

    def union(parent, x, y):
        x_set = find_parent(parent, x)
        y_set = find_parent(parent, y)
        parent[x_set] = y_set

    num_nodes = len(adj_list)
    edges = []

    # Step 1: Create a list of edges
    for i in range(num_nodes):
        for neighbor, weight in adj_list[i]:
            edges.append((weight, i, neighbor))

    # Step 2: Sort edges by weight
    edges.sort()

    # Step 3: Initialize connected components
    components = [-1] * num_nodes

    # Step 4: Initialize MST
    minimum_spanning_tree = []

    # Step 5: Iterate through sorted edges
    for edge in edges:
        weight, start_node, end_node = edge

        # Check if adding the edge creates a cycle
        if find_parent(components, start_node) != find_parent(components, end_node):
            minimum_spanning_tree.append(edge)

            # Merge the connected components
            old_component, new_component = find_parent(
                components, start_node), find_parent(components, end_node)
            for i in range(num_nodes):
                if find_parent(components, i) == old_component:
                    components[i] = new_component

            # Check if MST is complete (V-1 edges)
            if len(minimum_spanning_tree) == num_nodes - 1:
                break

    return minimum_spanning_tree


print("Number Of Nodes : ", num_nodes)

start_time = time.time()
minimum_spanning_tree_matrix, total_weight_matrix = lazy_prim_matrix(
    adj_matrix)
end_time = time.time()
execution_time_matrix = end_time - start_time


print("Execution Time for Prims Lazy Matrix Algorithm:", execution_time_matrix)
print("-------------------------------------------------------\n")

# Measure execution time for lazy_prim_list
start_time = time.time()
minimum_spanning_tree_list, total_weight_list = lazy_prim_list(adj_list)
end_time = time.time()
execution_time_list = end_time - start_time


print("Execution Time for Prims Lazy List Algorithm:", execution_time_list)
print("-------------------------------------------------------\n")

# Measure execution time for eager_prim_list
start_time = time.time()
minimum_spanning_tree_matrix_eager = prim_mst_eager_matrix(adj_matrix)
end_time = time.time()
execution_time_matrix_eager = end_time - start_time


print("Execution Time for Prims Eager Matrix Algorithm:",
      execution_time_matrix_eager)
print("-------------------------------------------------------\n")


# Measure execution time for eager_prim_list
start_time = time.time()
minimum_spanning_tree_list_eager = prim_mst_eager_list(adj_list)
end_time = time.time()
execution_time_list_eager = end_time - start_time


print("Execution Time for Prims Eager List Algorithm:",
      execution_time_list_eager)
print("-------------------------------------------------------\n")


start_time = time.time()
minimum_spanning_tree_matrix_kruskal = kruskal_matrix(adj_matrix)
end_time = time.time()
execution_time_matrix_kruskal = end_time - start_time


print("Execution Time for Kruskal Matrix Algorithm:",
      execution_time_matrix_kruskal)
print("-------------------------------------------------------\n")

start_time = time.time()
minimum_spanning_tree_list_kruskal = kruskal_list(adj_list)
end_time = time.time()
execution_time_list_kruskal = end_time - start_time


print("Execution Time for Kruskal List Algorithm:", execution_time_list_kruskal)
print("-------------------------------------------------------\n")
