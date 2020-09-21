class Node:
	def __init__(self , start_point):
		self.cost=999999999999999
		self.path=[start_point]

# Function to push an object into the queue
def push_node(node , q):
	i=0
	while(i<len(q) and q[i].cost<node.cost):
		i+=1
	while(i<len(q) and q[i].cost==node.cost and q[i].path<node.path):
		i+=1
	q.insert(i,node)


# function to find the shortest path to any goal state using UCS algorithm

def UCS_Traversal(cost,start_point,goals):
	n=len(cost)-1

	shortest_path_and_cost={}     # stores paths and costs to each goal state

	for i in goals:
		shortest_path_and_cost[i]=Node(start_point)  # initializing path and cost to each goal state

	visited=[0 for i in range(n+1)]    # visited array to check if the node is visited
 
	q=[]    # priotity queue

	node=Node(start_point)
	node.cost=0
	q.append(node)   # adding source to the queue
	
	while(len(q)>0):    # while all nodes are not visited (when the graph is connected)
		node=q[0]
		q.pop(0)
		visited[node.path[-1]]=1   # node.path[-1] gives us the current node 

		for i in range(len(goals)):    # to check if the current node is a goal state and insert path into the dict if the cost of this path is lesser
			if(node.path[-1]==goals[i]):
				if(shortest_path_and_cost[goals[i]].cost > node.cost):
					shortest_path_and_cost[goals[i]].cost=node.cost
					shortest_path_and_cost[goals[i]].path=node.path
				break
		
		for neighbor in range(1,n+1):    # finding all neighbors of the current node and adding them to the queue after updating cost and path
			if(cost[node.path[-1]][neighbor]>0 and visited[neighbor]==0):
				new_node=Node(start_point)
				new_node.cost=node.cost+cost[node.path[-1]][neighbor]
				new_node.path=node.path+[neighbor]
				push_node(new_node,q)

	minn=9999999999  
	
	for i in shortest_path_and_cost.keys():    # finding shortest of the shortest path to each goal states to give the path to a state with the least cost overall
		if(minn > shortest_path_and_cost[i].cost):
			minn=shortest_path_and_cost[i].cost
			min_path=shortest_path_and_cost[i].path
			
	return min_path

def A_star_Traversal(cost,heuristic,start_point,goals):   

	n=len(cost)-1

	visited=[0 for i in range(n+1)]

	q=[]
	node=Node(start_point)
	node.cost=heuristic[start_point]

	min_path=None
	q.append(node)

	while(len(q)>0):

		node=q[0]
		q.pop(0)

		visited[node.path[-1]]=1

		for i in range(len(goals)):
			if(node.path[-1]==goals[i]):
				min_path=node.path
				break
				
		if(min_path!=None):
			break
		
		for neighbor in range(1,n+1):
			if(cost[node.path[-1]][neighbor]>0 and visited[neighbor]==0):
				new_node=Node(start_point)
				new_node.cost=node.cost+cost[node.path[-1]][neighbor]+heuristic[neighbor]-heuristic[node.path[-1]]
				new_node.path=node.path+[neighbor]
				push_node(new_node,q)

	
			
	return min_path
	


def DFS_Traversal(cost, start, goals):
	stack = []  # stack to keep track of nodes
	vis = [False for i in range(len(cost))]  # visit array to check all the visited nodes
	path = []    # path array stores the final path
	stack.append(start)   #initializing the stack

	while(len(stack)):
		u = stack.pop()
		path.append(u)   # add the feasible node to the path
		if(u in goals):
			break
		vis[u] = True    # mark the node as visited
		check = False
		for v in range(len(cost)-1, 0, -1):   # check rest of the nodes in lexicographical order
			if(v != u and (not vis[v]) and cost[u][v]>=0):  # check the constraints for feasibility of the node
				check = True
				stack.append(v)

		if(check == False):
			path.pop()
			
	if(path and path[-1] in goals):
		return path
	else:
		return []

'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''
def tri_traversal(cost, heuristic, start_point, goals):
    l = []

    t1 = DFS_Traversal(cost, start_point, goals)
    t2 = UCS_Traversal(cost, start_point, goals)
    t3 = A_star_Traversal(cost, heuristic,start_point,goals)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l