# def A_star_Traversal(
#     #add your parameters 
# ):
#     l = []

#     return l


from collections import defaultdict

class Node:
	def __init__(self,start_point):
		self.cost=999999999999999
		self.path=[start_point]

def push_node(node,q):
	i=0
	while(i<len(q) and q[i].cost<node.cost):
		i+=1

	if(i<len(q) and q[i].cost==node.cost):
		inserted_flag=0
		for j in range(min(len(node.path),len(q[i].path))):
			if(q[i].path[j] > node.path[j]):
				q.insert(i,node)
				inserted_flag=1
			elif(q[i].path[j] < node.path[j]):
				q.insert(i+1,node)
				inserted_flag=1

		if(inserted_flag==0):
			if(len(node.path)>len(q[i].path)):	
				q.insert(i+1,node)
			else:
				q.insert(i,node)
	else:
		q.insert(i,node)

def UCS_Traversal(cost,start_point,goals):
	# print("hIHIH")
	n=len(cost)-1

	# print("n",n,goals)
	shortest_path_and_cost=defaultdict()
	for i in goals:
		shortest_path_and_cost[i]=Node(start_point)

	visited=[0 for i in range(n+1)]
	# print(visited)
	q=[]
	node=Node(start_point)
	node.cost=0
	q.append(node)
	while(len(q)>0):
		# print(len(q))
		# for i in q:
		# 	print("HIHIHIH",i.cost,i.path)
		node=q[0]
		q.pop(0)
		visited[node.path[-1]]=1

		for i in range(len(goals)):
			if(node.path[-1]==goals[i]):
				# print("IHIII",shortest_path_and_cost[goals[i]].cost,node.cost)
				if(shortest_path_and_cost[goals[i]].cost > node.cost):
					shortest_path_and_cost[goals[i]].cost=node.cost
					shortest_path_and_cost[goals[i]].path=node.path
				break
		
		for neighbor in range(1,n+1):
			try:
				if(cost[node.path[-1]][neighbor]>0 and visited[neighbor]==0):
					new_node=Node(start_point)
					new_node.cost=node.cost+cost[node.path[-1]][neighbor]
					new_node.path=node.path+[neighbor]
					push_node(new_node,q)
					# print("ANDAR")
					# for i in q:
					# 	print("HIHIHIH",i.cost,i.path)
			except Exception as e:
				print(e)
				print(neighbor,visited,node.path[-1])
	minn=9999999999
	for i in shortest_path_and_cost.keys():
		if(minn > shortest_path_and_cost[i].cost):
			minn=shortest_path_and_cost[i].cost
			min_path=shortest_path_and_cost[i].path
			
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
		for v in range(1, len(cost)):   # check rest of the nodes in lexicographical order
			if(v != u and (not vis[v]) and cost[u][v]>=0):  # check the constraints for feasibility of the node
				stack.append(v)
				break           # on getting a feasible nodes stop further checking

	return path

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
    t3 = 0
#     t2 = UCS_Traversal
#     #send whatever parameters you require 
# )
#     t3 = A_star_Traversal(
#     #send whatever parameters you require 
# )

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l