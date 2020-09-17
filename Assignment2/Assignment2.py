# def A_star_Traversal(
#     #add your parameters 
# ):
#     l = []

#     return l

# def UCS_Traversal(
#     #add your parameters 
# ):
#     l = []

#     return l

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
    t2 = 0
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