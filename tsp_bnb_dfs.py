# Python platform’s pointer that dictates the maximum size of lists and strings in Python
from sys import maxsize, exit, argv
from string import ascii_letters
import numpy as np
from time import time

# Helper methods
def create_adjacency_matrix(N, start=0, end=100, print_data=False) -> np.array:
        '''
        #### Creates an adjacency or symmetric matrix of size N.
        * N = desired size of matrix.
        * start = starting value (inclusive) of the range of values to randomly choose numbers from.
        * end = ending value (exclusive) of the range of values to randomly choose numbers from.
        * print_data is a utility option to print the matrix before returning it.
        '''
        matrix = []
        node_domains = {}
        for i in range(N):
                index = str(i)
                domain = [] # list of tuples (node, distance_to_node)
                matrix.append([])
                for j in range(N):
                        distance_to_node = 0
                        if i != j:
                                distance_to_node = np.random.randint(start, end)
                        if str(j) in list(node_domains.keys()):
                                distance_to_node = node_domains[str(j)][i][1]
                        matrix[i].append(distance_to_node)
                        node = j
                        current_node = (node, distance_to_node)
                        domain.append(current_node)
                node_domains[index] = domain
        matrix = np.array(matrix)
        
        if print_data is True:
                print("The Matrix:")
                print(matrix, end='\n\n')
                print(f"The shape of the matrix = {matrix.shape}\n")

        return matrix

def write_results_to_file(file_name, upper_bound, start_time, end_time, solution, ascii_solution=None) -> None:
        '''
        #### Writes results of TSP DFS BnB into a file.
        * file_name = path of the file, including name of the file. 
                Creates a new file if does not already exist, otherwise overwrites current one.
        * upper_bound = represents a limit of which a solution will never exceed.
        * start_time = time when the algorithm began executing.
        * end_time = time when the algorithm stopped executing.
        * solution = final solution or the path for the traveling salesman to take. 
        '''
        with open(file_name, 'w+') as file:
                file.write(f"Total Path Cost: {upper_bound}\n")
                file.write(f'{(end_time - start_time) * 1000:1.4f}ms\n')
                str_solution = ' '.join([str(s) for s in solution])
                file.write(f'{str_solution}\n')
                if ascii_solution != None:
                        str_ascii_solution = ' '.join(ascii_solution)
                        file.write(f'Ascii character version of solution:\n{str_ascii_solution}')

def load_matrix(file_name) -> np.array:
        '''
        #### Loads a matrix represented in a text file, delimiter is a space character.
        * file_name = path (including filename) of the file containing space delimited matrix. 
        '''
        return np.loadtxt(
                file_name,
                dtype='float',
                delimiter=' ',
                skiprows=1
        )
        
# Traveling Salesman Problem solved using the Depth First Search Branch & Bound Algorithm.
class TSP_BNB_DFS:
        '''
                #### Traveling Salesman Problem solved using the Depth First Search Branch & Bound Algorithm.
                * tree = 2D list (array) of integers of size NxN that represent the tree.
                * N = the number of Nodes in the tree.
                * initial_upper_bound = the intial upper bound, default value is set to maxsize.
                * ascii_format = Boolean, set to true if want to view final path in ascii character form.
                * print_frontier = Boolean, set to true if want to view initial frontier. 
        '''
        def __init__(self, tree, N, initial_upper_bound=maxsize, ascii_format=False, print_frontier=False) -> None:
                # ensure the matrix is symmetrical, otherwise call exit()
                self._is_adjacency_matrix(tree)

                self.tree = tree
                self.N = N

                # final_path[] stores the final solution i.e. a path of the salesman such as 0 3 4 1 2
                self.final_path = [None] * (self.N + 1)
                
                # visited[] keeps track of the already visited nodes in a particular path i.e. [True, False...]
                self.visited = [False] * N 

                # set upper bound to upper bound set by user
                self.upper_bound = initial_upper_bound

                # intial lower bound is calculated in tsp()
                self.lower_bound = None

                # stores the distance of the path so far 
                self.current_distance = 0

                # keeps track of which row in the matrix, first row is 
                self.current_row_level = 0

                # stores the best known solution 
                self.current_path = []

                self.ascii_format = ascii_format
                if self.ascii_format is True or print_frontier is True:
                        # domain and frontier are only for the initial state since maintaining it will decrease performance
                        self.domain = []
                        self.frontier = [] # each key is a node and the ndoe value is its LIFO Queue (Stack) of distances to other nodes
                        self._construct_frontier()

                if print_frontier is True:
                        print("Nodes and their domains: ")
                        print(self.frontier)

         # helper methods
        def _construct_frontier(self):
                counter = 1
                ascii_index = 0
                for index, _ in enumerate(self.tree):
                        if index > len(ascii_letters) - 1:
                                self.domain.append(ascii_letters[ascii_index] + '_' + str(counter))
                                ascii_index += 1
                                if ascii_index == len(ascii_letters):
                                        ascii_index = 0
                                        counter += 1
                        else:
                                self.domain.append(ascii_letters[index])
                
                for i, node in enumerate(self.domain):
                        node_domain_values = []
                        for j, distance in enumerate(self.tree[i]):
                                node_domain_values.append((self.domain[j], distance))
                        self.frontier.append({})
                        self.frontier[i]['node'] = node
                        self.frontier[i]['domain'] = node_domain_values

        def _is_adjacency_matrix(self, matrix) -> bool:
                for i, row in enumerate(matrix):
                        for j, column in enumerate(row):
                                try:
                                        assert matrix[i][j] == matrix[j][i]
                                except AssertionError as err:
                                        print("Error: The graph is not an adjanecy matrix (not symmetrical distances).")
                                        print(matrix)
                                        exit("Program is terminating because matrix does not follow requirements.")
                return True

        # matrix is the tree of the tsp
        # i = first index value, tells which row to look in
        def _first_min(self, matrix, i):
                '''
                Function to find the first minimum edge cost
                of an edge connected to node i
                '''
                min = self.upper_bound
                for k in range(self.N):
                        if matrix[i][k] < min and i != k:
                                min = matrix[i][k]
                return min
        
        # matrix is the tree of the tsp
        # i = first index value, tells which row to look in
        def _second_min(self, matrix, i):
                '''
                function to find the second minimum edge cost
                of an edge connected to node i       
                '''

                # calculate the first minimum cost first to know which edge to ignore
                first, second = self.upper_bound, self.upper_bound
                for j in range(self.N):
                        if i == j:
                                continue
                        if matrix[i][j] <= first:
                                second = first
                                first = matrix[i][j]
                
                        elif(matrix[i][j] <= second and matrix[i][j] != first):
                                second = matrix[i][j]
                
                return second

        # accessor method
        def get_final_path(self) -> list:
                return self.final_path

        # algorithm functions
        def _tsp_recursive(self, level):
                '''
                Recursive function that takes one argument:
                level-> current row level of 2D matrix (tree) while moving 
                '''
                
                # base case is when we have reached level N - indicates covered all the nodes once
                if level == self.N:
                        
                        # check if there is an edge from last node in path back to the first node
                        if self.tree[self.current_path[level - 1]][self.current_path[0]] != 0:
                                
                              
                                # current_upper_bound - total distance of current solution
                                current_upper_bound = self.current_distance + self.tree[self.current_path[level - 1]][self.current_path[0]]
                        
                                if current_upper_bound < self.upper_bound:
                                        # copy current path to be final solution
                                        self.final_path[:self.N + 1] = self.current_path[:]
                                        self.final_path[self.N] = self.current_path[0]
                                        self.upper_bound = current_upper_bound
                        # end the recursion loop
                        return None
                             
                # for any other level, recursively iterate through all of its nodes to build the search space tree
                for i in range(self.N):
                        
                        # only consider next node if it is not same 
                        # ensure not a diagonal entry in adjacency self.tree and not already visited
                        if self.tree[self.current_path[level-1]][i] != 0 and self.visited[i] == False:
                                temp = self.lower_bound
                                self.current_distance += self.tree[self.current_path[level - 1]][i]
                
                                
                                # different calculation of self.lower_bound for level 1 vs other levels
                                if level == 1:
                                        self.lower_bound -= ((self._first_min(self.tree, self.current_path[level - 1]) + self._first_min(self.tree, i)) / 2)
                                else:
                                        self.lower_bound -= ((self._second_min(self.tree, self.current_path[level - 1]) + self._first_min(self.tree, i)) / 2)
                        
                                
                                # lower bound of current node = self.lower_bound + self.current_distance
                                # if current lower bound < self.upper_bound, explore the node further
                                if self.lower_bound + self.current_distance < self.upper_bound:
                                        self.current_path[level] = i
                                        self.visited[i] = True
                                        
                                        # call _tsp_recursive for the next level
                                        self._tsp_recursive(level + 1)
                        
                                
                                # if current lower bound >= self.upper_bound, prune the node
                                # pruning of the node is done by resetting all changes to
                                # self.current_distance, self.lower_bound, self.visited array
                                self.current_distance -= self.tree[self.current_path[level - 1]][i]
                                self.lower_bound = temp
                                self.visited = [False] * len(self.visited)
                                for j in range(level):
                                        if self.current_path[j] != -1:
                                                self.visited[self.current_path[j]] = True                        

        # This function sets up final_path
        def tsp(self) -> None:
                '''
                Initialize the current_path and visited array.
                Calculate initial lower bound for the root node.
                '''

                intial_lower_bound = 0
                # set every visited node to -1 (-1 means not visited) except for root node.
                self.current_path = [-1 for _ in range(self.N + 1)]
                
                # calculate initial bound 1/2 * (sum of first min + second min)
                # iterate through each node that the root can visit and choose the least cost combination
                for i in range(self.N):
                        intial_lower_bound += (self._first_min(self.tree, i) + self._second_min(self.tree, i))
                
                # make path costs integer number
                intial_lower_bound = intial_lower_bound / 2
                
                self.lower_bound = intial_lower_bound
                self.visited[0] = True # set root visited to True
                self.current_path[0] = 0 # set root equal to 0 since takes 0 cost to go root from root 

                # move on 2nd row before entering recursion
                self.current_row_level += 1

                # call recursive method of algorithm
                self._tsp_recursive(self.current_row_level)
        
        def simulate(self) -> None:
                start_time = time()
                self.tsp()
                end_time = time()
                print("Total Path Cost:", self.upper_bound)
                print("Path Taken [integer form]:", end = ' ')
                for i in range(self.N + 1):
                        print(self.final_path[i], end = ' ')
                print("")
                ascii_solution = []
                if self.ascii_format is True:
                        print("Path Taken [letter form]:", end = ' ')
                        for i in self.final_path:
                                print(self.frontier[i]['node'], end= ' ')
                                ascii_solution.append(self.frontier[i]['node'])
                        print("")
                        print("Path Taken:", end = ' ')
                        for i in self.final_path:
                                if i <= len(self.final_path) - 1:
                                        print(self.frontier[i]['node'], end= ' -> ')
                        print("Complete")
                if self.ascii_format is True:
                        write_results_to_file(f"./tsp_bnb_dfs_{self.N}.out", self.upper_bound, start_time, end_time, self.final_path, ascii_solution)
                else:
                        write_results_to_file(f"./tsp_bnb_dfs_{self.N}.out", self.upper_bound, start_time, end_time, self.final_path)

# hardcoded example matrices to test
adjanecy_matrix = [
       # a, b, c, d, e
        [0, 3, 1, 5, 8], # a
        [3, 0, 6, 7, 9], # b
        [1, 6, 0, 4, 2], # c
        [5, 7, 4, 0, 3], # d
        [8, 9, 2, 3, 0]  # e
]
adjanecy_matrix_2 = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
]

# this matrix can be used to demonstrate the program fails if it is not an adjacency matrix (symmetrical) 
non_adjanecy_matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
]

def main(argc: int, argv: list[str]) -> None:
        if argc == 1:
                exit("Error: Please supply the path of the filename containing the matrix as an argument.\nExecute as follows: python tsp_bnb_dfs <filename containing matrix>.")
        if argc > 2:
                exit("Error: Too many arguments were passed to the program.\nExecute as follows: python tsp_bnb_dfs <filename containing matrix>.")
        fname = argv[1]
        tree = load_matrix(fname)
        N = len(tree)
        tsp = TSP_BNB_DFS(tree, N)
        tsp.simulate()

if __name__ == '__main__':
        main(len(argv), argv)