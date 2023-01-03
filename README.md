# TSP_DFS_BnB

### TSP Problem Background
Given a list of cities and distances between them, the Traveling Salesman Problem asks for the shortest possible route a salesman can take to travel all cities. One of TSP constraints is that the salesman has to start and finish at the same location. On top of that, each city is visited one time only. This problem is presented in code by representing the search space with a 2D array or a matrix. Each digit on row A and column B stands for the distance between city A and city B. For this version of TSP, only symmetrical state spaces are considered. Therefore, the distance from city A to city B is the same as from city B to city A.

### Description of the Branch & Bound Depth First Search (BnB DFS) Algorithm
- The Branch and Bound DFS keeps an upper bound: the current best-known solution and a lower bound:f(n) to estimate the total cost of the route. In our approach, the initial upper bound takes a value supplied by the user. If no value is specified, a default value of maxsize, a variable from the Python system module, is chosen. The initial lower bound is the total distance calculated by f(n) on the root node. The initial root node is set to choose the first node in the input matrix row, which is the first city specified.
- After the initial setup, the algorithm starts expanding on the root node just as the normal depth first search algorithm. The algorithm proceeds by recursively iterating through all of the unexpanded nodes to process the search space tree. When a new node is expanded, its depth is checked against the total number of cities to determine if it has finished expanding on every city. If so, a solution to the TSP has been found, and our upper bound is updated accordingly to any shorter total distance. If not, the current node’s lower bound is calculated using f(n). If the lower bound of the current node has a smaller cost than the upper bound, the node is explored further by recursion. In contrast, if the current node’s lower bound has a higher cost than the upper bound, the node is pruned because its cost is already greater than our best solution, and it is impossible to reach an optimal solution from this node.
- The program repeats the above process recursively to visit the next unexpanded node until the whole matrix has been explored either by expanding or pruning. The result is a complete and optimal solution.


### Instructions to run the program
1) In terminal, install pip3 packages located on top of tsp_bnb_dfs.py (if do not already have modules). For example, to install numpy:<br>
    `
        pip3 install numpy
    `
2) In terminal, cd into the TSP_DFS_BNB folder:<br>
    `
        cd {cs271p_project_path}/TSP_DFS_BNB
    `
3) In terminal, run the TSP_DFS_BNB program with a single matrix file using the following command:<br>
    `
        python3 tsp_bnb_dfs.py <filename>
    `
4) The results get saved into a file located in the same folder as the code.
    The name of the file is "tsp_bnb_dfs_{N}.out" where N is the number of cities the matrix supplied contains.

### Sources
- Scholarly things: (2022, May 11). Travelling salesman problem | Branch and bound | Scholarly things [Video]. YouTube. https://www.youtube.com/watch?v=aaaVm6uUY5A
- Kalev Kask: F 2022, Branch and Bound DFS, lecture Notes, Introduction to Artificial Intelligence CS271P, University of California Irvine, delivered 03 Oct 2022.
- Wikipedia contributors: (2022, November 18). Travelling salesman problem. Wikipedia. https://en.wikipedia.org/wiki/Travelling_salesman_problem
Rai, Anurag. “Traveling Salesman Problem Using Branch and Bound.” GeeksforGeeks, 31 Oct. 2022, https://www.geeksforgeeks.org/traveling-salesman-problem-using-branch-and-bound-2/.
