"""
Author: Moayad Alnammi
"""

"""
    We will use cProfile to get some function call stats. 
    Particularly, we want to count the number of times a node is popped from frontier and is generated. 
    To count the number of times a node is popped, we just need to count number of times pop() is called.
    To count number of times a node is generated, notice that we call p.result.
    So, we just need to count number of times result is called for a Problem object.
"""
import cProfile, pstats, io
from pstats import SortKey   
def run_profiler_searches(problem, searchers_list, searcher_names_list):
    problem_name = str(problem)[:28]
    total_gen_node, total_pop_node = 0, 0
    print('\nProfiler for problem: {}\n'.format(problem_name))
    for search_algo, search_algo_name in zip(searchers_list, searcher_names_list):
        pr = cProfile.Profile()
        pr.enable()
        goal_node = search_algo(problem)
        pr.disable()
        
        io_stream = io.StringIO()
        ps = pstats.Stats(pr, stream=io_stream).sort_stats(SortKey.CALLS).strip_dirs()
        ps.print_stats("pop*|result*")
        profiler_string = io_stream.getvalue()
        
        gen_node_count, pop_node_count = profiler_splitter(profiler_string)
        total_gen_node += gen_node_count
        total_pop_node += pop_node_count
        
    
        print('{:15s} {:9,d} generated nodes |{:9,d} popped |{:5.0f} solution cost |{:8,d} solution depth|'.format(
              search_algo_name, gen_node_count, pop_node_count, goal_node.path_cost, goal_node.depth))
    
    
    print('{:15s} {:9,d} generated nodes |{:9,d} popped'.format('TOTAL', total_gen_node, total_pop_node))
    print('________________________________________________________________________')


def profiler_splitter(profiler_data):
    profiler_data = profiler_data.split("\n")
    result_line = [l for l in profiler_data if 'result' in l][1]
    pop_line = [l for l in profiler_data if 'pop' in l][1]
    
    splitter = (lambda s: int(s.split()[0].split("/")[0]) )
    result_calls = splitter(result_line)
    pop_calls = splitter(pop_line)
    
    return result_calls, pop_calls 
    
if __name__ == "__main__":
    from sliding_block import *
    from search_algorithms import *
    
    # compare A* with null-h, h1, and h2
    import numpy as np
    np.random.seed(202413221)
    for _ in range(5):
        random_board = tuple(np.random.permutation(9))
        example_slidingblock_problem = SlidingBlock(initial_state=random_board)
        while not example_slidingblock_problem.check_solvability(random_board):
            random_board = tuple(np.random.permutation(9))
            example_slidingblock_problem = SlidingBlock(initial_state=random_board)
            
        print(f'Initial board: {random_board}')
        
        # get some statistics on generated nodes, popped nodes, solution
        searchers = [(lambda p: uniform_cost_search(p, treelike=False)),
                     (lambda p: astar_search(p, h=p.h1, treelike=False)),  
                     (lambda p: astar_search(p, h=p.h2, treelike=False))]
        searcher_names= ['Graph-like UCS null-heuristic', 
                         'Graph-like A* h1', 
                         'Graph-like A* h2']
                         
        run_profiler_searches(example_slidingblock_problem, searchers, searcher_names)