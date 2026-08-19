from itertools import product
from typing import Tuple, Callable

from core.utils.parallelizer import Parallelizer



class GridSearcher:
    def __init__(self, 
            params_grid: dict[list], 
            update_function: Callable[..., dict],
            eval_function: Callable[..., float],
            show_progress: bool = False,
        ):
        self.params_grid = params_grid
        self.show_progress = show_progress
        self.update_function = update_function
        self.eval_function = eval_function
        self.parallelizer = Parallelizer(use_processes=False)

        
    def eval(self, combination: dict, params: dict) -> float:
        updated_params = self.update_function(combination, **params)
        score = self.eval_function(**updated_params)
        return score, combination

    def fit(self, 
            params: dict,
        ) -> Tuple[dict, float]:
        
        keys = list(self.params_grid.keys())
        values = list(self.params_grid.values())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        best_score: float = float('-inf')
        best_params: dict = {}
        
        results = self.parallelizer.run(
            func=self.eval,
            iterable=(
                (combination, params) 
                for combination in combinations
            ),
            show_progress=self.show_progress,
            unpack_args=True
        )
                
        for score, params in results:
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score




