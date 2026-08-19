from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Any
from tqdm import tqdm

class Parallelizer:
    def __init__(self, max_workers: int = None, use_processes: bool = True):
        self.max_workers = max_workers
        self.pool_executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    def run(self, func: Callable, iterable: Iterable[Any], show_progress: bool = True, unpack_args: bool = False) -> list:
        results = []
        
        with self.pool_executor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(func, *item) if unpack_args else executor.submit(func, item): item for item in iterable}
            futures_iter = as_completed(futures)
            if show_progress:
                futures_iter = tqdm(futures_iter, total=len(futures), desc="Processing")

            for future in futures_iter:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    raise RuntimeError(f"[ERROR] Processing failed for future {future}: {e}")

        return results