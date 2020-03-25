import dataclasses
import itertools
import multiprocessing as mp
import os
from typing import Any, Callable, Iterable, List, cast

import tqdm


@dataclasses.dataclass
class Result:
    length: int
    items: List[Any]


def chunk(n: int, iterable: Iterable):
    """Collect data into chunks of size, enumerate them, and filter out fill values.
            grouper('ABCDEFG', 3, 'x') --> (0, ABC) (1, DEF) (2, G)"
    """
    args = [iter(iterable)] * n
    batches = itertools.zip_longest(*args, fillvalue=None)
    enumerated_and_filtered = []
    for idx, batch in enumerate(batches):
        enumerated_and_filtered.append((idx, list(filter(lambda elem: elem is not None, batch))))
    return enumerated_and_filtered


def execute(func: Callable, items: List[Any],
            batch_size: int = 20, workers: int = os.cpu_count(), debug=False) -> List[Any]:
    """Execute a callable over the iterable in parallel."""
    results = []
    kwargs = {
        'total': len(items),
        'leave': True
    }

    batches = chunk(batch_size, items)
    if debug:
        for result in tqdm.tqdm(map(func, batches), total=len(batches)):
            result = cast(Result, result)
            results.extend(result.items)
    else:
        with mp.Pool(workers) as pool:
            with tqdm.tqdm(**kwargs) as pbar:
                for result in pool.imap_unordered(func, batches):
                    result = cast(Result, result)
                    pbar.update(result.length)
                    results.extend(result.items)

    return results
