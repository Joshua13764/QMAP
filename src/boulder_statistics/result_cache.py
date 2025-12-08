from dataclasses import dataclass, field
from pathlib import Path
from pickle import dump, load
from typing import Any, Callable, Type

from stablehash import stablehash

from boulder_statistics.step_base import StepBase


@dataclass(frozen=True)
class ResultCache[V]():
    cache_folder: Path
    result_type: Type[V]
    verbose: bool = field(default=False)
    hash_method: Callable[[Any], str] = field(
        default=lambda obj: stablehash(obj).hexdigest())

    def get_result_cache_path(self, obj: StepBase, save_prefix: str) -> Path:

        hashable_obj: tuple[Any, ...] = obj.cleaned_hashable

        if self.verbose:
            for i in hashable_obj:
                print(f"{self.hash_method(i)}\t -> {type(i)}\t -> {i}")

        hashed_obj = stablehash(hashable_obj)

        return Path(*self.cache_folder.parts,
                    f"{save_prefix}-hash-{self.hash_method(hashed_obj)}.pkl")

    def does_result_cache_exist(self, obj: StepBase, save_prefix: str) -> bool:
        return self.get_result_cache_path(obj, save_prefix).exists()

    def open_result_cache(self, obj: StepBase,
                          save_prefix: str) -> V:
        result_cache_path: Path = self.get_result_cache_path(obj, save_prefix)

        if self.does_result_cache_exist(obj, save_prefix) == False:
            raise FileExistsError(
                f"Cache file {result_cache_path} doesn't exist")

        with result_cache_path.open("rb") as f:
            cached_result: Any = load(f)

        if isinstance(cached_result, self.result_type):
            return cached_result
        else:
            raise TypeError(
                f"""The cache found in {result_cache_path} is not of expected type {
                    type(self.result_type)} but instead of type {
                    type(cached_result)}""")

    def save_result_cache(self, result_cache_path: Path,
                          result_cache: V) -> None:

        result_cache_path.parent.mkdir(parents=True, exist_ok=True)

        with result_cache_path.open("wb") as f:
            dump(result_cache, f)
