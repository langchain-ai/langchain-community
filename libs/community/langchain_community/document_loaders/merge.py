from typing import AsyncIterator, Iterator, List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class MergedDataLoader(BaseLoader):
    """Merge documents from a list of loaders"""

    def __init__(self, loaders: List):
        """Initialize with a list of loaders"""
        self.loaders = loaders

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load docs from each individual loader."""
        for loader in self.loaders:
            # Check if lazy_load is implemented
            try:
                data = loader.lazy_load()
            except NotImplementedError:
                data = loader.load()
            for document in data:
                yield document

    @staticmethod
    async def _combine_async_iterables(
        *async_iterables: AsyncIterable[Any],
    ) -> AsyncIterator[Any]:
        """Yield from multiple async iterables as soon as items are available."""
        iterators: List[AsyncIterator[Any]] = [
            iterable.__aiter__() for iterable in async_iterables
        ]
        pending: dict[asyncio.Task[Any], AsyncIterator[Any]] = {
            asyncio.create_task(it.__anext__()): it for it in iterators
        }
        try:
            while pending:
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    iterator = pending.pop(task)
                    try:
                        value = task.result()
                    except StopAsyncIteration:
                        continue  # This iterator is exhausted
                    else:
                        yield value
                        # Schedule the next item from this iterator
                        pending[asyncio.create_task(iterator.__anext__())] = iterator
        finally:
            for task in pending:
                task.cancel()

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Lazy load docs from each individual loader."""
        generators = [loader.alazy_load() for loader in self.loaders]
        combined_generator = self._combine_async_iterables(*generators)
        async for document in combined_generator:
            yield document
