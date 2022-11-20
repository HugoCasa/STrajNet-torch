import warnings
import typing

import numpy as np
import torch

from tfrecord import reader

def shuffle_iterator(iterator: typing.Iterator,
                     queue_size: int,
                     gpu_id: int,
                     world_size: int) -> typing.Iterable[typing.Any]:
    buffer = []
    i = 0
    try:
        while len(buffer) < queue_size:
            val = next(iterator)
            if i % world_size == gpu_id: 
                # make sure not to get the same sample on different devices
                buffer.append(val)
            i += 1
        for _ in range(queue_size):
            buffer.append(next(iterator))
    except StopIteration:
        warnings.warn("Number of elements in the iterator is less than the "
                      f"queue size (N={queue_size}).")
    while buffer:
        index = np.random.randint(len(buffer))
        try:
            item = buffer[index]
            while i % world_size != gpu_id: 
                # make sure not to get the same sample on different devices
                val = next(iterator)
                i += 1
            buffer[index] = val
            yield item
        except StopIteration:
            yield buffer.pop(index)

class DistributedMultiTFRecordDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 data_pattern: str,
                 index_pattern: typing.Union[str, None],
                 splits: typing.Dict[str, float],
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 shuffle_queue_size: typing.Optional[int] = None,
                 transform: typing.Callable[[dict], typing.Any] = None,
                 sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 compression_type: typing.Optional[str] = None,
                 infinite: bool = True,
                 gpu_id: int = None,
                 world_size: int = None
                 ) -> None:
        super(DistributedMultiTFRecordDataset, self).__init__()
        self.data_pattern = data_pattern
        self.index_pattern = index_pattern
        self.splits = splits
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform
        self.compression_type = compression_type
        self.infinite = infinite
        self.gpu_id = gpu_id
        self.world_size = world_size

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:
        #     np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        it = reader.multi_tfrecord_loader(data_pattern=self.data_pattern,
                                          index_pattern=self.index_pattern,
                                          splits=self.splits,
                                          description=self.description,
                                          sequence_description=self.sequence_description,
                                          compression_type=self.compression_type,
                                          infinite=self.infinite,
                                         )
        if self.shuffle_queue_size:
            it = shuffle_iterator(it, self.shuffle_queue_size, self.gpu_id, self.world_size)
        if self.transform:
            it = map(self.transform, it)
        return it
