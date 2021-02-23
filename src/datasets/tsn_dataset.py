import logging
from typing import Callable
from typing import List


import numpy as np
import torch.utils.data

from .video_dataset import VideoDataset
from .video_dataset import VideoRecord

LOG = logging.getLogger(__name__)

# line_profiler injects a "profile" into __builtins__. When not running under
# line_profiler we need to inject our own passthrough
if type(__builtins__) is not dict or "profile" not in __builtins__:
    profile = lambda f: f


class TsnDataset(torch.utils.data.Dataset):
    """
    Wraps a :class:`VideoDataset` to implement TSN sampling
    """

    def __init__(
        self,
        dataset: VideoDataset,
        num_segments: int = 3,
        segment_length: int = 1,
        transform: Callable = None,
        random_shift: bool = True,
        test_mode: bool = False,
    ):
        """

        Args:
            dataset: Video dataset to load TSN-sampled segments from.
            num_segments: Number of segments per clip.
            segment_length: Length of segment in number of frames.
            transform: A applied to the list of frames sampled from the clip
            random_shift:
            test_mode: Whether to return center sampled frames from each segment.
        """
        self.dataset = dataset
        self.num_segments = num_segments
        self.segment_length = segment_length
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

    def __getitem__(self, index):
        record = self.dataset.video_records[index]
        if self.test_mode:
            segment_start_idxs = self._get_test_indices(record)
        else:
            segment_start_idxs = (
                self._sample_indices(record)
                if self.random_shift
                else self._get_val_indices(record)
            )
        return self._get(record, segment_start_idxs)

    def __len__(self):
        return len(self.dataset)

    @profile
    def _get(self, record: VideoRecord, segment_start_idxs: List[int]):
        images = self.dataset.load_frames(
            record, self._get_frame_idxs(segment_start_idxs, record)
        )
        if self.transform is not None:
            images = self.transform(images)
        metadata = record.metadata
        return images, metadata

    def _sample_indices(self, record: VideoRecord):
        average_duration = (
            record.num_frames - self.segment_length + 1
        ) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration
            ) + np.random.randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(
                np.random.randint(
                    record.num_frames - self.segment_length + 1, size=self.num_segments
                )
            )
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record: VideoRecord):
        if record.num_frames > self.num_segments + self.segment_length - 1:
            tick = (record.num_frames - self.segment_length + 1) / float(
                self.num_segments
            )
            offsets = np.array(
                [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
            )
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record: VideoRecord):
        tick = (record.num_frames - self.segment_length + 1) / float(self.num_segments)
        offsets = np.array(
            [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]
        )
        return offsets

    def _get_frame_idxs(
        self, segment_start_idxs: List[int], record: VideoRecord
    ) -> List[int]:
        seg_idxs = []
        for seg_ind in segment_start_idxs:
            p = int(seg_ind)
            for i in range(self.segment_length):
                seg_idxs.append(p)
                if p < record.num_frames:
                    p += 1
        return seg_idxs
