from abc import ABC
from typing import List

import PIL.Image
from torch.utils.data import Dataset


class VideoRecord(ABC):
    """
    Represents a video segment with an associated label.
    """

    @property
    def metadata(self):
        raise NotImplementedError()

    @property
    def num_frames(self) -> int:
        raise NotImplementedError()


class VideoDataset(Dataset, ABC):
    """
    A dataset interface for use with :class:`TsnDataset`. Implement this interface if you
    wish to use your dataset with TSN.

    We cannot use torch.utils.data.Dataset because we need to yield information about
    the number of frames per video, which we can't do with the standard
    torch.utils.data.Dataset.
    """
    @property
    def video_records(self) -> List[VideoRecord]:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def load_frames(
        self, metadata: VideoRecord, idx: List[int]
    ) -> List[PIL.Image.Image]:
        raise NotImplementedError()
