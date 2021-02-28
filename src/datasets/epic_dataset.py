import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Set
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar

import numpy as np
import PIL.Image
from gulpio2 import GulpDirectory

from .video_dataset import VideoDataset
from .video_dataset import VideoRecord

MetadataDict = Dict[str, Any]
FramesTypeVar = TypeVar("FramesTypeVar")

LOG = logging.getLogger(__name__)

# line_profiler injects a "profile" into __builtins__. When not running under
# line_profiler we need to inject our own passthrough
if type(__builtins__) is not dict or "profile" not in __builtins__:
    profile = lambda f: f


class GulpVideoRecord(VideoRecord):
    """
    SegmentRecord for a video segment stored in a gulp file.

    Assumes that the video segment has the following metadata in the gulp file:
    - id
    - num_frames
    """

    @property
    def metadata(self):
        return self._metadata

    def __init__(
        self,
        gulp_id: str,
        gulp_metadata_dict: Dict[str, Any],
    ):
        self._metadata = gulp_metadata_dict
        self.gulp_index = gulp_id

    @property
    def num_frames(self) -> int:
        return self.metadata["num_frames"]


class EpicVideoDataset(VideoDataset):
    def __init__(
        self,
        gulp_path: Path,
        sample_transform: Optional[Callable[[PIL.Image.Image], FramesTypeVar]] = None,
        filter_fn: Optional[Callable[[str], bool]] = None,
        drop_problematic_metadata: bool = True,
    ):
        """

        Args:
            gulp_path: Path to gulped dataset
            sample_transform: Transformation applied to sampled `[PIL.Image]` at a specific index
            filter_fn: A callable that is used to remove examples from the dataset.
                It should return whether or not the given sample (by id) should be
                kept or not.
            drop_problematic_metadata: Drop metadata entries that are non-scalar or
                have ``None`` values which cause the default pytorch collation function
                to error.
        """
        super().__init__()
        assert gulp_path.exists(), "Could not find the path {}".format(gulp_path)
        self.gulp_dir = GulpDirectory(str(gulp_path.absolute()))
        self.filter_fn = filter_fn
        if sample_transform is None:
            self.sample_transform = lambda x: x
        else:
            self.sample_transform = sample_transform
        self._drop_problematic_metadata = drop_problematic_metadata
        self._video_records = self._read_video_records(
            self.gulp_dir.merged_meta_dict, filter_fn
        )
        self._video_records_list: List[VideoRecord] = list(self._video_records.values())

    @property
    def video_records(self) -> List[VideoRecord]:
        return self._video_records_list

    @profile
    def load_frames(
        self, record: VideoRecord, indices: List[int]
    ) -> List[FramesTypeVar]:
        selected_frames: List[FramesTypeVar] = []
        for i in indices:
            # Without passing a slice to the gulp directory index we load ALL the frames
            # so we create a slice with a single element -- that way we only read a single frame
            # from the gulp chunk, and not the whole chunk.
            frames = self._sample_video_at_index(cast(GulpVideoRecord, record), i)
            frames = self.sample_transform(frames)
            selected_frames.extend(frames)
        return selected_frames

    def __len__(self) -> int:
        return len(self._video_records)

    def _read_video_records(
        self,
        gulp_dir_meta_dict,
        filter_fn: Optional[Callable[[str], bool]],
    ) -> Dict[str, GulpVideoRecord]:
        video_records = OrderedDict()
        dropped = 0
        for video_id in gulp_dir_meta_dict:
            if filter_fn is None or filter_fn(video_id):
                meta_dict = gulp_dir_meta_dict[video_id]["meta_data"][0]
                narration_timestamp_col = "narration_timestamp"
                if narration_timestamp_col in meta_dict:
                    narration_timestamp = meta_dict[narration_timestamp_col]
                    if isinstance(narration_timestamp, float) and np.isnan(
                        narration_timestamp
                    ):
                        # mixing dtypes in a field causes issues when we collate
                        # batches. So we replace NaN with 'UNKNOWN'
                        meta_dict[narration_timestamp_col] = "UNKNOWN"
                video_records[video_id] = GulpVideoRecord(video_id, meta_dict)
            else:
                dropped += 1
        if dropped > 0:
            LOG.info(f"Dropped {dropped}/{len(gulp_dir_meta_dict)} examples")
        if self._drop_problematic_metadata:
            self._filter_problematic_metadata_fields(video_records)
        return video_records

    def _filter_problematic_metadata_fields(
        self, video_records: Dict[Any, VideoRecord]
    ) -> None:
        """Drops metadata whose value is a non-scalar value or ``None``"""
        problematic_fields = self._determine_problematic_fields(video_records)
        for record in video_records.values():
            for field in problematic_fields:
                del record.metadata[field]

    def _determine_problematic_fields(
        self, video_records: Dict[Any, VideoRecord]
    ) -> Set[str]:
        def is_problematic_value(v: Any):
            return isinstance(v, (tuple, list)) or v is None

        problematic_fields = set()
        for record in video_records.values():
            for k, v in record.metadata.items():
                if is_problematic_value(v):
                    problematic_fields.add(k)
        return problematic_fields

    @profile
    def _sample_video_at_index(
        self, record: GulpVideoRecord, index: int
    ) -> List[PIL.Image.Image]:
        single_frame_slice = slice(index, index + 1)
        numpy_frame = self.gulp_dir[record.gulp_index, single_frame_slice][0][0]
        return [PIL.Image.fromarray(numpy_frame).convert("RGB")]


class EpicVideoFlowDataset(EpicVideoDataset):
    def _sample_video_at_index(
        self, record: GulpVideoRecord, index: int
    ) -> List[PIL.Image.Image]:
        # Flow pairs are stored in a contiguous manner in the gulp chunk:
        # [u_1, v_1, u_2, v_2, ..., u_n, v_n]
        # so we have to convert our desired frame index i to the gulp
        # indices j by j = (i * 2, (i + 1) * 2)
        flow_pair_slice = slice(index * 2, (index + 1) * 2)
        numpy_frames = self.gulp_dir[record.gulp_index, flow_pair_slice][0]
        frames = [
            PIL.Image.fromarray(numpy_frame).convert("L")
            for numpy_frame in numpy_frames
        ]
        return frames
