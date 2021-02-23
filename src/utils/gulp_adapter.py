from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List

import pandas as pd
from gulpio2.adapters import AbstractDatasetAdapter
from gulpio2.utils import resize_images


Result = Dict[str, Any]


class EpicDatasetAdapter(AbstractDatasetAdapter):
    """Gulp Dataset Adapter for Gulping RGB frames extracted from the EPIC-KITCHENS dataset"""

    def __init__(
        self,
        video_segment_dir: str,
        annotations_df: pd.DataFrame,
        frame_size: int = -1,
        extension: str = "jpg",
    ) -> None:
        """Gulp all action segments in  ``annotations_df`` reading the dumped frames from
        ``video_segment_dir``

        Args:
            video_segment_dir:
                Root directory containing segmented frames::

                    frame-segments/
                    ├── P01
                    │   ├── P01_01
                    │   │   ├── frame_0000000001.jpg
                    │   │   ...
                    │   │   ├── frame_0000012345.jpg
                    │   ...
            annotations_df:
                DataFrame containing labels to be gulped. Must have at least a
                start_frame and stop_frame column.
            frame_size:
                Size of shortest edge of the frame, if not already this size then it will
                be resized.
            extension:
                Extension of dumped frames.
        """
        self.video_segment_dir = video_segment_dir
        self.frame_size = int(frame_size)
        self.meta_data = self._df_to_list_of_dicts(annotations_df)
        self.extensions = {"jpg", "jpeg", extension}

    def iter_data(self, slice_element=None) -> Iterator[Result]:
        """Get frames and metadata corresponding to segment

        Args:
            slice_element (optional): If not specified all frames for the segment will be returned

        Yields:
            dict: dictionary with the fields

            * ``meta``: All metadata corresponding to the segment, this is the same as the data
              in the labels csv
            * ``frames``: list of :class:`PIL.Image.Image` corresponding to the frames specified
              in ``slice_element``
            * ``id``: UID corresponding to segment
        """
        slice_element = slice_element or slice(0, len(self))
        for meta in self.meta_data[slice_element]:
            folder = (
                Path(self.video_segment_dir) / meta["participant_id"] / meta["video_id"]
            )
            paths = [
                folder / f"frame_{idx:010d}.jpg"
                for idx in range(meta["start_frame"], meta["stop_frame"] + 1)
            ]
            frames = list(resize_images(map(str, paths), self.frame_size))
            meta["frame_size"] = frames[0].shape
            meta["num_frames"] = len(frames)

            result = {"meta": meta, "frames": frames, "id": (self.get_uid(meta))}
            yield result

    def get_uid(self, meta):
        if "narration_id" in meta:
            id_ = meta["narration_id"]
        elif "uid" in meta:
            id_ = meta["uid"]
        else:
            raise ValueError(f"meta must have a narration_id or uid key, {meta}")
        return id_

    def __len__(self):
        return len(self.meta_data)

    def _df_to_list_of_dicts(self, annotations: pd.DataFrame) -> List[Dict[str, Any]]:
        data = []
        i: int
        row: pd.Series
        for i, row in annotations.reset_index().iterrows():
            metadata: Dict[str, Any] = row.to_dict()
            data.append(metadata)
        return data


class EpicFlowDatasetAdapter(EpicDatasetAdapter):
    """Gulp Dataset Adapter for Gulping flow frames extracted from the EPIC-KITCHENS dataset"""

    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))
        for meta in self.meta_data[slice_element]:
            video_id = meta["video_id"]

            folder = Path(self.video_segment_dir) / meta["participant_id"] / video_id
            start_frame = meta["start_frame"]
            stop_frame = meta["stop_frame"]
            paths = {
                axis: [
                    folder / axis / f"frame_{idx:010d}.jpg"
                    for idx in range(start_frame, stop_frame)
                ]
                for axis in ["u", "v"]
            }

            frames = {}
            for axis in "u", "v":
                frames[axis] = list(
                    resize_images(map(str, paths[axis]), self.frame_size)
                )

            meta["frame_size"] = frames["u"][0].shape
            meta["num_frames"] = len(frames["u"])
            result = {
                "meta": meta,
                "frames": list(_intersperse(frames["u"], frames["v"])),
                "id": self.get_uid(meta),
            }
            yield result


def _intersperse(*lists):
    """
    Args:
        *lists:

    Examples:
        >>> list(_intersperse(['a', 'b']))
        ['a', 'b']
        >>> list(_intersperse(['a', 'c'], ['b', 'd']))
        ['a', 'b', 'c', 'd']
        >>> list(_intersperse(['a', 'd'], ['b', 'e'], ['c', 'f']))
        ['a', 'b', 'c', 'd', 'e', 'f']
        >>> list(_intersperse(['a', 'd', 'g'], ['b', 'e'], ['c', 'f']))
        ['a', 'b', 'c', 'd', 'e', 'f']

    """
    i = 0
    min_length = min(map(len, lists))
    total_element_count = len(lists) * min_length
    for i in range(0, total_element_count):
        list_index = i % len(lists)
        element_index = i // len(lists)
        yield lists[list_index][element_index]


class MissingDataException(Exception):
    pass
