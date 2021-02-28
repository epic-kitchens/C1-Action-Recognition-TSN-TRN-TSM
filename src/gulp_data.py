"""Program for building GulpIO directory of frames for training
See :ref:`cli_tools_gulp_ingestor` for usage details """
import argparse
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
from gulpio2 import GulpIngestor
from utils.gulp_adapter import EpicDatasetAdapter
from utils.gulp_adapter import EpicFlowDatasetAdapter


parser = argparse.ArgumentParser(
    "Gulp the EPIC dataset allowing for faster read times during training."
)
parser.add_argument(
    "in_folder",
    type=Path,
    help="Directory where subdirectory is a segment name containing frames for that segment.",
)
parser.add_argument(
    "out_folder", type=Path, help="Directory to store the gulped files."
)
parser.add_argument(
    "labels",
    type=Path,
    help="Path to the pickle or CSV file which contains the meta information about the dataset.",
)

parser.add_argument("modality", choices=["flow", "rgb"])
parser.add_argument(
    "--extension",
    type=str,
    default="jpg",
    help="Which file extension the frames are saved as.",
)
parser.add_argument("--frame-size", type=int, default=-1, help="Size of frames.")
parser.add_argument(
    "--segments-per-chunk",
    type=int,
    default=100,
    help="Number of action segments per chunk to save.",
)
parser.add_argument(
    "-j",
    "--num-workers",
    type=int,
    default=cpu_count(),
    help="Number of workers to run the task.",
)


def main(args):
    if args.labels.suffix.lower() == ".pkl":
        labels = pd.read_pickle(args.labels)
    elif args.labels.suffix.lower() == ".csv":
        labels = pd.read_csv(args.labels, index_col="uid")
    else:
        raise ValueError("Expected .csv or .pkl suffix for annotation file")
    if args.modality.lower() == "flow":
        epic_adapter = EpicFlowDatasetAdapter(
            str(args.in_folder), labels, args.frame_size, args.extension,
        )
    elif args.modality.lower() == "rgb":
        epic_adapter = EpicDatasetAdapter(
            str(args.in_folder), labels, args.frame_size, args.extension,
        )
    else:
        raise ValueError("Modality '{}' not supported".format(args.modality))
    ingestor = GulpIngestor(
        epic_adapter, str(args.out_folder), args.segments_per_chunk, args.num_workers
    )
    ingestor()


if __name__ == "__main__":
    main(parser.parse_args())
