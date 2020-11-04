import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(
    description="Convert the frame ids from RGB frames to flow frame ids.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("labels_in_pkl", type=Path)
parser.add_argument("labels_out_pkl", type=Path)


# For videos from 2018 we extracted flow by a temporal stride of 2, so there are
# half the number of flow frames as RGB
# For 2020 we didn't subsample frames, so there is almost a 1-to-1 correspondence
# (bar the last frame, given optical flow is computed between pairs of frames).
STRIDES = {
    2018: 2,
    2020: 1,
}


def convert_rgb_frame_to_flow_frame_idx(frame_idxs, stride: int):
    return (frame_idxs - 1) // stride + 1


def main(args):
    df = pd.read_pickle(args.labels_in_pkl)
    # Videos from 2018 are of the format PXX_YY and from 2020 are PXX_1YY
    df["year"] = df["video_id"].apply(
        lambda id_: 2020 if len(id_.split("_")[-1]) == 3 else 2018
    )
    columns_to_convert = [c for c in df.columns if c.endswith("frame")]
    year_masks = {year: df["year"] == year for year in [2020, 2018]}

    for year, year_mask in year_masks.items():
        stride = STRIDES[year]
        for col in columns_to_convert:
            df.loc[year_mask, col] = convert_rgb_frame_to_flow_frame_idx(
                df.loc[year_mask, col], stride
            )

    df.to_pickle(args.labels_out_pkl)


if __name__ == "__main__":
    main(parser.parse_args())
