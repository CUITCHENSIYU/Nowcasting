import h5py
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime


def timestamp_to_filename(ts) -> str:
    """Convert timestamp like '01-JAN-2016;00:00:00.000' to a filesystem-safe name."""
    if isinstance(ts, np.ndarray):
        ts = ts.item() if ts.ndim == 0 else ts[0]
    if isinstance(ts, bytes):
        ts = ts.decode("utf-8")
    ts = str(ts)
    # Prefer compact sortable names: 20160101_000000
    try:
        dt = datetime.strptime(ts, "%d-%b-%Y;%H:%M:%S.%f")
        return dt.strftime("%Y%m%d_%H%M%S")
    except ValueError:
        return ts.replace(";", "_").replace(":", "-").replace(" ", "_")


def save_sequence_h5(path: str, images: np.ndarray, timestamps):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "images",
            data=np.asarray(images, dtype=np.float32),
            compression="gzip",
            compression_opts=9,
        )
        f.create_dataset(
            "timestamps",
            data=np.asarray(timestamps),
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",
            compression_opts=9,
        )


def create_dataset(root: str, input_length: int, image_ahead: int, rain_amount_thresh: float):
    """Create per-sequence h5 files whose target frame has at least `rain_amount_thresh` rain pixels.

    Split follows the source file: train (2016-2018) and test (2019).
    Each sequence is written as one h5 under ``{out_dir}/{split}/{first_timestamp}.h5``.
    """

    precipitation_file = os.path.join(root, "RAD_NL25_RAC_5min_train_test_2016-2019.h5")
    out_dir = os.path.join(
        root,
        f"seq_{input_length}_out-seq_{image_ahead}_threshold_{int(rain_amount_thresh * 100)}",
    )

    with h5py.File(precipitation_file, "r") as orig_f:
        # Original split rule: only train / test groups in the source h5.
        splits = {
            "train": (orig_f["train"]["images"], orig_f["train"]["timestamps"]),
            "test": (orig_f["test"]["images"], orig_f["test"]["timestamps"]),
        }
        for split_name, (images, _) in splits.items():
            print(f"{split_name.capitalize()} shape", images.shape)

        img_size = next(iter(splits.values()))[0].shape[1]
        num_pixels = img_size * img_size
        sequence_length = input_length + image_ahead

        for split_name, (images, timestamps) in splits.items():
            split_dir = os.path.join(out_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            valid_indices = []
            for i in tqdm(
                range(sequence_length, len(images)),
                desc=f"Finding valid indices [{split_name}]",
            ):
                rain_pixels = np.sum(images[i] > 0)
                if rain_pixels >= num_pixels * rain_amount_thresh:
                    valid_indices.append(i)

            print(f"[{split_name}] {len(valid_indices)} sequences")

            for i in tqdm(valid_indices, desc=f"Writing sequences [{split_name}]"):
                imgs = images[i - sequence_length : i]
                timestamps_img = timestamps[i - sequence_length : i]
                first_ts = timestamps_img[0]
                filename = f"{timestamp_to_filename(first_ts)}.h5"
                save_sequence_h5(os.path.join(split_dir, filename), imgs, timestamps_img)


if __name__ == "__main__":

    root = "/mnt/disk6/csy/netherlands"
    print("Creating dataset with at least 20% of rain pixel in target image")
    create_dataset(root, input_length=12, image_ahead=6, rain_amount_thresh=0.2)
    # print("Creating dataset with at least 50% of rain pixel in target image")
    # create_dataset(root, input_length=12, image_ahead=6, rain_amount_thresh=0.5)
