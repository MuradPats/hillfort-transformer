import os
from pickletools import uint8
import cv2

try:
    import rasterio
except Exception:
    rasterio = None
import torch
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting["rgb_root"]
        self._rgb_format = setting["rgb_format"]
        self._gt_path = setting["gt_root"]
        self._gt_format = setting["gt_format"]
        self._transform_gt = setting["transform_gt"]
        self._x_path = setting["x_root"]
        self._x_format = setting["x_format"]
        self._x_single_channel = setting["x_single_channel"]
        self._train_source = setting["train_source"]
        self._eval_source = setting["eval_source"]
        self.class_names = setting["class_names"]
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]
        rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
        x_path = os.path.join(self._x_path, item_name + self._x_format)
        gt_path = os.path.join(self._gt_path, item_name + self._gt_format)

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt)

        if self._x_single_channel:
            # Try reading with OpenCV first (preserve original dtype using IMREAD_UNCHANGED)
            x_arr = None
            try:
                x_cv = cv2.imread(x_path, cv2.IMREAD_UNCHANGED)
                if x_cv is not None:
                    x_arr = np.array(x_cv)
            except Exception:
                x_arr = None

            # If cv2 couldn't read (e.g., 32-bit TIFF), try rasterio (if available)
            if (
                x_arr is None
                and rasterio is not None
                and str(x_path).lower().endswith((".tif", ".tiff"))
            ):
                try:
                    with rasterio.open(x_path) as src:
                        band = src.read(1)
                    x_arr = np.array(band)
                except Exception:
                    x_arr = None

            if x_arr is None:
                raise RuntimeError(f"Failed to load single-channel input: {x_path}")

            # Normalize shape to HxWx3 numeric array
            if x_arr.ndim == 2:
                x = np.stack([x_arr, x_arr, x_arr], axis=2)
            elif x_arr.ndim == 3:
                # handle channel-last HxWxC
                if x_arr.shape[2] == 1:
                    x = np.concatenate([x_arr, x_arr, x_arr], axis=2)
                elif x_arr.shape[2] == 3:
                    x = x_arr
                else:
                    # reduce to first channel and replicate
                    first = x_arr[:, :, 0]
                    x = np.stack([first, first, first], axis=2)
            else:
                raise RuntimeError(f"Unsupported DTM array shape: {x_arr.shape}")

            x = x.astype(np.float32)
        else:
            x = self._open_image(x_path, cv2.COLOR_BGR2RGB)

        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        if self._split_name == "train":
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(
            data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names)
        )

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ["train", "val"]
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[: length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    def stem_to_index(self, stem: str) -> int | None:
        """Return dataset index for a given file stem or None if not found."""
        try:
            return self._file_names.index(stem)
        except ValueError:
            return None

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
