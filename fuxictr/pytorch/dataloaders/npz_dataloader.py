# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class NpzDataset(Dataset):
    """PyTorch Dataset that loads a single NPZ file into memory.

    Args:
        feature_map (FeatureMap): Feature map that defines columns and labels.
        data_path (str): Path to the NPZ file.
    """

    def __init__(self, feature_map, data_path):
        self.feature_map = feature_map
        self.darray = self.load_data(data_path)

    def __getitem__(self, index):
        """Get a single row by index.

        Args:
            index (int): Row index.

        Returns:
            np.ndarray: The row at the given index.
        """
        return self.darray[index, :]

    def __len__(self):
        """Return the number of rows in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.darray.shape[0]

    def load_data(self, data_path):
        """Load data from an NPZ file and stack columns.

        Args:
            data_path (str): Path to the NPZ file.

        Returns:
            np.ndarray: Stacked array of shape (num_samples, num_columns).
        """
        data_dict = np.load(data_path) # dict of arrays
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        data_arrays = [data_dict[col] for col in all_cols]
        return np.column_stack(data_arrays)


class NpzDataLoader(DataLoader):
    """DataLoader for a single NPZ dataset.

    Args:
        feature_map (FeatureMap): Feature map that defines columns and labels.
        data_path (str): Path to the NPZ file.
        batch_size (int, optional): Number of samples per batch. Default: ``32``.
        shuffle (bool, optional): Whether to shuffle the data. Default: ``False``.
        num_workers (int, optional): Number of worker processes. Default: ``1``.
        **kwargs: Additional arguments passed to ``DataLoader``.
    """

    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False, num_workers=1, **kwargs):
        if not data_path.endswith(".npz"):
            data_path += ".npz"
        self.dataset = NpzDataset(feature_map, data_path)
        super(NpzDataLoader, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers,
                                            collate_fn=BatchCollator(feature_map))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples * 1.0 / self.batch_size))

    def __len__(self):
        """Return the number of batches per epoch.

        Returns:
            int: Number of batches.
        """
        return self.num_batches


class BatchCollator(object):
    """Collate a batch of rows into a dictionary of column tensors.

    Args:
        feature_map (FeatureMap): Feature map that defines columns and labels.
    """

    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, batch):
        """Collate a list of rows into a batched dictionary.

        Args:
            batch (list[np.ndarray]): List of rows.

        Returns:
            dict[str, torch.Tensor]: Mapping from column name to batched tensor.
        """
        batch_tensor = default_collate(batch)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        batch_dict = dict()
        for col in all_cols:
            batch_dict[col] = batch_tensor[:, self.feature_map.get_column_index(col)]
        return batch_dict
