# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, darray):
        self.darray = darray
        
    def __getitem__(self, index):
        X = self.darray[index, 0:-1]
        y = self.darray[index, -1]
        return X, y
    
    def __len__(self):
        return self.darray.shape[0]


class DataGenerator(data.DataLoader):
    def __init__(self, data_array, batch_size=32, shuffle=False, num_workers=1, **kwargs):
        self.dataset = Dataset(data_array)
        super(DataGenerator, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
    def __len__(self):
        return int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))

