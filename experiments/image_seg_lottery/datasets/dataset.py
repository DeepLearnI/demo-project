# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from bedrock import dataset_base
import numpy as np
from satellite_segmentation.constants import BATCH_SIZE, ISZ, N_Cls, targets_file_name, inputs_file_name
from tqdm import tqdm


def get_patches(img, msk, amt=10000, aug=True):
  is2 = int(1.0 * ISZ)
  xm, ym = img.shape[0] - is2, img.shape[1] - is2
  
  x, y = [], []
  
  tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
  for i in tqdm(range(amt)):
    xc = np.random.randint(0, xm)
    yc = np.random.randint(0, ym)
    
    im = img[xc:xc + is2, yc:yc + is2]
    ms = msk[xc:xc + is2, yc:yc + is2]
    
    for j in range(N_Cls):
      sm = np.sum(ms[:, :, j])
      if 1.0 * sm / is2 ** 2 > tr[j]:
        if aug:
          if np.random.uniform(0, 1) > 0.5:
            im = im[::-1]
            ms = ms[::-1]
          if np.random.uniform(0, 1) > 0.5:
            im = im[:, ::-1]
            ms = ms[:, ::-1]
        
        x.append(im)
        y.append(ms)
  
  x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
  print(x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
  return x, y

class Dataset(dataset_base.DatasetBase):
  def __init__(self,
               train_order_seed=None):
    
    print('Downloading dataset..')
    input_path, target_path = self.download_data()
    
    print('Preprocessing data..')
    inputs = np.load(input_path)
    targets = np.load(target_path)
    X, y = get_patches(inputs, targets, amt=10000)
    
    print('Splitting data..')
    split_idx = int(X.shape[0]*0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('y_test shape: {}'.format(y_test.shape))

    # Prepare the dataset.
    super(Dataset, self).__init__(
      (X_train, y_train),
      BATCH_SIZE, (X_test, y_test),
      train_order_seed=train_order_seed)
    
  def download_data(self):
    import urllib.request
  
    input_path = '/tmp/inputs.npy'
    target_path = '/tmp/targets.npy'
  
    urllib.request.urlretrieve("http://dl-shareable.s3.amazonaws.com/{}".format(inputs_file_name), input_path)
    urllib.request.urlretrieve("http://dl-shareable.s3.amazonaws.com/{}".format(targets_file_name), target_path)
    return input_path, target_path
