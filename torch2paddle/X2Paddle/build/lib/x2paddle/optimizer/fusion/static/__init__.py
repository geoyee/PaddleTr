#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from .bn_scale_fuser import Static_BNScaleFuser
from .bn_scale_fuse_pass import Static_BNScaleFusePass
from .conv2d_add_fuser import StaticConv2DAddFuser
from .conv2d_add_fuse_pass import StaticConv2DAddFusePass
from .prelu_fuser import StaticPReLUFuser
from .prelu_fuse_pass import StaticPReLUFusePass
from .tf_batchnorm_fuser import StaticTFBatchNormFuser
from .tf_batchnorm_fuse_pass import StaticTFBatchNormFusePass

