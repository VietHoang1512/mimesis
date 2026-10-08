# Copyright 2024 PRIME team and/or its affiliates
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

from .registry import get_reward_manager_cls, register  # noqa: I001
from .naive import NaiveRewardManager

# Only the naive manager ships here: it is the one the released configs select
# (`reward_manager: naive`) and it carries the `interact_*` scoring branch.
# Upstream's batch/dapo/prime managers were dropped along with the reward_score
# modules they depend on -- see README.md for the full list of pruned backends.
__all__ = ["NaiveRewardManager", "register", "get_reward_manager_cls"]
