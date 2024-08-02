# Copyright 2024 Cisco Systems, Inc. and its affiliates
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
#
# SPDX-License-Identifier: Apache-2.0

"""Decorator to measure function execution time."""
import time
from functools import wraps


def timeit(func):
    """Measure function execution time as decorator.

    Usage example:
    @timeit
    def do_something(num):
        total = sum((x for x in range(0, num)))
        return total

    To access the execution time outside the function is achieved as follows.
    e.g.: print(do_simething.elapsed)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        wrapper.elapsed = end_time - start_time

        print(f"elapsed time = {wrapper.elapsed}")

        return result

    wrapper.elaspsed = 0

    return wrapper
