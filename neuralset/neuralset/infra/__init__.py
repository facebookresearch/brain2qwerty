# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import

# set the following for legacy uid computation:
# os.environ.setdefault("CONFDICT_UID_VERSION", "1")

import exca

if not hasattr(exca, "__version__"):
    raise RuntimeError("exca must be updated to version 0.2.0 or newer.")

from exca import ConfDict as ConfDict
from exca import MapInfra as MapInfra
from exca import TaskInfra as TaskInfra
from exca import helpers as helpers
from exca.base import DEFAULT_CHECK_SKIPS
from exca.cachedict import CacheDict as CacheDict


# pylint: disable=unused-argument
def _skip_new_event_types(key, val, prev):
    if "event_types" in key and not prev:
        return True
    return False


DEFAULT_CHECK_SKIPS.append(_skip_new_event_types)
