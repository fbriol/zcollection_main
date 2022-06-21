# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the ABC interface.
==========================
"""
from .. import representation


def test_maybe_truncate():
    """Test the truncation of a string to a given length."""
    data = list(range(1000))
    # pylint: disable=protected-access
    assert representation.maybe_truncate(data, 10) == "[0, 1, ..."
    assert representation.maybe_truncate(data, len(str(data))) == str(data)
    # pylint: enable=protected-access
