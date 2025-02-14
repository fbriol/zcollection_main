# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Test of views
=============
"""
import itertools

import numpy
import pytest

from .. import convenience, meta, view
# pylint: disable=unused-import # Need to import for fixtures
from .cluster import dask_client, dask_cluster
from .data import create_test_collection
from .fs import local_fs, s3, s3_base, s3_fs

# pylint: enable=unused-import


@pytest.mark.parametrize(["arg", "delayed"],
                         list(
                             itertools.product(["local_fs", "s3_fs"],
                                               [True, False])))
def test_view(
    dask_client,  # pylint: disable=redefined-outer-name,unused-argument
    arg,
    delayed,
    request,
):
    """Test the creation of a view."""
    tested_fs = request.getfixturevalue(arg)

    create_test_collection(tested_fs, delayed=False)
    instance = convenience.create_view(str(tested_fs.view),
                                       view.ViewReference(
                                           str(tested_fs.collection),
                                           tested_fs.fs),
                                       filesystem=tested_fs.fs)
    assert isinstance(instance, view.View)
    assert isinstance(str(instance), str)

    # No variable recorded, so no data can be loaded
    ds = instance.load(delayed=delayed)
    assert ds is None

    var = meta.Variable(
        name="var2",
        dtype=numpy.float64,
        dimensions=("num_lines", "num_pixels"),
        attrs=(meta.Attribute(name="attr", value=1), ),
    )

    with pytest.raises(ValueError):
        instance.add_variable(var)

    var.name = "var3"
    instance.add_variable(var)

    with pytest.raises(ValueError):
        instance.add_variable(var)

    instance = convenience.open_view(str(tested_fs.view),
                                     filesystem=tested_fs.fs)
    ds = instance.load(delayed=delayed)
    assert ds is not None
    assert set(ds["time"].values.astype("datetime64[D]")) == {
        numpy.datetime64("2000-01-01"),
        numpy.datetime64("2000-01-04"),
        numpy.datetime64("2000-01-07"),
        numpy.datetime64("2000-01-10"),
        numpy.datetime64("2000-01-13"),
        numpy.datetime64("2000-01-16"),
    }

    # Loading a variable existing only in the view.
    ds = instance.load(selected_variables=("var3", ), delayed=delayed)
    assert ds is not None
    assert tuple(ds.variables) == ("var3", )

    # Loading a non existing variable.
    ds = instance.load(selected_variables=("var55", ), delayed=delayed)
    assert ds is not None
    assert len(ds.variables) == 0

    # Loading data from an indexer.
    indexers = instance.map(
        lambda x: slice(0, x.dimensions["num_lines"]),  # type: ignore
        delayed=delayed).compute()
    try:
        ds = instance.load(indexer=indexers, delayed=delayed)
        assert ds is not None
        assert len(ds.variables) == 4
        assert set(ds["time"].values.astype("datetime64[D]")) == {
            numpy.datetime64("2000-01-01"),
            numpy.datetime64("2000-01-04"),
            numpy.datetime64("2000-01-07"),
            numpy.datetime64("2000-01-10"),
            numpy.datetime64("2000-01-13"),
            numpy.datetime64("2000-01-16"),
        }
    except ValueError:
        if delayed != False:
            pytest.fail("Unexpected error")

    ds = instance.load_indexed(dict(
        (ix, (item, )) for ix, item in enumerate(indexers)),
                               delayed=delayed)
    for ix, _ in enumerate(indexers):
        assert numpy.all(ds[ix].variables["var1"].values == ix)

    # Test view loading that is no longer synchronized with the reference
    # collection.
    tested_fs.fs.rm(str(
        tested_fs.view.joinpath("year=2000", "month=01", "day=13")),
                    recursive=True)

    assert len(tuple(instance.partitions())) == 5
    assert len(tuple(instance.view_ref.partitions())) == 6

    ds = instance.load(delayed=delayed)
    assert ds is not None
    assert set(ds["time"].values.astype("datetime64[D]")) == {
        numpy.datetime64("2000-01-01"),
        numpy.datetime64("2000-01-04"),
        numpy.datetime64("2000-01-07"),
        numpy.datetime64("2000-01-10"),
        numpy.datetime64("2000-01-16"),
    }

    # Create a variable with the unsynchronized view
    var.name = "var4"
    instance.add_variable(var)

    ds = instance.load(delayed=delayed)
    assert ds is not None

    def update(ds):
        """Update function used for this test."""
        return ds.variables["var1"].values * 0 + 5

    instance.update(update, "var3", delayed=delayed)  # type: ignore

    with pytest.raises(ValueError):
        instance.update(update, "varX")  # type: ignore

    with pytest.raises(ValueError):
        instance.update(update, "var2")  # type: ignore

    ds = instance.load(delayed=delayed)
    assert ds is not None
    numpy.all(ds.variables["var3"].values == 5)

    indexers = instance.map(
        lambda x: slice(0, x.dimensions["num_lines"]),  # type: ignore
        delayed=delayed).compute()
    try:
        ds1 = instance.load(indexer=indexers, delayed=delayed)
        assert ds1 is not None
        ds2 = instance.load(delayed=delayed)
        assert ds2 is not None

        assert numpy.allclose(ds1.variables["var1"].values,
                              ds2.variables["var1"].values)
    except ValueError:
        if delayed != False:
            pytest.fail("Unexpected error")

        ds2 = instance.load(delayed=delayed)
        assert ds2 is not None

    instance.drop_variable("var3")

    with pytest.raises(ValueError):
        convenience.open_view(str(tested_fs.collection),
                              filesystem=tested_fs.fs)
