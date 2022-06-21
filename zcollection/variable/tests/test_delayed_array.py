# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the Dask array.
=======================
"""
import pickle

import dask.array.core
import numpy
import pytest
import zarr

from ... import meta
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster
from ..delayed_array import DelayedArray

# pylint enable=unused-import


def create_test_variable(name="var1", fill_value=0):
    """Create a test variable."""
    return DelayedArray(name=name,
                        data=numpy.arange(10, dtype="int64").reshape(5, 2),
                        dimensions=("x", "y"),
                        attrs=(meta.Attribute(name="attr", value=1), ),
                        compressor=zarr.Blosc(cname="zstd", clevel=1),
                        fill_value=fill_value,
                        filters=(zarr.Delta("int64", "int32"),
                                 zarr.Delta("int32", "int32")))


def test_delayed_array(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test variable creation."""
    var = create_test_variable()
    assert var.name == "var1"
    assert var.dtype == numpy.dtype("int64")
    assert var.shape == (5, 2)
    assert var.dimensions == ("x", "y")
    assert var.attrs == (meta.Attribute(name="attr", value=1), )
    assert var.compressor.cname == "zstd"  # type: ignore
    assert var.compressor.clevel == 1  # type: ignore
    assert var.fill_value == 0
    assert var.size == 10
    assert var.nbytes == 80
    assert var.filters == (
        zarr.Delta("int64", "int32"),
        zarr.Delta("int32", "int32"),
    )
    assert numpy.all(var.values == numpy.arange(10).reshape(5, 2))
    assert numpy.all(var.values == var.values)
    assert tuple(var.dimension_index()) == (("x", 0), ("y", 1))
    assert isinstance(var.metadata(), meta.Variable)
    assert isinstance(str(var), str)
    assert isinstance(repr(var), str)

    def foo(a, b):
        return a + b

    assert numpy.all(
        foo(var, var.values) == numpy.arange(10).reshape(5, 2) +
        numpy.arange(10).reshape(5, 2))

    assert numpy.all(
        foo(var, var.data).compute() == numpy.arange(10).reshape(5, 2) +
        numpy.arange(10).reshape(5, 2))

    var.data = numpy.ones((10, 4), dtype="int64")
    assert var.data.shape == (10, 4)
    assert isinstance(var.data, dask.array.core.Array)
    assert numpy.all(var.values == 1)

    with pytest.raises(ValueError):
        var.data = numpy.ones((10, 4, 2), dtype="int64")


def test_delayed_array_concat(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test concatenation of variables."""
    var_a = create_test_variable()
    var_b = create_test_variable()
    var_c = create_test_variable()

    vard = var_a.concat((var_b, var_c), "x")
    assert numpy.ma.all(vard.values == numpy.concatenate(
        (var_a.values, var_b.values, var_c.values), axis=0))

    vard = var_a.concat(var_b, "x")
    assert numpy.all(
        vard.values == numpy.concatenate((var_a.values, var_b.values), axis=0))

    with pytest.raises(ValueError):
        var_a.concat([], "y")


def test_delayed_array_datetime64_to_xarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test conversion to xarray."""
    dates = numpy.arange(
        numpy.datetime64("2000-01-01", "ms"),
        numpy.datetime64("2000-02-01", "ms"),
        numpy.timedelta64("1", "h"),
    )
    var = DelayedArray(
        name="time",
        data=dates,
        dimensions=("num_lines", ),
        attrs=(meta.Attribute(name="attr", value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta("int64", "int64"), ),
    )
    xr_var = var.to_xarray()
    assert xr_var.dims == ("num_lines", )
    assert xr_var.attrs == dict(attr=1)
    assert xr_var.dtype == "datetime64[ns]"


def test_delayed_array_timedelta64_to_xarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test conversion to xarray."""
    delta = numpy.diff(
        numpy.arange(
            numpy.datetime64("2000-01-01", "ms"),
            numpy.datetime64("2000-02-01", "ms"),
            numpy.timedelta64("1", "h"),
        ))

    var = DelayedArray(
        name="timedelta",
        data=delta,
        dimensions=("num_lines", ),
        attrs=(meta.Attribute(name="attr", value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta("int64", "int64"), ),
    )
    xr_var = var.to_xarray()
    assert xr_var.dims == ("num_lines", )
    assert xr_var.attrs == dict(attr=1)
    assert xr_var.dtype.kind == "m"


def test_delayed_array_dimension_less(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Concatenate two dimensionless variables."""
    data = numpy.array([0, 1], dtype=numpy.int32)
    args = ("nv", data, ("nv", ))
    kwargs = dict(attrs=(meta.Attribute("comment", "vertex"),
                         meta.Attribute("units", "1")))
    n_vertex = DelayedArray(*args, **kwargs)
    assert n_vertex.fill_value is None
    metadata = n_vertex.metadata()
    assert metadata.fill_value is None
    assert meta.Variable.from_config(metadata.get_config()) == metadata

    other = DelayedArray(*args, **kwargs)

    concatenated = n_vertex.concat((other, ), "time")
    assert numpy.all(concatenated.values == n_vertex.values)
    assert concatenated.metadata() == n_vertex.metadata()


def test_delayed_array_getitem(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    var = create_test_variable()
    values = var.values
    result = var[0].compute()
    assert numpy.all(result == values[0])
    result = var[0:2].compute()
    assert numpy.all(result == values[0:2])
    result = var[0:2, 0].compute()
    assert numpy.all(result == values[0:2, 0])
    result = var[0:2, 0:2].compute()
    assert numpy.all(result == values[0:2, 0:2])


def test_delayed_array_fill(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test filling of variables."""
    var = create_test_variable()
    assert not var.values.all() is numpy.ma.masked
    var.fill()
    assert var.values.all() is numpy.ma.masked


def test_delayed_array_masked_array(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test masked array."""
    var = create_test_variable()
    var2 = var.rename("var2")
    assert var2.array is var.array
    assert var2.name == "var2"
    assert var2.dimensions == var.dimensions
    assert var2.attrs == var.attrs
    assert var2.compressor == var.compressor
    assert var2.filters == var.filters
    assert var2.fill_value == var.fill_value
    assert var2.dtype == var.dtype
    assert var2.shape == var.shape
    assert var2.size == var.size
    assert var2.ndim == var.ndim


def test_delayed_array_pickle(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test variable creation."""
    var = create_test_variable()
    var2 = pickle.loads(pickle.dumps(var))
    assert var2.name == var.name
    assert var2.dimensions == var.dimensions
    assert var2.attrs == var.attrs
    assert var2.compressor == var.compressor
    assert var2.filters == var.filters
    assert var2.fill_value == var.fill_value
    assert var2.dtype == var.dtype
    assert var2.shape == var.shape
    assert var2.size == var.size
    assert var2.ndim == var.ndim
    assert numpy.all(var2.values == var.values)


def test_array_from_zarr():
    """Test creation of variable from zarr array."""
    array = zarr.array(numpy.full((20, 10), 1, dtype="int64"))
    array.attrs["_ARRAY_DIMENSIONS"] = ("x", "y")
    var = DelayedArray.from_zarr(array,
                                 name="var1",
                                 dimension="_ARRAY_DIMENSIONS")
    assert var.shape == (20, 10)
    assert var.dtype == "int64"
    assert var.dimensions == ("x", "y")
    assert var.attrs == tuple()
    assert var.compressor == array.compressor
    assert var.fill_value == array.fill_value
    assert var.size == array.size
    assert var.nbytes == array.nbytes
    assert var.filters == array.filters
    assert numpy.all(var.values == array[...])

    array = zarr.full((20, 10), 1, dtype="int64")
    array.attrs["_ARRAY_DIMENSIONS"] = ("x", "y")
    var = DelayedArray.from_zarr(array,
                                 name="var1",
                                 dimension="_ARRAY_DIMENSIONS")
    assert var.shape == (20, 10)
    assert var.dtype == "int64"
    assert var.dimensions == ("x", "y")
    assert var.attrs == tuple()
    assert var.compressor == array.compressor
    assert var.fill_value == array.fill_value
    assert var.size == array.size
    assert var.nbytes == array.nbytes
    assert var.filters == array.filters
    assert var.values.all() is numpy.ma.masked
