# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Testing the array class.
========================
"""
import pickle

import dask.array.core
import numpy
import pytest
import zarr

from .. import array
from ... import meta
# pylint: disable=unused-import # Need to import for fixtures
from ...tests.cluster import dask_client, dask_cluster


# pylint enable=unused-import
def create_concatenated_variable(fill_value=None):
    """Create a concatenated variable."""
    arr1 = zarr.array(numpy.full((20, 10), 1, dtype="int64"))
    arr2 = zarr.array(numpy.full((10, 10), 2, dtype="int64"))
    arr3 = zarr.array(numpy.full((5, 10), 3, dtype="int64"))
    arr4 = zarr.array(numpy.full((2, 10), 4, dtype="int64"))

    var1 = array.Array("var1", arr1, ("x", "y"), fill_value=fill_value)
    var2 = array.Array("var2", arr2, ("x", "y"), fill_value=fill_value)
    var3 = array.Array("var3", arr3, ("x", "y"), fill_value=fill_value)
    var4 = array.Array("var4", arr4, ("x", "y"), fill_value=fill_value)

    return var1.concat((var2, var3, var4), dim="x"), (arr1, arr2, arr3, arr4)


def test_array_interface(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test variable creation."""
    arr = zarr.ones((100, 100), chunks=(10, 10), dtype='int64')
    var = array.Array("var1",
                      arr, ("x", "y"),
                      attrs=(array.Attribute(name="attr", value=1), ),
                      compressor=zarr.Blosc(cname="zstd", clevel=1),
                      fill_value=None,
                      filters=(
                          zarr.Delta("int64", "int32"),
                          zarr.Delta("int32", "int32"),
                      ))
    assert var.name == "var1"
    assert var.dtype == numpy.dtype("int64")
    assert var.shape == arr.shape
    assert var.dimensions == ("x", "y")
    assert var.attrs == (array.Attribute(name="attr", value=1), )
    assert var.compressor.cname == "zstd"  # type: ignore
    assert var.compressor.clevel == 1  # type: ignore
    assert var.fill_value is None
    assert var.size == arr.size
    assert var.nbytes == arr.size * arr.dtype.itemsize
    assert var.filters == (
        zarr.Delta("int64", "int32"),
        zarr.Delta("int32", "int32"),
    )
    assert numpy.all(var.values == numpy.ones(arr.shape, dtype=arr.dtype))
    assert numpy.all(var.values == var.values)
    assert tuple(var.dimension_index()) == (("x", 0), ("y", 1))
    assert isinstance(var.metadata(), meta.Variable)
    assert isinstance(str(var), str)
    assert isinstance(repr(var), str)

    var.data = numpy.full(arr.shape, 3, dtype=arr.dtype)
    assert var.data.shape == arr.shape
    assert isinstance(var.data, dask.array.core.Array)
    assert numpy.all(var.values == 3)

    with pytest.raises(ValueError):
        var.data = numpy.ones((10, 4, 2), dtype="int64")


def test_array_concatenate(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test variable concatenation."""
    var, (arr1, arr2, arr3, arr4) = create_concatenated_variable()

    cat = numpy.concatenate(
        (arr1[:], arr2[:], arr3[:], arr4[:]),
        axis=0,
    )

    assert var.name == "var1"
    assert var.dtype == numpy.dtype("int64")
    assert var.shape == (20 + 10 + 5 + 2, 10)
    assert var.dimensions == ("x", "y")
    assert var.attrs == ()
    assert var.compressor == arr1.compressor
    assert var.fill_value is None
    assert var.size == cat.size
    assert var.nbytes == cat.size * arr1.dtype.itemsize
    assert var.filters is None
    assert numpy.all(var.values == cat)
    assert numpy.all(var.values == var.data.compute())
    assert tuple(var.dimension_index()) == (("x", 0), ("y", 1))
    assert isinstance(var.metadata(), meta.Variable)
    assert isinstance(str(var), str)
    assert isinstance(repr(var), str)

    with pytest.raises(ValueError, match="dtype"):
        var.concat(array.Array("var2", arr1.astype("float64"), ("x", "y")),
                   dim="x")

    with pytest.raises(ValueError, match="shape"):
        var.concat(
            array.Array("var2",
                        zarr.array(numpy.full((20, 5), 1, dtype="int64")),
                        ("x", "y")),
            dim="x",
        )

    with pytest.raises(ValueError, match="shape"):
        var.concat(array.Array("var2", arr1, ("x", "y")), dim="y")

    with pytest.raises(ValueError, match="Fill values"):
        var.concat(array.Array("var2", arr1, ("x", "y"), fill_value=32767),
                   dim="x")


def test_array_isel(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test array isel."""
    var, (arr1, arr2, arr3, arr4) = create_concatenated_variable()

    cat = numpy.concatenate(
        (arr1[:], arr2[:], arr3[:], arr4[:]),
        axis=0,
    )

    for test_case in [
            slice(-20, -10),
            slice(15, 36),
            slice(20, 10),
            slice(20, 30),
            slice(20, 36),
            slice(35, 36),
            slice(5, 15),
            slice(5, 5),
            slice(None, -5),
            slice(None, 2),
            slice(None, 20),
            slice(None),
    ]:
        selarr = cat[test_case, :]
        selvar = var.isel((test_case, ))
        assert selvar.shape == selarr.shape
        assert selvar.size == selarr.size
        assert numpy.all(selvar.values == selarr)
        assert numpy.all(selvar.data.compute() == selarr)
        assert numpy.all(selvar[:10, :] == selarr[:10, :])
        assert numpy.all(selvar[-5:, :] == selarr[-5:, :])
        assert numpy.all(selvar[:-1, :] == selarr[:-1, :])
        selarr += 10
        selvar.data = selarr
        assert numpy.all(selvar.values == selarr)

    # Vectorized indexing
    selarr = cat[slice(0, None, 2), :]
    selvar = var.isel((slice(0, None, 2), ))
    assert selvar.shape == selarr.shape
    assert selvar.size == selarr.size
    assert numpy.all(selvar.values == selarr)
    assert numpy.all(selvar.data.compute() == selarr)
    assert numpy.all(selvar[2:-1, :] == selarr[2:-1, :])
    assert numpy.all(selvar[-8:-1, :] == selarr[-8:-1, :])

    # Slice indexing of a view
    selarr = selarr[2:-1, :]
    selvar = selvar.isel((slice(2, -1), ))
    assert selvar.shape == selarr.shape
    assert selvar.size == selarr.size
    assert numpy.all(selvar.values == selarr)
    assert numpy.all(selvar.data.compute() == selarr)
    assert numpy.all(selvar[2:-1, :] == selarr[2:-1, :])

    # Vectorized indexing of a view
    selarr = selarr[slice(2, -1, 2), :]
    selvar = selvar.isel((slice(2, -1, 2), ))
    assert selvar.shape == selarr.shape
    assert selvar.size == selarr.size
    assert numpy.all(selvar.values == selarr)
    assert numpy.all(selvar.data.compute() == selarr)
    assert numpy.all(selvar[2:-1, :] == selarr[2:-1, :])


def test_array_getitem(dask_client):
    """Test array getitem."""
    var, (arr1, arr2, arr3, arr4) = create_concatenated_variable()

    cat = numpy.concatenate(
        (arr1[:], arr2[:], arr3[:], arr4[:]),
        axis=0,
    )

    assert numpy.all(var[0:20, :] == arr1)
    assert numpy.all(var[20:30, :] == arr2)
    assert numpy.all(var[30:35, :] == arr3)
    assert numpy.all(var[35:38, :] == arr4)

    assert numpy.all(var[15:25, :] == cat[15:25, :])
    assert numpy.all(var[15:32, :] == cat[15:32, :])
    assert numpy.all(var[15:36, :] == cat[15:36, :])
    assert numpy.all(var[15:39, :] == cat[15:39, :])
    assert numpy.all(var[45:25, :] == cat[45:25, :])

    assert numpy.all(var[15:, :] == cat[15:, :])
    assert numpy.all(var[:25, :] == cat[:25, :])

    assert numpy.all(var[-10:-15, :] == cat[-10:-15, :])
    assert numpy.all(var[:-1, :] == cat[:-1, :])

    assert numpy.all(var[::5, ::5] == cat[::5, ::5])

    with pytest.raises(IndexError):
        var[:, :, :]


def test_array_pickle():
    """Test pickling."""
    var, _ = create_concatenated_variable()
    var_unpickled = pickle.loads(pickle.dumps(var))

    assert var_unpickled.name == "var1"
    assert var_unpickled.dtype == var.dtype
    assert var_unpickled.shape == var.shape
    assert var_unpickled.dimensions == var.dimensions
    assert var_unpickled.attrs == var.attrs
    assert var_unpickled.compressor == var.compressor
    assert var_unpickled.fill_value == var.fill_value
    assert var_unpickled.size == var.size
    assert var_unpickled.nbytes == var.nbytes
    assert var_unpickled.filters == var.filters
    assert numpy.all(var_unpickled.values == var.values)


def test_array_datetime64_to_xarray(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test conversion to xarray."""
    dates = numpy.arange(
        numpy.datetime64("2000-01-01", "ms"),
        numpy.datetime64("2000-02-01", "ms"),
        numpy.timedelta64("1", "h"),
    )
    var = array.Array(
        name="time",
        data=zarr.array(dates),
        dimensions=("num_lines", ),
        attrs=(array.Attribute(name="attr", value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta("int64", "int64"), ),
    )
    xr_var = var.to_xarray()
    assert xr_var.dims == ("num_lines", )
    assert xr_var.attrs == dict(attr=1)
    assert xr_var.dtype == "datetime64[ns]"


def test_array_timedelta64_to_xarray():
    """Test conversion to xarray."""
    delta = numpy.diff(
        numpy.arange(
            numpy.datetime64("2000-01-01", "ms"),
            numpy.datetime64("2000-02-01", "ms"),
            numpy.timedelta64("1", "h"),
        ))

    var = array.Array(
        name="timedelta",
        data=zarr.array(delta),
        dimensions=("num_lines", ),
        attrs=(array.Attribute(name="attr", value=1), ),
        compressor=zarr.Blosc(),
        filters=(zarr.Delta("int64", "int64"), ),
    )
    xr_var = var.to_xarray()
    assert xr_var.dims == ("num_lines", )
    assert xr_var.attrs == dict(attr=1)
    assert xr_var.dtype.kind == "m"


def test_array_dimension_less():
    """Concatenate two dimensionless variables."""
    data = numpy.array([0, 1], dtype=numpy.int32)
    args = ("nv", zarr.array(data), ("nv", ))
    kwargs = dict(attrs=(array.Attribute("comment", "vertex"),
                         array.Attribute("units", "1")))
    n_vertex = array.Array(*args, **kwargs)
    assert n_vertex.fill_value is None
    metadata = n_vertex.metadata()
    assert metadata.fill_value is None
    assert meta.Variable.from_config(metadata.get_config()) == metadata

    other = array.Array(*args, **kwargs)

    concatenated = n_vertex.concat((other, ), "time")
    assert numpy.all(concatenated.values == n_vertex.values)
    assert concatenated.metadata() == n_vertex.metadata()


def test_array_fill():
    """Test filling of variables."""
    var, _ = create_concatenated_variable(fill_value=32767)
    assert not var.values.all() is numpy.ma.masked
    var.fill()
    assert var.values.all() is numpy.ma.masked


def test_array_from_zarr():
    """Test creation of variable from zarr array."""
    zarray = zarr.array(numpy.full((20, 10), 1, dtype="int64"))
    zarray.attrs["_ARRAY_DIMENSIONS"] = ("x", "y")
    var = array.Array.from_zarr(zarray,
                                name="var1",
                                dimension="_ARRAY_DIMENSIONS")
    assert var.shape == (20, 10)
    assert var.dtype == "int64"
    assert var.dimensions == ("x", "y")
    assert var.attrs == tuple()
    assert var.compressor == zarray.compressor
    assert var.fill_value == zarray.fill_value
    assert var.size == zarray.size
    assert var.nbytes == zarray.nbytes
    assert var.filters == zarray.filters
    assert numpy.all(var.values == zarray[...])

    zarray = zarr.full((20, 10), 1, dtype="int64")
    zarray.attrs["_ARRAY_DIMENSIONS"] = ("x", "y")
    var = array.Array.from_zarr(zarray,
                                name="var1",
                                dimension="_ARRAY_DIMENSIONS")
    assert var.shape == (20, 10)
    assert var.dtype == "int64"
    assert var.dimensions == ("x", "y")
    assert var.attrs == tuple()
    assert var.compressor == zarray.compressor
    assert var.fill_value == zarray.fill_value
    assert var.size == zarray.size
    assert var.nbytes == zarray.nbytes
    assert var.filters == zarray.filters
    assert var.values.all() is numpy.ma.masked


def test_array_concat_view(
        dask_client,  # pylint: disable=redefined-outer-name,unused-argument
):
    """Test array isel."""
    arr1 = zarr.array(numpy.full((20, 10), 1, dtype="int64"))
    arr2 = zarr.array(numpy.full((10, 10), 2, dtype="int64"))
    arr3 = zarr.array(numpy.full((5, 10), 3, dtype="int64"))
    arr4 = zarr.array(numpy.full((2, 10), 4, dtype="int64"))

    arr = numpy.concatenate((numpy.full(
        (20, 10), 1, dtype="int64"), numpy.full(
            (10, 10), 2, dtype="int64"), numpy.full((5, 10), 3, dtype="int64"),
                             numpy.full((2, 10), 4, dtype="int64")))
    var = array.Array(
        "var1",
        arr,
        dimensions=("x", "y"),
    )

    cat = numpy.concatenate(
        (arr1[:], arr2[:], arr3[:], arr4[:]),
        axis=0,
    )

    selvar1 = var.isel((slice(None, 8), ))
    selvar2 = var.isel((slice(10, 20), ))
    selvar3 = var.isel((slice(22, 28), ))
    selvar4 = var.isel((slice(30, None), ))

    with pytest.raises(ValueError):
        selvar1.concat((selvar2, selvar3, selvar4), "x")
