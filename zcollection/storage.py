# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
I/O operations
==============
"""
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import collections
import json
import logging

import dask.array.core
import dask.distributed
import dask.local
import fsspec
import numpy
import zarr

from . import dataset, meta, sync, variable
#
from .typing import ArrayLike

#: Block size limit used with dask arrays. (128 MiB)
BLOCK_SIZE_LIMIT = 134217728

#: Name of the attribute storing the names of the dimensions of an array.
DIMENSIONS = "_ARRAY_DIMENSIONS"

#: Configuration file that describes the attributes of an array.
ZATTRS = ".zattrs"

#: Configuration file that describes the attributes of a group.
ZGROUP = ".zgroup"

#: Module logger.
_LOGGER = logging.getLogger(__name__)


def execute_transaction(
    client: dask.distributed.Client,
    synchronizer: sync.Sync,
    futures: Sequence[dask.distributed.Future],
) -> Any:
    """Execute a transaction in the collection.

    Args:
        client: The Dask client.
        synchronizer: The instance handling access to critical resources.
        futures: Lazy tasks to be done.

    Returns:
        The result of the transaction.
    """
    if not futures:
        return None
    awaitables = []
    try:
        with synchronizer:
            awaitables = client.compute(futures)
            return client.gather(awaitables)
    except:  # noqa: E722
        # Before throwing the exception, we wait until all future scheduled
        # ones finished.
        dask.distributed.wait(awaitables)
        dask.distributed.wait(futures)
        raise


def _to_zarr(array: dask.array.core.Array, mapper: fsspec.FSMap, path: str,
             **kwargs) -> None:
    """Write a Dask array to a Zarr dataset.

    Args:
        array: The array to write.
        mapper: The file system mapper.
        path: The path to the Zarr dataset.
        **kwargs: Keyword arguments to pass to :func:`zarr.create`.
    """
    chunks = [chunk[0] for chunk in array.chunks]
    target = zarr.create(
        shape=array.shape,
        chunks=chunks,  # type: ignore
        dtype=array.dtype,
        store=mapper,
        path=path,
        overwrite=True,
        write_empty_chunks=False,
        **kwargs)
    array.store(target,
                lock=False,
                compute=True,
                scheduler=dask.local.get_sync,
                return_stored=False)


def write_zattrs(
    dirname: str,
    var: Union[meta.Variable, dataset.Variable],
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Write the attributes of a variable to a Zarr dataset.

    Args:
        dirname: The storage directory of the Zarr dataset.
        var: The variable to process.
        fs: The file system on which the Zarr dataset is stored.
    """
    attrs = collections.OrderedDict(item.get_config() for item in var.attrs)
    attrs[DIMENSIONS] = var.dimensions
    path = fs.sep.join((dirname, var.name, ZATTRS))
    with fs.open(path, mode="w") as stream:
        json.dump(attrs, stream, indent=2)  # type: ignore


def write_zarr_variable(
    args: Tuple[str, variable.Variable],
    dirname: str,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Write a variable to a Zarr dataset.

    Args:
        args: The arguments to the function:
            - Name of the variable to write.
            - The variable to write.
        dirname: The target directory.
        fs: The file system on which the Zarr dataset is stored.
    """
    name, var = args
    kwargs = dict(filters=var.filters)
    data = var.array

    chunks = {ix: -1 for ix in range(var.ndim)}
    data = data.rechunk(
        chunks,  # type: ignore
        block_size_limit=BLOCK_SIZE_LIMIT,
    )

    _to_zarr(array=data,
             mapper=fs.get_mapper(dirname),
             path=name,
             compressor=var.compressor,
             fill_value=var.fill_value,
             **kwargs)
    write_zattrs(dirname, var, fs)


def write_zarr_group(
    ds: dataset.Dataset,
    dirname: str,
    fs: fsspec.AbstractFileSystem,
    synchronizer: sync.Sync,
) -> None:
    """Write a partition to a Zarr group.

    Args:
        ds: The dataset to write.
        dirname: The name of the partition.
        fs: The file system that the partition is stored on.
        synchronizer: The instance handling access to critical resources.
    """
    with dask.distributed.worker_client() as client:
        iterables = [(name, client.scatter(variable))
                     for name, variable in ds.variables.items()]
        futures = client.map(write_zarr_variable,
                             iterables,
                             dirname=dirname,
                             fs=fs)
        execute_transaction(client, synchronizer, futures)

    path = fs.sep.join((dirname, ZATTRS))
    attrs = collections.OrderedDict(item.get_config() for item in ds.attrs)
    with fs.open(path, mode="w") as stream:
        json.dump(
            attrs,
            stream,  # type: ignore
            indent=2)

    path = fs.sep.join((dirname, ZGROUP))
    with fs.open(path, mode="w") as stream:
        json.dump({"zarr_format": 2}, stream, indent=2)  # type: ignore

    zarr.consolidate_metadata(fs.get_mapper(dirname))
    # Invalidate any cached directory information.
    fs.invalidate_cache(dirname)


def open_zarr_array(array: zarr.Array, name: str,
                    delayed: bool) -> dataset.Variable:
    """Open a Zarr array as a Dask array.

    Args:
        array: The Zarr array to open.
        name: The name of the variable.
        delayed: If True, the values of the variable are stored in a Dask
            array, otherwise they are directly handled from the Zarr arrays.

    Returns:
        The variable.
    """
    handler = dataset.DelayedArray if delayed else dataset.Array
    return handler.from_zarr(array, name, DIMENSIONS)


def open_zarr_group(dirname,
                    fs: fsspec.AbstractFileSystem,
                    selected_variables: Optional[Iterable[str]] = None,
                    delayed: bool = True) -> dataset.Dataset:
    """Open a Zarr group stored in a partition.

    Args:
        dirname: The name of the partition.
        fs: The file system that the partition is stored on.
        selected_variables: The list of variables to retain from the Zarr
            group. If None, all variables are selected.
        delayed: If True, the variables are opened as lazy Dask arrays,
            otherwise as Zarr arrays.

    Returns:
        The zarr group stored in the partition.
    """
    _LOGGER.debug("Opening Zarr group %r", dirname)
    store: zarr.Group = zarr.open_consolidated(  # type: ignore
        fs.get_mapper(dirname))
    # Ignore unknown variables to retain.
    selected_variables = set(selected_variables) & set(
        store) if selected_variables is not None else set(store)
    variables = [
        open_zarr_array(store[name], name, delayed)  # type: ignore
        for name in selected_variables
    ]

    return dataset.Dataset(
        variables=variables,
        attrs=tuple(dataset.Attribute(*item) for item in store.attrs.items()))


def update_zarr_array(
    dirname: str,
    array: ArrayLike,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Update a Zarr array.

    Args:
        dirname: The storage directory of the Zarr array..
        array: The data updated to write.
        fs: The file system that the Zarr array is stored on.
    """
    _LOGGER.debug("Updating Zarr array %r", dirname)
    store = zarr.open_array(fs.get_mapper(dirname), mode="a")

    if isinstance(array, dask.array.core.Array):
        array = array.compute()

    if isinstance(array,
                  numpy.ma.MaskedArray) and store.fill_value is not None:
        array = array.filled(store.fill_value)  # type: ignore

    store[:] = array

    # Invalidate any cached directory information.
    fs.invalidate_cache(dirname)


def del_zarr_array(
    dirname: str,
    name: str,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Delete a variable from a Zarr dataset.

    Args:
        dirname: The name of the dataset.
        name: The name of the variable.
        fs: The file system that the dataset is stored on.
    """
    _LOGGER.debug("Deleting Zarr array %r", dirname)
    path = fs.sep.join((dirname, name))
    if fs.exists(path):
        fs.rm(path, recursive=True)
        zarr.consolidate_metadata(fs.get_mapper(dirname))
        # Invalidate any cached directory information.
        fs.invalidate_cache(dirname)


def add_zarr_array(
    dirname: str,
    var: meta.Variable,
    template: str,
    fs: fsspec.AbstractFileSystem,
) -> None:
    """Add a variable to a Zarr dataset.

    Args:
        dirname: The name of the dataset.
        var: The variable to add.
        template: The name of the template variable.
        fs: The file system that the dataset is stored on.
    """
    _LOGGER.debug("Adding variable %r to Zarr dataset %r", var.name, dirname)
    shape = zarr.open(fs.get_mapper(fs.sep.join((dirname, template)))).shape
    store = fs.get_mapper(fs.sep.join((dirname, var.name)))
    zarr.create(
        shape,
        chunks=True,
        dtype=var.dtype,
        compressor=var.compressor,  # type: ignore
        fill_value=var.fill_value,  # type: ignore
        store=store,
        filters=var.filters)
    write_zattrs(dirname, var, fs)
    zarr.consolidate_metadata(fs.get_mapper(dirname))
