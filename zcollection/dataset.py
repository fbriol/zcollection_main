# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Dataset
=======
"""
from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import collections

import dask.array.core
import dask.array.creation
import dask.array.ma
import dask.array.routines
import dask.array.wrap
import dask.base
import dask.threaded
import xarray

from . import meta, representation
from .meta import Attribute
from .typing import ArrayLike
from .variable import Array, DelayedArray, Variable


def _dask_repr(array: dask.array.core.Array) -> str:
    """Get the string representation of a dask array.

    Args:
        array: A dask array.

    Returns:
        The string representation of the dask array.
    """
    chunksize = tuple(item[0] for item in array.chunks)
    return f"dask.array<chunksize={chunksize}>"


def _dataset_repr(ds: "Dataset") -> str:
    """Get the string representation of a dataset.

    Args:
        ds: A dataset.

    Returns:
        The string representation of the dataset.
    """
    # Dimensions
    dims_str = representation.dimensions(ds.dimensions)
    lines = [
        f"<{ds.__module__}.{ds.__class__.__name__}>",
        f"  Dimensions: {dims_str}", "Data variables:"
    ]
    # Variables
    if len(ds.variables) == 0:
        lines.append("    <empty>")
    else:
        width = representation.calculate_column_width(ds.variables)
        for name, variable in ds.variables.items():
            dims_str = f"({', '.join(map(str, variable.dimensions))} "
            name_str = f"    {name:<{width}s} {dims_str} {variable.dtype}"
            lines.append(
                representation.pretty_print(
                    f"{name_str}: {_dask_repr(variable.data)}"))
    # Attributes
    if len(ds.attrs):
        lines.append("  Attributes:")
        lines += representation.attributes(ds.attrs)

    return "\n".join(lines)


class Dataset:
    """Hold variables, dimensions, and attributes that together form a dataset.

    Attrs:
        variables: Dataset variables
        attrs: Dataset attributes

    Raises:
        ValueError: If the dataset contains variables with the same dimensions
            but with different values.
    """
    __slots__ = ("dimensions", "variables", "attrs")

    def __init__(self,
                 variables: Iterable[Variable],
                 attrs: Optional[Sequence[Attribute]] = None) -> None:
        #: The list of global attributes on this dataset
        self.attrs = attrs or []
        #: Dataset contents as dict of :py:class:`Variable` objects.
        self.variables = collections.OrderedDict(
            (item.name, item) for item in variables)
        #: A dictionary of dimension names and their index in the dataset
        self.dimensions: Dict[str, int] = {}

        for var in self.variables.values():
            for ix, dim in enumerate(var.dimensions):
                if dim not in self.dimensions:
                    self.dimensions[dim] = var.shape[ix]
                elif self.dimensions[dim] != var.shape[ix]:
                    raise ValueError(f"variable {var.name} has conflicting "
                                     "dimensions")

    def __getitem__(self, name: str) -> Variable:
        """Return a variable from the dataset.

        Args:
            name: The name of the variable to return

        Returns:
            The variable

        Raises:
            KeyError: If the variable is not found
        """
        return self.variables[name]

    def __getstate__(self) -> Tuple[Any, ...]:
        return self.dimensions, self.variables, self.attrs

    def __setstate__(self, state: Tuple[Any, ...]) -> None:
        self.dimensions, self.variables, self.attrs = state

    @property
    def nbytes(self) -> int:
        """Return the total number of bytes in the dataset.

        Returns:
            The total number of bytes in the dataset
        """
        return sum(item.nbytes for item in self.variables.values())

    def add_variable(self,
                     variable: meta.Variable,
                     /,
                     data: Optional[ArrayLike] = None,
                     delayed: bool = True) -> None:
        """Add a variable to the dataset.

        Args:
            variable: The variable to add
            data: The data to add to the variable. If not provided, the variable
                will be created with the default fill value.
        delayed: If True, create a delayed array.

        Raises:
            ValueError: If the variable added has dimensions that conflict with
                existing dimensions, or if the variable has dimensions not
                defined in the dataset.
        """
        if set(variable.dimensions) - set(self.dimensions):
            raise ValueError(
                f"variable {variable.name} has dimensions "
                f"{variable.dimensions} that are not in the dataset")

        if data is None:
            shape = tuple(self.dimensions[dim] for dim in variable.dimensions)
            data = dask.array.wrap.full(shape,
                                        variable.fill_value,
                                        dtype=variable.dtype)
        else:
            for dim, size in zip(variable.dimensions,
                                 data.shape):  # type: ignore
                if size != self.dimensions[dim]:
                    raise ValueError(
                        f"Conflicting sizes for dimension {dim!r}: "
                        f"length {self.dimensions[dim]} on the data but length "
                        f"{size} defined in dataset.")
        self.variables[variable.name] = (DelayedArray if delayed else Array)(
            variable.name,
            data,  # type: ignore
            variable.dimensions,
            attrs=variable.attrs,
            compressor=variable.compressor,
            fill_value=variable.fill_value,
            filters=variable.filters,
        )

    def rename(self, names: Mapping[str, str]) -> None:
        """Rename variables in the dataset.

        Args:
            names: A mapping from old names to new names

        Raises:
            ValueError: If the new names conflict with existing names
        """
        for old, new in names.items():
            if new in self.variables:
                raise ValueError(f"{new} already exists in the dataset")
            self.variables[new] = self.variables.pop(old).rename(new)

    def drops_vars(self, names: Union[str, Sequence[str]]) -> None:
        """Drop variables from the dataset.

        Args:
            names: Variable names to drop.
        """
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = [names]
        # pylint: disable=expression-not-assigned
        {self.variables.pop(name) for name in names}
        # pylint: enable=expression-not-assigned

    def select_vars(self, names: Union[str, Sequence[str]]) -> "Dataset":
        """Return a new dataset containing only the selected variables.

        Args:
            names: Variable names to select.

        Returns:
            A new dataset containing only the selected variables.
        """
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = [names]
        return Dataset(
            [self.variables[name] for name in names],
            self.attrs,
        )

    def metadata(self) -> meta.Dataset:
        """Get the dataset metadata.

        Returns:
            Dataset metadata
        """
        return meta.Dataset(dimensions=tuple(self.dimensions.keys()),
                            variables=tuple(
                                item.metadata()
                                for item in self.variables.values()),
                            attrs=self.attrs)

    @staticmethod
    def from_xarray(ds: xarray.Dataset, delayed: bool = True) -> "Dataset":
        """Create a new dataset from an xarray dataset.

        Args:
            ds: Dataset to convert.
            delayed: If True, the values of the variable are stored in a Dask
                array, otherwise they are directly handled from the Zarr arrays.

        Returns:
            New dataset.
        """
        array_type = DelayedArray if delayed else Array
        variables = [
            array_type(
                name,  # type: ignore
                array.data,  # type: ignore
                tuple(array.dims),
                attrs=tuple(
                    Attribute(*attr)  # type: ignore
                    for attr in array.attrs.items()),
                compressor=array.encoding.get("compressor", None),
                fill_value=array.encoding.get("_FillValue", None),
                filters=array.encoding.get("filters", None))
            for name, array in ds.variables.items()
        ]

        return Dataset(
            variables=variables,
            attrs=tuple(
                Attribute(*item)  # type: ignore
                for item in ds.attrs.items()))

    def to_xarray(self, **kwargs) -> xarray.Dataset:
        """Convert the dataset to an xarray dataset.

        Args:
            **kwargs: Additional parameters are passed through the function
                :py:func:`xarray.conventions.decode_cf_variables`.

        Returns:
            Dataset as an xarray dataset.
        """
        data_vars = collections.OrderedDict(
            (name, variable.to_xarray())
            for name, variable in self.variables.items())
        attrs = collections.OrderedDict(
            (item.name, item.value) for item in self.attrs)
        data_vars, attrs, coord_names = xarray.conventions.decode_cf_variables(
            data_vars, attrs, **kwargs)
        ds = xarray.Dataset(data_vars, attrs=attrs)
        ds = ds.set_coords(coord_names.intersection(data_vars))
        return ds

    def isel(self, slices: Dict[str, Any]) -> "Dataset":
        """Return a new dataset with each array indexed along the specified
        slices.

        Args:
            slices: Dictionary of dimension names and slices

        Returns:
            New dataset.
        """
        dims_invalid = set(slices) - set(self.dimensions)
        if dims_invalid:
            raise ValueError(
                f"Slices contain invalid dimension name(s): {dims_invalid}")
        default = slice(None)
        variables = [
            var.isel(tuple(slices.get(dim, default) for dim in var.dimensions))
            for var in self.variables.values()
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def compute(self, **kwargs) -> "Dataset":
        """Compute the dataset variables.

        Args:
            **kwargs: Additional parameters are passed through to
                :py:func:`dask.array.compute`.

        Returns:
            New dataset.
        """
        for var in self.variables.values():
            var.data = var.compute(**kwargs)
        return self

    def persist(self, **kwargs) -> "Dataset":
        """Persist the dataset variables.

        Args:
            **kwargs: Additional parameters are passed to the function
                :py:func:`dask.array.Array.persist`.

        Returns:
            The dataset with the variables persisted into memory.
        """
        for variable in self.variables.values():
            variable.persist(**kwargs)
        return self

    def concat(self, other: Union["Dataset", Iterable["Dataset"]],
               dim: str) -> "Dataset":
        """Concatenate datasets along a dimension.

        Args:
            other: Datasets to concatenate.
            dim: Dimension along which to concatenate the datasets.

        Returns:
            New dataset.

        Raises:
            ValueError: If the provided sequence of datasets is empty.
        """
        variables = []
        if not isinstance(other, Iterable):
            other = [other]
        if not other:
            raise ValueError("cannot concatenate an empty sequence")
        variables = [
            variable.concat(tuple(item.variables[name] for item in other), dim)
            for name, variable in self.variables.items()
        ]
        return Dataset(variables=variables, attrs=self.attrs)

    def __str__(self) -> str:
        return _dataset_repr(self)

    def __repr__(self) -> str:
        return _dataset_repr(self)


def get_variable_metadata(
        variable: Union[Variable, meta.Variable]) -> meta.Variable:
    """Get the variable metadata.

    Args:
        variable: Variable to get the metadata for.

    Returns:
        Variable metadata.
    """
    if isinstance(variable, Variable):
        return variable.metadata()
    return variable
