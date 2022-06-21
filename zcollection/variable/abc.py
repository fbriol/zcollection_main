# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Variable interface
==================
"""
from typing import (
    Any,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
import abc
import collections
import functools
import operator

import dask.array.core
import numcodecs.abc
import numpy
import xarray
import zarr

from .. import meta, representation
from ..meta import Attribute
from ..typing import NDArray, NDMaskedArray

#: Generic type for a dataset variable.
T = TypeVar('T', bound='Variable')


def _variable_repr(var: "Variable") -> str:
    """Get the string representation of a variable.

    Args:
        var: A variable.

    Returns:
        The string representation of the variable.
    """
    # Dimensions
    dims_str = representation.dimensions(dict(zip(var.dimensions, var.shape)))
    lines = [
        f"<{var.__module__}.{var.__class__.__name__} {dims_str}>",
        f"{var.data!r}"
    ]

    # Attributes
    if len(var.attrs):
        lines.append("  Attributes:")
        lines += representation.attributes(var.attrs)

    # Filters
    if var.filters:
        lines.append("  Filters:")
        lines += [f"    {item!r}" for item in var.filters]  # type: ignore

    # Compressor
    if var.compressor:
        lines.append("  Compressor:")
        lines += [f"    {var.compressor!r}"]
    return "\n".join(lines)


def prod(iterable: Iterable) -> int:
    """Get the product of an iterable.

    Args:
        iterable: An iterable.

    Returns:
        The product of the iterable.
    """
    return functools.reduce(operator.mul, iterable, 1)


class Variable(abc.ABC):
    """Variables hold multi-dimensional arrays of data.

    Args:
        name: Variable name.
        data: Variable data.
        dimensions: Variable dimensions.
        attrs: Variable attributes.
        compressor: Compressor used to compress the data during writing data to
            disk.
        fill_value: Value to use for uninitialized values.
        filters: Filters to apply before writing data to disk.
    """
    __slots__ = ("_array", "attrs", "compressor", "dimensions", "fill_value",
                 "filters", "name")

    def __init__(
            self,
            name: str,
            data: Any,
            dimensions: Sequence[str],
            *,
            attrs: Optional[Sequence[Attribute]] = None,
            compressor: Optional[numcodecs.abc.Codec] = None,
            fill_value: Optional[Any] = None,
            filters: Optional[Sequence[numcodecs.abc.Codec]] = None) -> None:
        #: Variable name
        self.name = name

        #: Variable data
        self._array = data

        #: Variable dimensions
        self.dimensions = dimensions

        #: Variable attributes
        self.attrs: Sequence[Attribute] = attrs or tuple()

        #: Compressor used to compress the data during writing data to disk
        self.compressor = compressor

        #: Value to use for uninitialized values
        self.fill_value = fill_value

        #: Filters to apply before writing data to disk
        self.filters = filters

    @property
    @abc.abstractmethod
    def array(self) -> dask.array.core.Array:
        """Variable data."""

    @property
    @abc.abstractmethod
    def data(self) -> dask.array.core.Array:  # type:ignore
        """Return the array where values equal the fill value are masked.

        If no fill value is set, the array is returned.
        """

    @data.setter
    @abc.abstractmethod
    def data(self, data: Any) -> None:
        """Set the variable data."""

    @property
    @abc.abstractmethod
    def values(self) -> Union[NDArray, NDMaskedArray]:  # type: ignore
        """Return the values of the array."""

    @property
    @abc.abstractmethod
    def dtype(self) -> numpy.dtype:  # type: ignore
        """Return the data type of the variable."""

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]:  # type: ignore
        """Return the shape of the variable."""

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the variable."""
        return len(self.dimensions)

    @property
    def size(self: Any) -> int:
        """Return the size of the variable."""
        return prod(self.shape)

    @property
    def nbytes(self):
        """Return the number of bytes used by the variable."""
        return self.size * self.dtype.itemsize

    @abc.abstractmethod
    def persist(self: T, **kwargs) -> T:
        """Persist the variable data into memory.

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            The variable
        """

    @abc.abstractmethod
    def compute(self, **kwargs) -> Union[NDArray, NDMaskedArray]:
        """Return the variable data as a numpy array.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            The variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.
        """

    @abc.abstractmethod
    def fill(self: T) -> T:
        """Fill the variable with the fill value. If the variable has no fill
        value, this method does nothing.

        Returns:
            The variable.
        """

    @classmethod
    @abc.abstractmethod
    def from_zarr(
        cls: T,  # type: ignore
        array: zarr.Array,
        name: str,
        dimension: str,
        **kwargs,
    ) -> T:  # type: ignore
        """Create a new variable from a zarr array.

        Args:
            array: The zarr array.
            name: Name of the variable.
            dimension: Name of the attribute that defines the dimensions of the
                variable.
            **kwargs: Additional keyword arguments.

        Returns:
            The variable.
        """

    @abc.abstractmethod
    def concat(self, other: Union[T, Sequence[T]], dim: str) -> T:
        """Concatenate this variable with another variable or a list of
        variables along a dimension.

        Args:
            other: Variable or list of variables to concatenate with this
                variable.
            dim: Dimension to concatenate along.

        Returns:
            New variable.

        Raises:
            ValueError: if the variables provided is an empty sequence.
        """

    @abc.abstractmethod
    def __getitem__(self, key: Any) -> Any:
        """Get a slice of the variable.

        Args:
            key: Slice or index to use.

        Returns:
            The variable slice.
        """

    @abc.abstractmethod
    def isel(self: T, key: Tuple[slice, ...]) -> T:
        """Return a new variable with a subset of the data.

        Args:
            key: Tuple of indexes to use.

        Returns:
            The new variable.
        """

    def metadata(self) -> meta.Variable:
        """Get the variable metadata.

        Returns:
            Variable metadata.
        """
        return meta.Variable(self.name, self.dtype, self.dimensions,
                             self.attrs, self.compressor, self.fill_value,
                             self.filters)

    def have_same_properties(self, other: "Variable") -> bool:
        """Return true if this instance and the other variable have the same
        properties."""
        return self.metadata() == other.metadata()

    def rename(self: T, name: str) -> T:
        """Rename the variable.

        Args:
            name: New variable name.

        Returns:
            The variable.
        """
        return self.__class__(name,
                              self.array,
                              self.dimensions,
                              attrs=self.attrs,
                              compressor=self.compressor,
                              fill_value=self.fill_value,
                              filters=self.filters)

    def dimension_index(self) -> Iterator[Tuple[str, int]]:
        """Return an iterator over the variable dimensions and their index.

        Returns:
            An iterator over the variable dimensions
        """
        yield from ((item, ix) for ix, item in enumerate(self.dimensions))

    def to_xarray(self) -> xarray.Variable:
        """Convert the variable to an xarray.Variable.

        Returns:
            Variable as an xarray.Variable
        """
        encoding = {}
        if self.filters:
            encoding["filters"] = self.filters
        if self.compressor:
            encoding["compressor"] = self.compressor
        data = self.data
        if self.dtype.kind == "M":
            # xarray need a datetime64[ns] dtype
            data = data.astype("datetime64[ns]")
            encoding["dtype"] = "int64"
        elif self.dtype.kind == "m":
            encoding["dtype"] = "int64"
        attrs = collections.OrderedDict(
            (item.name, item.value) for item in self.attrs)
        if self.fill_value is not None:
            attrs["_FillValue"] = self.fill_value
        return xarray.Variable(self.dimensions, data, attrs, encoding)

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        """Return a string representation of the variable."""
        return _variable_repr(self)

    def __repr__(self) -> str:
        """Return a string representation of the variable."""
        return _variable_repr(self)
