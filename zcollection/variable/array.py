# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Direct access to chunked array.
===============================
"""
from typing import Any, Optional, Sequence, Tuple, Union

import dask.array.core
import dask.array.ma
import dask.base
import numcodecs.abc
import numpy
import zarr

from . import indexing
from .. import utilities
from ..meta import Attribute
from ..typing import NDArray, NDMaskedArray
from .abc import Variable, prod

#: Type of arrays handled by this module.
ArrayLike = Union[NDArray, NDMaskedArray]


class Array(Variable):
    """Direct access handling to the chunked arrays. Reading data will
    automatically load them into memory. This access is preferable if you can
    store all the data read in RAM. Otherwise, delayed accesses using Dask
    should be used.

    Args:
        name: Name of the variable.
        data: Variable data.
        dimensions: Variable dimensions.
        attrs: Variable attributes.
        compressor: Variable compressor.
        fill_value: Value to use for uninitialized values.
        filters: Filters to apply before writing data to disk
    """
    __slots__ = ("_axis", "_tensor_domain")

    def __init__(
            self,
            name: str,
            data: Union[zarr.Array, ArrayLike],
            dimensions: Sequence[str],
            *,
            attrs: Optional[Sequence[Attribute]] = None,
            compressor: Optional[numcodecs.abc.Codec] = None,
            fill_value: Optional[Any] = None,
            filters: Optional[Sequence[numcodecs.abc.Codec]] = None) -> None:
        data = zarr.array(
            numpy.asarray(data)) if not isinstance(data, zarr.Array) else data
        super().__init__(
            name,
            [data],
            dimensions,
            attrs=attrs,
            compressor=compressor or data.compressor,
            fill_value=fill_value,
            filters=filters or data.filters,
        )
        #: Axis along which the variable is chunked. None if the variable is
        #: not chunked.
        self._axis = None

        #: If a view of the array is used, the range limits for each axis on
        #: the original array.
        self._tensor_domain: Optional[Sequence[indexing.Key]] = None

    @property
    def array(self) -> dask.array.core.Array:
        """Return the dask array representing the variable data."""
        axis = self._axis or 0
        key = self._tensor_domain

        arrays = [
            dask.array.core.from_array(
                chunk,
                chunk.size,
                name=f"{self.name}-{dask.base.tokenize(chunk, chunk.chunks)}",
            ) for chunk in self._array
        ]

        if key is not None:
            if len(arrays) == 1:
                return arrays[0][key]
            arrays = dask.array.core.concatenate(arrays, axis=axis)
            return arrays[key]

        if len(arrays) == 1:
            return arrays[0]
        return dask.array.core.concatenate(arrays, axis=axis)

    @property
    def data(self) -> dask.array.core.Array:
        """Returns the dask array where values equal to the fill value are
        masked.

        If no fill value is defined, the returned array is the same as
        the underlying array.
        """
        result = self.array
        if self.fill_value is not None:
            return dask.array.ma.masked_equal(result, self.fill_value)
        return result

    @data.setter
    def data(self, data: ArrayLike) -> None:
        """Defines the underlying Zarr array. If supplied data is a masked
        array, it's converted to an array where the masked values are replaced
        with fill value defined in this instance. Otherwise, the data provided
        set the underlying Zarr array.

        Args:
            data: Variable data to set.
        """
        shape = self.shape
        if self._tensor_domain is not None:
            shape = tuple(
                indexing.slice_length(key, shape[ix])
                for ix, key in enumerate(self._tensor_domain))
        if shape != data.shape:
            raise ValueError(
                f"Shape of data ({data.shape}) does not match shape of "
                f"variable ({shape})")

        # If this instance and the data are empty, we have nothing to do.
        if prod(shape) == 0:
            return

        # If the data is a masked array, we need to convert it to an array.
        if isinstance(data, numpy.ma.MaskedArray):
            data = data.filled(self.fill_value)

        # Calculate the indexing to assign the different chunks with the
        # provided data.
        key = [slice(0, item) for item in shape]
        axis = self._axis or 0
        if self._tensor_domain:
            item = self._tensor_domain[axis]
            shift = item.start if isinstance(item, slice) else item[0]
            item = key[axis]
            key[axis] = slice(item.start + shift, item.stop + shift)
        indexer = indexing.get_indexer(axis, self._array, tuple(key), shape,
                                       None)

        # Finally, set the data on the underlying zarr array an each selected
        # chunk.
        start = 0
        for chunk, chunk_key in zip(self._array, indexer.key):
            if chunk_key is not None:
                stop = indexing.slice_length(chunk_key, chunk.shape[axis])
                key[axis] = slice(start, start + stop)
                chunk[chunk_key] = data[tuple(key)]
                start += stop

    @property
    def values(self) -> ArrayLike:
        """Return the variable data as a numpy array.

        Returns:
            Variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.
        """
        array = (self._array[0][...] if self._axis is None else
                 numpy.concatenate([chunk[...] for chunk in self._array],
                                   axis=self._axis))
        if self._tensor_domain is not None:
            array = array[self._tensor_domain]  # type: ignore
        return array if self.fill_value is None else numpy.ma.masked_equal(
            array, self.fill_value)

    @property
    def dtype(self) -> numpy.dtype:
        """Return the dtype of the underlying array."""
        return self._array[0].dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying array."""
        if self._axis is None:
            return indexing.calculate_shape(self._tensor_domain,
                                            self._array[0].shape)
        size_along_axis = sum(chunk.shape[self._axis] for chunk in self._array)
        return indexing.calculate_shape(
            self._tensor_domain, self._array[0].shape[:self._axis] +
            (size_along_axis, ) + self._array[0].shape[self._axis + 1:])

    def persist(self, **kwargs) -> "Array":
        """Persist the variable data into memory.

        Returns:
            The variable
        """
        return self

    def compute(self, **kwargs) -> ArrayLike:
        """Return the variable data as a numpy array.

        Returns:
            The variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.
        """
        return self.values

    def fill(self) -> "Array":
        """Fill the variable with the fill value. If the variable has no fill
        value, this method does nothing.

        Returns:
            The variable.
        """
        if self.fill_value is not None:
            for chunk in self._array:
                chunk[...] = numpy.full(chunk.shape, self.fill_value,
                                        self.dtype)
        return self

    def _duplicate(
        self,
        data: Any,
        tensor_domain: Optional[Sequence[indexing.Key]] = None,
    ) -> "Array":
        """Create a new variable with the same properties as this instance, but
        with the given data.

        Args:
            name: Name of the variable
            data: Variable data
            tensor_domain: Start/stop index limits for each dimension of the
                variable.

        Returns:
            A new variable with the given data.
        """
        # pylint: disable=duplicate-code, protected-access
        # For optimization, we don't want to call an inherited method from the
        # base class.
        # _array, _axis and _tensor_domain are protected attributes of this
        # class.
        result = self.__class__.__new__(self.__class__)
        result._array = data
        result.attrs = self.attrs
        result._axis = self._axis
        result.compressor = self.compressor
        result.dimensions = self.dimensions
        result.fill_value = self.fill_value
        result.filters = self.filters
        result.name = self.name
        result._tensor_domain = tensor_domain or self._tensor_domain
        return result
        # pylint: enable=duplicate-code, protected-access

    @classmethod
    def from_zarr(
        cls,
        array: zarr.Array,
        name: str,
        dimension: str,
        **kwargs,
    ) -> "Array":
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
        attrs = tuple(
            Attribute(k, v) for k, v in array.attrs.items() if k != dimension)
        return Array(name,
                     array,
                     array.attrs[dimension],
                     attrs=attrs,
                     compressor=array.compressor,
                     fill_value=array.fill_value,
                     filters=array.filters)

    @staticmethod
    def _concatenate_tensor_domain(
        items: Sequence["Array"],
        axis: int,
    ) -> Optional[Sequence[slice]]:
        """Concatenate the tensor domain of a list of arrays.

        Args:
            items: The list of arrays.
            axis: The chunked axis.

        Returns:
            The tensor domain.

        Raises:
            ValueError: If the tensor domain is not consistent or if an indexer
                is defined on the chunked axis.
        """
        # pylint: disable=protected-access
        # _tensor_domain is a protected attribute of this class.
        # If all the arrays doesn't have a tensor domain, we don't need to
        # concatenate
        if all(item._tensor_domain is None for item in items):
            return None

        # If one of the arrays doesn't have a tensor domain, it's an error, as
        # it's not possible to concatenate non-unique chunks
        if any(item._tensor_domain is None for item in items):
            raise ValueError("Unable to concatenate non-uniform chunks")

        def _get_tensor_domain(
                item: "Array") -> Optional[Tuple[Tuple[int, int, int], ...]]:
            """Get the tensor domain of an array.

            Return None if the array has a vectorized indexer on the
            chunked axis.
            """
            result = []
            if item._tensor_domain is not None:
                for key in item._tensor_domain:
                    if not isinstance(key, slice):
                        return None
                    # Slice are not hashable, so we need to convert them to a
                    # tuple
                    result.append((key.start, key.stop, key.step))
            return tuple(result)

        keys = list(_get_tensor_domain(item) for item in items)
        if set(keys) == set({None}) or len(keys) != 1:
            raise ValueError("Unable to concatenate non-uniform chunks")

        key = keys.pop()
        assert key is not None
        if len(key) == 0:
            return None

        if key[axis] is not None:
            raise ValueError(
                "Unable to concatenate chunks along the chunked axis")

        return tuple(slice(*item) for item in key)
        # pylint: enable=protected-access

    def concat(
        self,
        other: Union["Array", Sequence["Array"]],
        dim: str,
    ) -> "Array":
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
        # pylint: disable=protected-access
        # _axis, _array and _tensor_domain are protected attributes of this
        # class.
        if not isinstance(other, Sequence):
            other = [other]
        if not other:
            raise ValueError("other must be a non-empty sequence")

        try:
            axis = self.dimensions.index(dim)
        except ValueError:
            # If the concatenation dimension is not within the dimensions of the
            # variable, then the original variable is returned (i.e.
            # concatenation is not necessary).
            return self._duplicate(self._array)

        # The concatenation axes must always be the same.
        items = set(item._axis for item in other)
        if items not in ({None}, {axis}):
            raise ValueError(
                "Concatenation axes must be the same for all variables. "
                f"Found {items}, expected {axis}.")

        # The data type must always be the same.
        items = set(item.dtype for item in other)
        if items != {self.dtype}:
            raise ValueError(
                "Data types must be the same for all variables. Found "
                f"{items}, expected {self.dtype}.")

        # The fill value must always be the same.
        if any(
                utilities.are_they_different(self.fill_value, item.fill_value)
                for item in other):
            raise ValueError("Fill values must be the same for all variables.")

        # Only the concatenation axis can be chunked.
        shape = indexing.expected_shape(self.shape, axis)
        if any(
                indexing.expected_shape(item.shape, axis) != shape
                for item in other):
            raise ValueError("Expected all variables to have shape " +
                             str(shape).replace("-1", "*"))

        result = self._duplicate(self._array)
        result._axis = axis
        result._tensor_domain = self._concatenate_tensor_domain(
            (self, ) + tuple(other), axis)

        # pylint: disable=expression-not-assigned
        # Using a set comprehension to inline the concatenation.
        {result._array.extend(item._array) for item in other}
        # pylint: enable=expression-not-assigned
        return result
        # pylint: enable=protected-access

    def isel(self, key: Tuple[slice, ...]) -> "Array":
        """Select a subset of the variable.

        Args:
            key: Dictionary of dimension names and slice objects.

        Returns:
            The variable slice.
        """
        indexer = indexing.get_indexer(self._axis or 0, self._array, key,
                                       self.shape, self._tensor_domain)

        # Did the user select a single chunk or an empty data set?
        if isinstance(indexer,
                      (indexing.SingleSelection, indexing.EmptySelection)):
            return self._duplicate(self._array,
                                   tensor_domain=indexer.tensor_domain)

        # The user has selected a set of chunks.
        assert isinstance(indexer, indexing.MultipleSelection)
        return self._duplicate(
            [
                # Discard chunks that are not part of this selection.
                array for key, array in zip(indexer.key, self._array)
                if key is not None
            ],
            tensor_domain=indexer.tensor_domain,
        )

    def __getitem__(self, key: Tuple[slice, ...]) -> numpy.ndarray:
        """Get a slice of the variable.

        Args:
            key: Slice or index to use.

        Returns:
            The variable slice.
        """
        return self.isel(key).values
