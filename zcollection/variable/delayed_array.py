# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Delayed access to the chunked array.
====================================
"""
from typing import (
    Any,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import dask.array.core
import dask.array.creation
import dask.array.ma
import dask.base
import dask.threaded
import numcodecs.abc
import numpy
import zarr

from ..meta import Attribute
from ..typing import ArrayLike, NDArray, NDMaskedArray
from .abc import Variable


def _asarray(
    arr: ArrayLike,
    fill_value: Optional[Any] = None,
) -> Tuple[dask.array.core.Array, Any]:
    """Convert an array-like object to a dask array.

    Args:
        arr: An array-like object.
        fill_value: The fill value.

    Returns:
        If the data provided is a masked array, the functions return the array
        with masked data replaced by its fill value and the fill value of the
        offered masked array. Otherwise, the provided array and fill value.
    """
    result = dask.array.core.asarray(arr)  # type: dask.array.core.Array
    _meta = result._meta  # pylint: disable=protected-access
    if isinstance(_meta, numpy.ma.MaskedArray):
        if fill_value is not None and fill_value != _meta.fill_value:
            raise ValueError(
                f"The fill value {fill_value!r} does not match the fill value "
                f"{_meta.fill_value!r} of the array.")
        return dask.array.ma.filled(result, _meta.fill_value), _meta.fill_value
    return result, fill_value


class DelayedArray(Variable):
    """Access to the chunked data using Dask arrays.

    Args:
        name: Name of the variable
        data: Variable data
        dimensions: Variable dimensions
        attrs: Variable attributes
        compressor: Compression codec
        fill_value: Value to use for uninitialized values
        filters: Filters to apply before writing data to disk
    """

    def __init__(
            self,
            name: str,
            data: ArrayLike,
            dimensions: Sequence[str],
            *,
            attrs: Optional[Sequence[Attribute]] = None,
            compressor: Optional[numcodecs.abc.Codec] = None,
            fill_value: Optional[Any] = None,
            filters: Optional[Sequence[numcodecs.abc.Codec]] = None) -> None:
        data, fill_value = _asarray(data, fill_value)
        super().__init__(
            name,
            data,
            dimensions,
            attrs=attrs,
            compressor=compressor,
            fill_value=fill_value,
            filters=filters,
        )

    @property
    def dtype(self) -> numpy.dtype:
        """Return the dtype of the underlying array."""
        return self._array.dtype

    @property
    def shape(self) -> Sequence[int]:
        """Return the shape of the underlying array."""
        return self._array.shape

    @property
    def array(self) -> dask.array.core.Array:
        return self._array

    @property
    def data(self) -> dask.array.core.Array:
        """Return the underlying dask array where values equal to the fill
        value are masked. If no fill value is set, the returned array is the
        same as the underlying array.

        Returns:
            The dask array

        .. seealso::

            :meth:`Variable.array`
        """
        if self.fill_value is None:
            return self._array
        return dask.array.ma.masked_equal(self._array, self.fill_value)

    @data.setter
    def data(self, data: Any) -> None:
        """Defines the underlying dask array.

        If the data provided is a masked
        array, it's converted to an array, where the masked values are replaced
        by its fill value, and its fill value becomes the new fill value of
        this instance. Otherwise, the underlying array is defined as the new
        data and the fill value is set to None.
        Args:
            data: The new data to use
        Raises:
            ValueError: If the shape of the data does not match the shape of
                the stored data.
        """
        data, fill_value = _asarray(data, self.fill_value)
        if len(data.shape) != len(self.dimensions):
            raise ValueError("data shape does not match variable dimensions")
        self._array, self.fill_value = data, fill_value

    @property
    def values(self) -> Union[NDArray, NDMaskedArray]:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Returns:
            The variable data
        """
        return self.compute()

    def persist(self, **kwargs) -> "DelayedArray":
        """Persist the variable data into memory.

        Args:
            **kwargs: Keyword arguments passed to
                :func:`dask.array.Array.persist`.

        Returns:
            The variable
        """
        self._array = self._array.persist(**kwargs)
        return self

    def compute(self, **kwargs) -> Union[NDArray, NDMaskedArray]:
        """Return the variable data as a numpy array.

        .. note::

            If the variable has a fill value, the result is a masked array where
            masked values are equal to the fill value.

        Args:
            **kwargs: Keyword arguments passed to
            :func:`dask.array.Array.compute`.
        """
        if self.fill_value is None:
            return self._array.compute(**kwargs)
        return numpy.ma.masked_equal(self._array.compute(**kwargs),
                                     self.fill_value)

    def _duplicate(
        self,
        data: Any,
    ) -> "DelayedArray":
        """Create a new variable with the same properties as this instance, but
        with the given data.

        Args:
            name: Name of the variable
            data: Variable data

        Returns:
            A new variable with the given data.
        """
        # pylint: disable=protected-access
        # _array is a protected member of this class.
        result = self.__class__.__new__(self.__class__)
        result._array = data
        result.attrs = self.attrs
        result.compressor = self.compressor
        result.dimensions = self.dimensions
        result.fill_value = self.fill_value
        result.filters = self.filters
        result.name = self.name
        return result
        # pylint: enable=protected-access

    def fill(self) -> "DelayedArray":
        """Fill the variable with the fill value. If the variable has no fill
        value, this method does nothing.

        Returns:
            The variable.
        """
        if self.fill_value is not None:
            self._array = dask.array.creation.full_like(
                self._array, self.fill_value)
        return self

    @classmethod
    def from_zarr(
        cls,
        array: zarr.Array,
        name: str,
        dimension: str,
        **kwargs,
    ) -> "DelayedArray":
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
        data = dask.array.core.from_array(
            array,
            array.chunks,
            name=f"{name}-{dask.base.tokenize(array, array.chunks)}",
            **kwargs)
        result = cls.__new__(cls)
        result._array = data
        result.attrs = attrs
        result.compressor = array.compressor
        result.dimensions = array.attrs[dimension]
        result.fill_value = array.fill_value
        result.filters = array.filters
        result.name = name
        return result

    def concat(self, other: Union["DelayedArray", Sequence["DelayedArray"]],
               dim: str) -> "DelayedArray":
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
        if not isinstance(other, Sequence):
            other = [other]
        if not other:
            raise ValueError("other must be a non-empty sequence")

        # pylint: disable=protected-access
        # _array is a protected member of this class.
        result = self._duplicate(self._array)

        try:
            axis = self.dimensions.index(dim)
            result._array = dask.array.core.concatenate(
                [self._array, *[item._array for item in other]], axis=axis)
        except ValueError:
            # If the concatenation dimension is not within the dimensions of the
            # variable, then the original variable is returned (i.e.
            # concatenation is not necessary).
            pass
        return result
        # pylint: enable=protected-access

    def isel(self, key: Tuple[slice, ...]) -> "DelayedArray":
        """Return a new variable with the data selected by the given index.

        Args:
            key: Index to select data.

        Returns:
            The variable.
        """
        return self._duplicate(self._array[key])

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __array__(self):
        return self.values

    def to_dask_array(self):
        """Return the underlying dask array.

        Returns:
            The underlying dask array

        .. seealso::

            :func:`dask.array.core.asarray`
        """
        return self.data

    def __dask_graph__(self) -> Optional[Mapping]:
        """Return the dask Graph."""
        return self._array.__dask_graph__()

    def __dask_keys__(self) -> List:
        """Return the output keys for the Dask graph."""
        return self._array.__dask_keys__()

    def __dask_layers__(self) -> Tuple:
        """Return the layers for the Dask graph."""
        return self._array.__dask_layers__()

    def __dask_tokenize__(self):
        """Return the token for the Dask graph."""
        return dask.base.normalize_token(
            (type(self), self.name, self._array, self.dimensions, self.attrs,
             self.fill_value))

    @staticmethod
    def __dask_optimize__(dsk: MutableMapping, keys: List,
                          **kwargs) -> MutableMapping:
        """Returns whether the Dask graph can be optimized.

        .. seealso::
            :func:`dask.array.Array.__dask_optimize__`
        """
        return dask.array.core.Array.__dask_optimize__(dsk, keys, **kwargs)

    #: The default scheduler get to use for this object.
    __dask_scheduler__ = staticmethod(dask.threaded.get)

    def _dask_finalize(self, results, array_func, *args, **kwargs):
        return DelayedArray(
            self.name,
            array_func(results, *args, **kwargs),
            self.dimensions,
            attrs=self.attrs,
            compressor=self.compressor,
            fill_value=self.fill_value,
            filters=self.filters,
        )

    def __dask_postcompute__(self) -> Tuple:
        """Return the finalizer and extra arguments to convert the computed
        results into their in-memory representation."""
        array_func, array_args = self._array.__dask_postcompute__()
        return self._dask_finalize, (array_func, ) + array_args

    def __dask_postpersist__(self) -> Tuple:
        """Return the rebuilder and extra arguments to rebuild an equivalent
        Dask collection from a persisted or rebuilt graph."""
        array_func, array_args = self._array.__dask_postpersist__()
        return self._dask_finalize, (array_func, ) + array_args
