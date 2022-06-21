# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Indexing chunked arrays.
========================
"""
from typing import Any, List, Optional, Sequence, Tuple, Union
import abc
import dataclasses

import numpy
import zarr

from ..typing import NDArray

#: The type of key for indexing.
Key = Union[slice, NDArray]


def slice_length(key: Key, length: int) -> int:
    """Calculate the length of the slice.

    Args:
        key: The slice to calculate the length of.
        length: The length of the array.

    Returns:
        The length of the slice.
    """
    # To properly calculate the size of the slice, even if the indexes are
    # relative to the end of the array, we use the range function.
    if isinstance(key, slice):
        return len(range(key.start or 0, key.stop or length, key.step or 1))
    return key.size


def calculate_shape(key: Optional[Sequence[Key]],
                    shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate the shape of the array for the given indexer."""
    if key is None:
        return shape
    return tuple(
        slice_length(key_item, shape_item)
        for key_item, shape_item in zip(key, shape))


def expected_shape(shape: Tuple[int, ...],
                   axis: int,
                   value: int = -1) -> Sequence[int]:
    """Return the expected shape of a variable after concatenation."""
    return shape[:axis] + (value, ) + shape[axis + 1:]


def calculate_offset(chunk_size: NDArray, ix: Any) -> int:
    """Calculate the offset of the given chunk."""
    return chunk_size[ix - 1] if ix > 0 else 0


def _expand_indexer(
    key: Any,
    shape: Sequence[int],
) -> Tuple[slice, ...]:
    """Given a key for indexing, return an equivalent key which is a tuple with
    length equal to the number of dimensions of the array."""
    if not isinstance(key, tuple):
        key = (key, )

    ndim = len(shape)
    result = []
    found_ellipsis = False

    for item in key:
        if item is Ellipsis:
            if not found_ellipsis:
                result.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                result.append(slice(None))
        else:
            result.append(item)

    if len(result) > ndim:
        raise IndexError(
            f"too many indices for array: array is {ndim}-dimensional,"
            f"but {len(result)} were indexed")

    result.extend((ndim - len(result)) * [slice(None)])
    for ix, dim in enumerate(shape):
        item = result[ix]
        start, stop, step = item.start or 0, item.stop or dim, item.step or 1
        result[ix] = slice(start + dim if start < 0 else start,
                           stop + dim if stop < 0 else stop, step)
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class ShiftSlice:
    """Properties of the slice offset when the user selects a subset of the
    chunks."""
    #: Axis along which the slice is shifted.
    axis: int

    #: Number of items to shift the slice by.
    offset: int


@dataclasses.dataclass(frozen=True)
class Selection(abc.ABC):
    """Base class for indexing into a variable."""
    key: Sequence[Optional[Key]]
    tensor_domain: Sequence[Key]

    @abc.abstractmethod
    def getitem(self, array: Sequence[zarr.Array], axis: int = 0) -> NDArray:
        """Apply the selection to the given array.

        Args:
            array: The chunked array to index.
            axis: The axis along which the concatenation is performed.

        Returns:
            The values of the array read from the given indexer.
        """


@dataclasses.dataclass(frozen=True)
class SingleSelection(Selection):
    """Selection of a single chunk."""

    def getitem(self, array: Sequence[zarr.Array], axis: int = 0) -> NDArray:
        """Apply the selection to the given array."""
        return array[axis][self.key]


@dataclasses.dataclass(frozen=True)
class EmptySelection(Selection):
    """Empty selection : no chunk is selected."""

    def getitem(self, array: zarr.Array, axis: int = 0) -> NDArray:
        """Apply the selection to the given array."""
        # To be consistent with numpy, we return an empty array with the
        # concatenated dimension equal to zero.
        return numpy.empty(expected_shape(array.shape, axis, 0),
                           dtype=array.dtype)


@dataclasses.dataclass(frozen=True)
class MultipleSelection(Selection):
    """Selection of multiple chunks."""

    def getitem(self, array: Sequence[zarr.Array], axis: int = 0) -> NDArray:
        """Apply the selection to the given array."""
        data = [
            chunk[key] for chunk, key in zip(array, self.key)
            if key is not None
        ]
        return numpy.concatenate(data, axis=axis)


def _shift_to_tensor_domain(
    axis: int,
    key: Sequence[slice],
    tensor_domain: Sequence[Key],
) -> Tuple[slice]:
    """Shift the slice to the previous selection applied.

    Args:
        axis: The chunked axis.
        key: The slice to shift.
        tensor_domain: The indices which it is possible to index
            on each axis of the tensor.

    Returns:
        The shifted indexer key.
    """
    # Reading the offset of the current tensor domain.
    chunked_key = tensor_domain[axis]
    offset = (chunked_key.start if isinstance(chunked_key, slice) else 0)

    # Shift the chunked key to the previous selection if needed.
    if offset != 0:
        key = tuple(
            slice(item.start + offset, item.stop +
                  offset, item.step) if ix == axis else item
            for ix, item in enumerate(key))

    def shift_start(key: Key, new: Optional[int]) -> int:
        """Shift the start of the slice to the view bounds."""
        start = key.start if isinstance(key, slice) else 0
        return max(start, new) if new is not None else start

    def shift_stop(key: Key, new: Optional[int]) -> int:
        """Shift the stop of the slice to the view bounds."""
        stop = key.stop if isinstance(key, slice) else len(key)
        return min(stop, new) if new is not None else stop

    return tuple(
        slice(
            shift_start(limit, new.start),
            shift_stop(limit, new.stop),
            new.step,
        ) for limit, new in zip(tensor_domain, key))


def _calculate_tensor_domain(
        key: Sequence[slice],
        shift: Optional[ShiftSlice] = None) -> Sequence[slice]:
    """Calculation of the tensor domain of the selection.

    Args:
        key: The key for indexing.
        shift: The properties of the slice offset when the user selects a subset
            of the chunks.

    Returns:
        The tensor domain of the selection.
    """
    result = list(slice(item.start, item.stop, item.step) for item in key)

    # If the selection doesn't start at the first chunk, then we need to shift
    # the selection on the chunked axis.
    if shift is not None and shift.offset != 0:
        axis = shift.axis
        result[axis] = slice(
            result[axis].start - shift.offset,
            result[axis].stop - shift.offset,
            None,
        )
    return tuple(result)


def _is_countiguous(key: Optional[Key]) -> bool:
    """Calculate if the current selected view is contiguous.

    Args:
        key: The key for indexing.

    Returns:
        True if the selection is contiguous, False otherwise.
    """
    if key is not None:
        if isinstance(key, slice):
            return (key.step or 1) == 1
        assert isinstance(key, NDArray)
        step = set(numpy.diff(key))
        return len(step) == 1 and step.pop() == 1
    return True


def _calculate_chunked_indices(
    indices: NDArray,
    chunk_size: NDArray,
) -> Tuple[NDArray, List[Optional[Key]]]:
    """Calculate a vector of indices for the selected chunks.

    Args:
        indices: The indices to select in the different chunks.

    Returns:
        The index vector that represents the selected indices on the axis and
        the list of indices to select in each chunk.
    """

    # Looking for the indices of the selected chunks.
    selected_chunk = numpy.searchsorted(chunk_size, indices, side="right")

    # Calculate the entries indicating where the array is divided.
    sections = numpy.where(numpy.diff(selected_chunk))[0] + 1

    # Enumerate the indices of the selected chunks.
    selected_chunk = set(numpy.unique(selected_chunk))

    # Split the selected indices for the selected chunks.
    chunked_indices = list(numpy.array_split(indices, sections))

    # Creation of the table containing all the selected indices.
    all_selected_indices = numpy.concatenate(chunked_indices)

    # Shift this table to the first index of the first selected chunk.
    all_selected_indices -= calculate_offset(chunk_size, min(selected_chunk))

    # Finally, moves the indices of the selected chunks to the first index of
    # the selected chunk.
    for ix in range(chunk_size.size):

        # If the current chunk is not selected, we insert None in the array to
        # indicate that.
        if not selected_chunk & {ix}:
            chunked_indices.insert(ix, None)  # type: ignore
        else:
            chunked_indices[ix] -= calculate_offset(chunk_size, ix)
    return all_selected_indices, chunked_indices  # type: ignore


def _calculate_vectorized_indexer(
    axis: int,
    chunk_size: NDArray,
    expanded_key: Tuple[slice, ...],
    is_countiguous: bool,
    key: slice,
    tensor_domain: Optional[Sequence[Key]],
) -> Selection:
    # In this case we need to build a vector index in order to
    # determine the different indices used in each chunk.
    indices = numpy.arange(key.start, key.stop)
    item = tensor_domain[axis] if tensor_domain is not None else None
    if (item is not None and not is_countiguous):

        # The previous selection was not contiguous, so we need to
        # subsample the indices according to the previous selection.
        assert isinstance(item, NDArray), str(item)
        indices = item[indices]

    if key.step != 1:

        # Subsample the indices according to the step.
        indices = indices[::key.step]

    indices, chunked_indices = _calculate_chunked_indices(indices, chunk_size)

    return MultipleSelection(
        chunked_indices,
        tuple(indices if ix == axis else expanded_key[ix]
              for ix in range(len(expanded_key))))


def get_indexer(
    axis: int,
    chunks: Sequence[zarr.Array],
    key: Any,
    shape: Sequence[int],
    tensor_domain: Optional[Sequence[Key]],
) -> Selection:
    """Get the indexer for the given key.

    Args:
        axis: The axis along which the array is chunked.
        chunks: The chunked array to index.
        key: The key used for indexing.
        shape: The shape of the array.
        tensor_domain: The interval defining the validity range of the array
            indexer. If None, the limits are defined by the shape variable.

    Returns:
        The indexer for the given key.
    """
    expanded_key = _expand_indexer(key, shape)

    if tensor_domain is not None:
        expanded_key = _shift_to_tensor_domain(axis, expanded_key,
                                               tensor_domain)

    if len(chunks) == 1:
        return SingleSelection(expanded_key,
                               _calculate_tensor_domain(expanded_key))

    # Reference to the chunked slice.
    chunked_key = expanded_key[axis]

    # Calculate the cumulative shape of the chunks.
    chunk_size = numpy.cumsum(
        numpy.array(tuple(item.shape[axis] for item in chunks)))

    # Start index on the chunked axis.
    start = chunked_key.start

    # Stop index on the chunked axis.
    stop = min(chunked_key.stop or chunk_size[-1], chunk_size[-1])

    # Slice step on the chunked axis.
    step = chunked_key.step

    # The result of an invalid selection is an empty array.
    if start >= stop:
        return EmptySelection(expanded_key,
                              _calculate_tensor_domain(expanded_key))

    # First index on the chunked axis.
    ix0 = numpy.searchsorted(chunk_size, start, side="right")

    # Last index on the chunked axis.
    ix1 = min(
        len(chunk_size) - 1,
        numpy.searchsorted(chunk_size, stop, side="left"),  # type: ignore int
    )

    # Determine if the current domain is contiguous.
    is_countiguous = _is_countiguous(
        tensor_domain[axis] if tensor_domain else None)

    if step != 1 or not is_countiguous:
        return _calculate_vectorized_indexer(
            axis,
            chunk_size,
            expanded_key,
            is_countiguous,
            slice(start, stop, step),
            tensor_domain,
        )

    # Move the start index forward to the selected chunk.
    start -= calculate_offset(chunk_size, ix0)

    # Move the stop index forward to the selected chunk.
    stop -= calculate_offset(chunk_size, ix1)

    # Build the key for selecting each chunk.
    chunk_slice: List[Optional[Key]] = [None] * len(chunks)

    if ix0 == ix1:
        # Only one chunk is selected.
        chunk_slice[ix0] = slice(start, stop, None)
    else:
        chunk_slice[ix0] = slice(start, None, None)
        chunk_slice[ix1] = slice(None, stop, None)
        chunk_slice[ix0 + 1:ix1] = [slice(None)] * (ix1 - ix0 - 1)

    return MultipleSelection(
        chunk_slice,
        _calculate_tensor_domain(
            expanded_key,
            ShiftSlice(axis, calculate_offset(chunk_size, ix0)),
        ),
    )
