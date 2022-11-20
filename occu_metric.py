import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np


class ResamplingType(enum.Enum):
  NEAREST = 0
  BILINEAR = 1


class BorderType(enum.Enum):
  ZERO = 0
  DUPLICATE = 1


class PixelType(enum.Enum):
  INTEGER = 0
  HALF_INTEGER = 1

import enum
from typing import Optional
from typing import Union, Sequence

Integer = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64]
Float = Union[float, np.float16, np.float32, np.float64]
TensorLike = Union[Integer, Float, Sequence, np.ndarray, torch.Tensor]

def sample(image: torch.Tensor,
           warp: torch.Tensor,
           resampling_type: ResamplingType = ResamplingType.BILINEAR,
           border_type: BorderType = BorderType.ZERO,
           pixel_type: PixelType = PixelType.HALF_INTEGER) -> torch.Tensor:
    """Samples an image at user defined coordinates.

    Note:
        The warp maps target to source. In the following, A1 to An are optional
        batch dimensions.

    Args:
        image: A tensor of shape `[B, H_i, W_i, C]`, where `B` is the batch size,
        `H_i` the height of the image, `W_i` the width of the image, and `C` the
        number of channels of the image.
        warp: A tensor of shape `[B, A_1, ..., A_n, 2]` containing the x and y
        coordinates at which sampling will be performed. The last dimension must
        be 2, representing the (x, y) coordinate where x is the index for width
        and y is the index for height.
    resampling_type: Resampling mode. Supported values are
        `ResamplingType.NEAREST` and `ResamplingType.BILINEAR`.
        border_type: Border mode. Supported values are `BorderType.ZERO` and
        `BorderType.DUPLICATE`.
        pixel_type: Pixel mode. Supported values are `PixelType.INTEGER` and
        `PixelType.HALF_INTEGER`.
        name: A name for this op. Defaults to "sample".

    Returns:
        Tensor of sampled values from `image`. The output tensor shape
        is `[B, A_1, ..., A_n, C]`.

    Raises:
        ValueError: If `image` has rank != 4. If `warp` has rank < 2 or its last
        dimension is not 2. If `image` and `warp` batch dimension does not match.
    """

    # shape.check_static(image, tensor_name="image", has_rank=4)
    # shape.check_static(
    #     warp,
    #     tensor_name="warp",
    #     has_rank_greater_than=1,
    #     has_dim_equals=(-1, 2))
    # shape.compare_batch_dimensions(
    #     tensors=(image, warp), last_axes=0, broadcast_compatible=False)

    if pixel_type == PixelType.HALF_INTEGER:
        warp -= 0.5

    if resampling_type == ResamplingType.NEAREST:
        warp = torch.math.round(warp)

    if border_type == BorderType.ZERO:
        image = F.pad(image.permute(0,3,1,2), (0,1,1,0)).permute(0,2,3,1)
        warp = warp + 1

    warp_shape = warp.size()
    flat_warp = torch.reshape(warp, (warp_shape[0], -1, 2))
    flat_sampled = _interpolate_bilinear(image, flat_warp, indexing="xy")
    output_shape = [*warp_shape[:-1], flat_sampled.size()[-1]]
    return torch.reshape(flat_sampled, output_shape)

def _interpolate_bilinear(
    grid,
    query_points,
    indexing,
):
    """pytorch implementation of tensorflow interpolate_bilinear."""
    device = grid.device
    grid_shape = grid.size()
    query_shape = query_points.size()

    batch_size, height, width, channels = (
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        grid_shape[3],
    )

    num_queries = query_shape[1]

    query_type = query_points.dtype
    grid_type = grid.dtype

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == "ij" else [1, 0]
    unstacked_query_points = torch.unbind(query_points, dim=2)

    for i, dim in enumerate(index_order):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = grid_shape[i + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_type, device=device)
        min_floor = torch.tensor(0.0, dtype=query_type, device=device)
        floor = torch.minimum(
            torch.maximum(min_floor, torch.floor(queries)), max_floor
        )
        int_floor = floor.to(torch.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = (queries - floor).to(grid_type)
        min_alpha = torch.tensor(0.0, dtype=grid_type, device=device)
        max_alpha = torch.tensor(1.0, dtype=grid_type, device=device)
        alpha = torch.minimum(torch.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 2)
        alphas.append(alpha)

        flattened_grid = torch.reshape(grid, [batch_size * height * width, channels])
        batch_offsets = torch.reshape(
            torch.arange(0, batch_size, device=device) * height * width, [batch_size, 1]
        )

    # This wraps tf.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using tf.gather_nd.
    def gather(y_coords, x_coords, name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = flattened_grid[linear_coordinates]
        return torch.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], "top_left")
    top_right = gather(floors[0], ceils[1], "top_right")
    bottom_left = gather(ceils[0], floors[1], "bottom_left")
    bottom_right = gather(ceils[0], ceils[1], "bottom_right")

    # now, do the actual interpolation
    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp

