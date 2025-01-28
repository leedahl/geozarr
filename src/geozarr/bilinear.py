from datetime import datetime, timezone
import numpy as np


def create_overview(array_shape: tuple, array_type: np.dtype, content: bytes, zoom_level: int) -> np.ndarray:
    """
    Create a zoom level overview of the input image array.  Return an image array of the overview.  The overview pixel
    size will be a power of 2 larger than the input image which results in a power of 2 fewer pixels.  The power of 2
    factor is based on the zoom level.  Because the weight in a bilinear interpolation will always be 0.5 in both the
    x and y direction because of the power of 2 zoom, the result of the both interpolations is equivalent to the
    average of the non-zero values for the four neighboring pixels.  Thus, the bilinear equation in this routine is
    simplified to the weighted average of the four neighboring pixels.  The weight applied to each pixel is 1 for
    non-zero pixels and 0 for zero pixels.

    :param array_shape: The shape of the array to crate from the content.
    :param array_type: The data type of the array.
    :param content: The content of the array as bytes.
    :param zoom_level: The level of the zoom to create the overview for.
    :return: The image created.
    """

    image = np.ndarray(array_shape, array_type, content)
    bands = image.shape[-1]
    start = 2 ** (zoom_level - 1) - 1
    step = 2 ** zoom_level
    zoom_shape = (image.shape[0] // step, image.shape[1] // step, bands)

    zoom_image = np.zeros(zoom_shape, array_type)
    for row in range(start, image.shape[0], step):
        for col in range(start, image.shape[1], step):
            pixel_values = image[row:row + 2, col:col + 2]
            if pixel_values.max() > 0:  # At least one color value.
                weight = pixel_values > 0
                avg_values = np.round(np.average(pixel_values, (0, 1), weight))
                zoom_image[row // step, col // step] = avg_values

            else:
                zoom_image[row // step, col // step] = 0

    return zoom_image


# Example usage:
if __name__ == "__main__":
    # Create a dummy batch of 10,000 tiles with random values between 0 and 255
    num_tiles = 1000
    tiles_256 = np.random.randint(0, 256, (num_tiles, 256, 256, 3), dtype=np.uint8)

    # Choose a zoom level
    level = 1

    # Perform bilinear interpolation to reduce resolution
    times = list()
    for tile in range(num_tiles):
        start_time = datetime.now(timezone.utc).timestamp()
        create_overview(tiles_256[tile].shape, tiles_256[tile].dtype, tiles_256[tile].tobytes(), level)
        end_time = datetime.now(timezone.utc).timestamp()
        delta = end_time - start_time
        times.append(delta)

    average = sum(times) / num_tiles

    print(f"Number of tiles: {num_tiles}.")
    print(f"Time: {sum(times)} seconds; Average: {average} seconds per tile.")
