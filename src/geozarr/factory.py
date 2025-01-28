from geozarr.dataset import GeoZarrDataset


class GeoZarr:
    """
    This class creates GeoZarr dataset instances.
    """
    def __init__(self, dataset_path: str, mode: str = 'r') -> None:
        self._dataset_path = dataset_path
        self._mode = mode
        self._dataset = None

    def __enter__(self) -> GeoZarrDataset:
        """
        Opens or creates a new GeoZarr dataset.

        :return: The new or existing GeoZarr dataset.
        """
        self._dataset = self.open(self._dataset_path, self._mode)

        return self._dataset

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._dataset.close()

    @classmethod
    def open(cls, dataset_path: str, mode: str = 'r') -> GeoZarrDataset:
        return GeoZarrDataset(dataset_path, mode)
