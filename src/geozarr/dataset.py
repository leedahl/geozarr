from enum import Enum
from typing import Optional, Tuple, Any, List
import zarr
import numpy as np

from pyproj import CRS


class DimensionType(Enum):
    DIMENSION_VALUE = 1
    COORDINATE_X = 2
    COORDINATE_Y = 3
    COORDINATE_Z = 4


class GeoZarrDataset:
    """
    This class creates and reads GeoZarr datasets.
    """
    def __init__(self, dataset_path: str, mode: str = 'r') -> None:
        self._dataset_path = dataset_path
        self._mode = mode

        match mode:
            case 'r':
                # Open a GeoZarr dataset
                self._dataset = zarr.open(self._dataset_path, mode='r')

            case 'x':
                # Create a dataset
                self._dataset = zarr.open(self._dataset_path, mode='w')

    @property
    def schema(self) -> dict:
        return {dim: self._dataset.__getattr__(str(dim)).dtype for dim in self._dataset.dims.keys()}

    @schema.setter
    def schema(self, value: dict) -> None:
        def _transverse_mercator(crs: CRS) -> dict:
            """
            Creates a dictionary of Transverse mercator arguments.

            :param crs: The pyproj CRS argument with the values to extract.
            :return: The created dictionary.
            """
            coord_operation_params = crs.source_crs.coordinate_operation.params
            param_dict = {
                'false easting': lambda x: {'falseEast': x.value},
                'false northing': lambda x: {'falseNorth': x.value},
                'scale factor at natural origin': lambda x: {'scale': x.value},
                'longitude of natural origin': lambda x: {'longOrigin': x.value},
                'latitude of natural origin': lambda x: {'latOrigin': x.value}
            }

            params = dict()
            for param in coord_operation_params:
                if param.name.lower() in param_dict:
                    params.update(param_dict[param.name.lower()](param))

            attributes = {
                'grid_mapping_name': 'transverse_mercator',
                'inverse_flattening': crs.datum.ellipsoid.inverse_flattening,
                'longitude_of_prime_meridian': crs.prime_meridian.longitude,
                'false_easting': params['falseEast'] if 'falseEast' in params else 0.0,
                'false_northing': params['falseNorth'] if 'falseNorth' in params else 0.0,
                'scale_factor_at_central_meridian':params['scale'] if 'scale' in params else 1.0,
                'longitude_of_central_meridian': params['longOrigin'] if 'longOrigin' in params else 0.0,
                'latitude_of_projection_origin': params['latOrigin'] if 'latOrigin' in params else 0.0
            }

            if crs.ellipsoid.is_semi_minor_computed:
                attributes['semi_major_axis'] = crs.ellipsoid.semi_major_metre
                attributes['semi_minor_axis'] =  crs.ellipsoid.semi_minor_metre

            return attributes

        if self._mode == 'x':
            if 'global_attributes' in value:
                self._dataset.attrs.update(value['global_attributes'])
                del value['global_attributes']

            grid = value['grid'] if 'grid' in value else None
            grid_mapping = dict()
            if grid is not None:
                if 'crs' not in grid:
                    crs_value = CRS.from_epsg(4326)
                    grid_mapping['spatial_ref'] = crs_value.to_wkt()

                elif grid['crs'].startswith('EPSG:'):
                    crs_value = CRS.from_epsg(int(grid['crs'][5:]))
                    if crs_value.is_projected and (
                        crs_value.coordinate_operation.method_name.upper() == 'TRANSVERSE MERCATOR'
                    ):
                        grid_mapping = _transverse_mercator(crs_value)

                    grid_mapping['spatial_ref'] = crs_value.to_wkt()
                    pass

                else:
                    crs_value = CRS.from_wkt(grid['crs'])
                    grid_mapping = _transverse_mercator(crs_value)
                    grid_mapping['spatial_ref'] = grid['crs']

                grid_mapping['projection_x_coordinate'] = grid['upperLeft'][0]
                grid_mapping['projection_y_coordinate'] = grid['upperLeft'][1]

                del value['grid']

            for var in value.keys():
                var_attr = value[var]['attributes']
                var_attr['_ARRAY_DIMENSIONS'] = [f'/{var}/{item[0]}' for item in value[var]['dimensions']]
                if grid is not None:
                    var_attr['grid_mapping'] = f'/{var}/crs_grid'

                self._dataset.create_group(var)
                dimensions = value[var]['dimensions']
                shape_index = 0
                for dim in dimensions:
                    if dim[4] == DimensionType.DIMENSION_VALUE:
                        self._dataset[var].create_dataset(
                            dim[0], dtype=dim[3], data=np.empty(value[var]['shape'][shape_index], dim[3])
                        )

                    elif dim[4] == DimensionType.COORDINATE_X:
                        self._dataset[var].create_dataset(dim[0], dtype=dim[3], data=[
                            np.array([
                                grid['upperLeft'][0] + grid['unitSize'] * index
                                for index in range(value[var]['shape'][shape_index])
                            ], dim[3]) if grid is not None
                            else np.empty(value[var]['shape'][shape_index], dim[3])
                        ])

                    else:
                        self._dataset[var].create_dataset(dim[0], dtype=dim[3], data=[
                            np.array([
                                grid['upperLeft'][1] + grid['unitSize'] * index
                                for index in range(value[var]['shape'][shape_index])
                            ], dim[3]) if grid is not None
                            else np.empty(value[var]['shape'][shape_index], dim[3])
                        ])

                    self._dataset[var][dim[0]].attrs.update({
                        '_ARRAY_DIMENSION': [f'/{var}/{dim[0]}'],
                        'standard_name': dim[1],
                        'units': dim[2]
                    })
                    shape_index += 1

                self._dataset[var].create_dataset('crs_grid', dtype='S1', shape=0, chunks=False)
                self._dataset[var]['crs_grid'].attrs.update(grid_mapping)
                self._dataset[var].create_dataset(
                    value[var]['name'], dtype=value[var]['dtype'],
                    data=np.empty(value[var]['shape'], value[var]['dtype']),
                    chunks=value[var]['chunks'] if 'chunks' in value[var] else value[var]['shape']
                )
                self._dataset[var][value[var]['name']].attrs.update(var_attr)

        return

    def close(self) -> None:
        pass

    def set_index(self, index_name: str, values: np.ndarray) -> None:
        self._dataset[index_name][:] = values

    # noinspection PyUnusedLocal
    async def insert(
            self, dataset_name: str, data: np.ndarray, indexes: Optional[List[Tuple[str, Any]]] = None
    ) -> None:
        """
        Use to insert data into the dataset.

        :param dataset_name: The name of the dataset variable to update.
        :param data: The data to insert.
        :param indexes: A list of one or more tuples, upto the shape of the dataset, that contain the dim name and value.
        :return: None
        """
        index_expr = f'self._dataset["{dataset_name}"]'
        index_keys = [item[0] for item in indexes]
        for index in self._dataset[dataset_name].attrs['_ARRAY_DIMENSIONS']:
            if index in index_keys:
                index_key_offset = index_keys.index(index)
                index_offset = list(self._dataset[index][:]).index(indexes[index_key_offset][1])
                index_expr += f'[{index_offset}]'
            else:
                index_expr += f'[:]'

        index_expr += ' = data'
        exec(index_expr)
