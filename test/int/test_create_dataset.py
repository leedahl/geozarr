import asyncio
import unittest
import numpy as np
import xarray as xr
import zarr
from os import path
from geozarr.factory import GeoZarr
from geozarr.dataset import DimensionType


class CreateDatasetTestCase(unittest.TestCase):
    def test_create_dataset(self) -> None:
        """
        This test converts a file into a GeoZarr dataset.

        :return: None
        """
        print('test_create_dataset:')
        resource_path = f"{'/'.join(path.split(__file__)[0].split('/')[0:-1])}/resources"
        test_netcdf_path = f'{resource_path}/SWE_time.nc'

        base_path = f'file://{path.split(__file__)[0]}/test_results'
        file_path = f'{base_path}/create_dataset.zarr'
        with GeoZarr(file_path, 'x') as geo_zarr:
            # Load sample data
            with xr.load_dataset(test_netcdf_path) as dataset:
                # Define schema
                chunks = [1]
                chunks.extend(list(dataset['SWE'].shape[1:]))
                schema = {
                    'grid': {
                        'upperLeft': [433570.9001397601, 4663716.608805167],
                        'unitSize': 800,
                        'crs': dataset['transverse_mercator'].attrs['spatial_ref']
                    },
                    'global_attributes': {'conventions': 'CF-1.11'},
                    'SWE': {
                        'name': 'SWE',
                        'attributes': {
                            'long_name': 'Snow water equivalent',
                            'units': 'm'
                        },
                        'shape': dataset['SWE'].shape,
                        'chunks': chunks,
                        'dtype': np.float64,
                        'dimensions': [
                            ('time', 'time',  'ns', np.dtype('<M8[ns]'), DimensionType.DIMENSION_VALUE),
                            ('y', 'projection_y_coordinate', 'm', np.float64, DimensionType.COORDINATE_Y),
                            ('x', 'projection_x_coordinate', 'm', np.float64, DimensionType.COORDINATE_X)
                        ]
                    }
                }

                geo_zarr.schema = schema

                # Insert Data
                geo_zarr.set_index('/SWE/time', dataset.get_index('time').values)

                loop = asyncio.get_event_loop()
                futures = set()
                for index in range(dataset.variables['SWE'].sizes['time']):
                    data: np.ndarray = dataset.variables['SWE'].values[index]
                    dim_index = [('/SWE/time', dataset.get_index('time').values[index])]
                    futures.add(geo_zarr.insert('/SWE/SWE', data, dim_index))

                loop.run_until_complete(asyncio.gather(*futures))
                loop.close()

        self.assertTrue(True)  # add assertion here

        print('-------------------')


if __name__ == '__main__':
    unittest.main()
