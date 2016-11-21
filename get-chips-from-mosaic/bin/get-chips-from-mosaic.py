# gbdx.Task(get-chips-from-mosaic, geojson, bucket_name,  mosaic_location, aws_access_key, aws_secret_key, min_side_dim, max_side_dim)

## ASSUMPTIONS ##
# mosaic data strux will be consistent
# mosaic saved as bucket_name/mosaic_location
# appropriate vrt shapefile located at .../wms/vsitindex_z12.shp

import logging
import geojson
import subprocess, os

from functools import partial
from osgeo import gdal
from multiprocessing import Pool, Process, cpu_count
from gbdx_task_interface import GbdxTaskInterface

# log file for debugging
logging.basicConfig(filename='out.log',level=logging.DEBUG)

def execute_command(cmd):
    '''
    Execute a command. This is outside of the class because you cannot use
        multiprocessing.Pool on a class methods.
    '''
    subprocess.call(cmd, shell=True)

def check_mask_chip(feature, min_side_dim=0, max_side_dim=None, mask=False):
    '''
    check a chip for appropriate size. This is outside of the class because you cannot
        use multiprocessing.Pool on a class methods.
    '''

    if not (max_side_dim or min_side_dim or mask):
        return True

    # Open chip in gdal
    chip_name = str(feature['properties']['feature_id']) + '.tif'
    chip = gdal.Open(chip_name)

    min_side = min(chip.RasterXSize, chip.RasterYSize)
    max_side = max(chip.RasterXSize, chip.RasterYSize)

    # Close chip
    chip = None

    # Remove chip if too small or large
    if max_side_dim or (min_side_dim > 0):
        if min_side < min_side_dim or max_side > max_side_dim:
            os.remove(chip_name)
            os.remove(chip_name + '.msk')
            return True

    # Mask area outside polygon
    if mask:
        vectordata = {'type': 'FeatureCollection', 'features': [feature]}
        fn = chip_name.strip('.tif') + '.geojson'

        # Save polygon to ogr format
        with open(fn, 'wb') as f:
            geojson.dump(vectordata, f)

        # Mask raster
        cmd = 'gdal_rasterize -i -b 1 -b 2 -b 3 -burn 0 -burn 0 -burn 0 {} {}'.format(fn, chip_name)
        subprocess.call(cmd, shell=True)

        os.remove(fn)


class GetChipsFromMosaic(GbdxTaskInterface):
    '''
    Extract features in a geojson from a mosaic on S3
    '''

    def __init__(self):
        '''
        Get inputs
        '''
        GbdxTaskInterface.__init__(self)
        self.execute_command = execute_command
        self.check_mask_chip = check_mask_chip

        self.geojson_dir = self.get_input_data_port('geojson')
        self.geojsons = [f for f in os.listdir(self.geojson_dir) if f.endswith('.geojson')]
        logging.info('geojson directory: ' + self.geojson_dir)
        logging.info('geojson contents: ' + str(self.geojsons))

        # Assert exactly one geojson file passed
        if len(self.geojsons) != 1:
            logging.debug('There are {} geojson files found in the geojson directory'.format(str(len(self.geojsons))))
            raise AssertionError('Please make sure there is exactly one geojson file in the geojson directory. {} found.'.format(str(len(self.geojsons))))

        self.geojson = os.path.join(self.geojson_dir, self.geojsons[0])


        self.bucket = self.get_input_string_port('bucket_name')
        logging.info('bucket name: ' + self.bucket)

        self.mosaic = self.get_input_string_port('mosaic_location')
        logging.info('mosaic: ' + self.mosaic)

        self.min_side_dim = int(self.get_input_string_port('min_side_dim', default = '0'))
        logging.info('min_side_dim: ' + str(self.min_side_dim))

        self.max_side_dim = int(self.get_input_string_port('max_side_dim', default = 'None'))
        logging.info('max_side_dim: ' + str(self.max_side_dim))

        self.mask = bool(self.get_input_string_port('mask', default='False'))
        self.a_key = self.get_input_string_port('aws_access_key', default=None)
        self.s_key = self.get_input_string_port('aws_secret_key', default=None)

        # Set AWS environment variables
        if self.a_key and self.s_key:
            os.environ['AWS_ACCESS_KEY_ID'] = self.a_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = self.s_key
            logging.info('Set AWS env variables.')

        # Create output directory
        self.out_dir = self.get_output_data_port('chips')
        os.makedirs(self.out_dir)


    def create_vrt(self):
        '''
        create a vrt from the mosaic shapefile (vsitindex_z12.shp)
        '''

        shp_dir = os.path.join('/vsis3', self.bucket, self.mosaic, 'wms/vsitindex_z12.shp')
        cmd = 'env GDAL_DISABLE_READDIR_ON_OPEN=YES VSI_CACHE=TRUE gdalbuildvrt mosaic.vrt ' + shp_dir
        logging.info('VRT command: ' + cmd)

        subprocess.call(cmd, shell=True)

        if os.path.isfile('mosaic.vrt'):
            self.vrt_file = 'mosaic.vrt'
            logging.info('VRT created, class variable set')
            return 'mosaic.vrt'

        else:
            logging.debug('VRT not created, check S3 vars and VRT command')
            raise Exception('VRT could not be created. Make sure AWS credentials are accurate and vsitindex_z12.shp is in the project/wms/ location')


    def get_gdal_translate_cmds(self, vrt_file):
        '''
        Generate commands for extracting each chip
        '''
        gdal_cmds = []

        # get features to extract
        with open(self.geojson) as f:
            feature_collection = geojson.load(f)['features']
            logging.info('{} features loaded'.format(str(len(feature_collection))))

        for feat in feature_collection:
            # get bounding box of input polygon
            geom = feat['geometry']['coordinates'][0]
            f_id = feat['properties']['feature_id']
            xs, ys = [i[0] for i in geom], [i[1] for i in geom]
            ulx, lrx, uly, lry = min(xs), max(xs), max(ys), min(ys)

            # format gdal_translate command
            out_loc = os.path.join(self.out_dir, str(f_id) + '.tif')
            cmd = 'gdal_translate -projwin {0} {1} {2} {3} {4} {5}'.format(str(ulx), str(uly), str(lrx), str(lry), vrt_file, out_loc)
            gdal_cmds.append(cmd)
            logging.info(cmd)

        return gdal_cmds, feature_collection


    def generate_feature_ids(self, feature_collection):

        fid = 0
        for feat in feature_collection:
            feat['properties']['feature_id'] = fid
            fid += 1

        return feature_collection


    def invoke(self):

        # Create VRT as a pointer to mosiac tiles on S3
        vrt_file = self.create_vrt()

        # Create commands for extracting chips
        cmds, feature_collection = self.get_gdal_translate_cmds(vrt_file)

        if not feature_collection[0]['properties']['feature_id']:
            feature_collection = self.generate_feature_ids(feature_collection)

        # Execute gdal_translate commands in parallel
        p = Pool(cpu_count())
        p.map(self.execute_command, cmds)
        p.close()
        p.join()

        # Check chip size and mask
        os.chdir('/mnt/work/output/chips')

        p = Pool(cpu_count())
        part_check = partial(self.check_mask_chip, min_side_dim=self.min_side_dim,
                             max_side_dim=self.max_side_dim, mask=self.mask)
        p.map(part_check, feature_collection)
        p.close()
        p.join()


if __name__ == '__main__':

    with GetChipsFromMosaic() as task:
        task.invoke()
