# gbdx.Task(get-chips-from-mosaic, geojson, bucket_name,  mosaic_location, aws_access_key, aws_secret_key)

## ASSUMPTIONS ##
# mosaic data strux will be consistent
# mosaic saved as bucket_name/mosaic_location
# appropriate vrt shapefile located at .../wms/vsitindex_z12.shp

import logging
import geojson
import subprocess, os
from multiprocessing import Pool, cpu_count
from gbdx_task_interface import GbdxTaskInterface

# log file for debugging
logging.basicConfig(filename='out.log',level=logging.DEBUG)

def execute_command(cmd):
    '''
    Execute a command. This is outside of the class because you cannot use
        multiprocessing.Pool on a class methods.
    '''
    subprocess.call(cmd, shell=True)


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

        return gdal_cmds

    def invoke(self):

        # Create VRT as a pointer to mosiac tiles on S3
        vrt_file = self.create_vrt()

        # Create commands for extracting chips
        cmds = self.get_gdal_translate_cmds(vrt_file)

        # Execute commands in parallel
        p = Pool(cpu_count())
        p.map(self.execute_command, cmds)
        p.close()
        p.join()



if __name__ == '__main__':

    with GetChipsFromMosaic() as task:
        task.invoke()
