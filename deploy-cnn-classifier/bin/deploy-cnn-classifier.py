import os, time, ast
import json, geojson, geoio
import numpy as np
from glob import glob
from shutil import copyfile, move
from pool_net import PoolNet
from mltools import geojson_tools as gt
from mltools import data_extractors as de
from gbdx_task_interface import GbdxTaskInterface

start = time.time()

class DeployCnnClassifier(GbdxTaskInterface):

    def invoke(self):

        # Get sting inputs
        classes = self.get_input_string_port('classes', default=None)

        if classes:
            classes = [clss.strip() for clss in classes.split(',')]

        bit_depth = int(self.get_input_string_port('bit_depth', default='8'))
        min_side_dim = int(self.get_input_string_port('min_side_dim', default='0'))

        # Get geojson input
        shp_dir = self.get_input_data_port('geojson')
        shp_list = [f for f in os.listdir(shp_dir) if f[-8:] == '.geojson']

        if len(shp_list) != 1:
            raise Exception('Make sure there is exactly one geojson in image_dest s3 bucket')

        shp = os.path.join(shp_dir, shp_list[0])

        # Count number of training polygons
        with open(shp) as f:
            info = geojson.load(f)['features']
            poly_ct = len(info)

        # Get model input
        model_inp = self.get_input_data_port('model')
        arch_file = [f for f in os.listdir(model_inp) if f[-5:] == '.json']
        weight_file = [f for f in os.listdir(model_inp) if f[-3:] == '.h5']

        if len(arch_file) != 1 or len(weight_file) != 1:
            raise Exception('Make sure there is exactly one json and h5 files in the model directory')

        arch = os.path.join(model_inp, arch_file[0])
        weights = os.path.join(model_inp, weight_file[0])

        collect_input = time.time() - start
        print 'Took {} seconds to collect input directories'.format(str(collect_input))

        # Make output directories
        output_dir = self.get_output_data_port('classified_geojson')
        os.makedirs(output_dir)

        # Organize inputs
        images_dir = self.get_input_data_port('images')
        copyfile(shp, os.path.join(images_dir, 'shp.geojson'))
        copyfile(arch, os.path.join(images_dir, 'model.json'))
        copyfile(weights, os.path.join(images_dir, 'model.h5'))
        os.chdir(images_dir)
        arch, weights, shp = 'model.json', 'model.h5', 'shp.geojson'

        imgs = glob('*.tif')
        if len(imgs) == 0:
            raise Exception('No imagery found in input directory. Pladsfease make sure the image directory has at least one tif image')

        make_dirs = time.time() - collect_input
        print 'Took {} seconds to make output directories'.format(str(make_dirs))

        # Load trained model
        if classes:
            p = PoolNet(classes=classes, old_model=True, model_name='model',
                        bit_depth=bit_depth)
        else:
            p = PoolNet(old_model=True, model_name='model', bit_depth=bit_depth)

        p.model.load_weights(weights)
        load_architecture = time.time() - start
        print 'Architecture loaded, running for {} seconds'.format(str(load_architecture))

        # Check that image matches input shape
        max_side_dim = p.model.input_shape[-1]
        bands = p.model.input_shape[1]
        for img in imgs:
            img_bands = geoio.GeoImage(img).shape[0]
            if bands != img_bands:
                print 'Model input shape and img shape do not match for one or more input images.'
                raise Exception('Make sure the model was trained on an image with the same number of bands as all input images.')

        # Filter shapefile
        gt.filter_polygon_size(shp, shp, min_polygon_hw=0, max_polygon_hw=max_side_dim)

        # Classify file
        out_name = 'classified.geojson'
        if classes:
            p.classify_shapefile(shp, numerical_classes=False, output_name=out_name)
        else:
            p.classify_shapefile(shp, output_name=out_name)

        classify_file = time.time() - start
        print 'Took {} seconds to classify the file'.format(str(classify_file))

        # Write output
        move(out_name, output_dir)

        total_time = time.time() - start
        print 'Total run time: {} seconds'.format(str(total_time))



if __name__ == '__main__':
    with DeployCnnClassifier() as task:
        task.invoke()
