import os, time, ast, shutil
import json, geojson, geoio
import numpy as np
from glob import glob
from pool_net import PoolNet
from mltools import geojson_tools as gt
from mltools import data_extractors as de
from gbdx_task_interface import GbdxTaskInterface

start = time.time()

class TrainCnnClassifier(GbdxTaskInterface):

    def invoke(self):

        # Get string inputs
        classes = [i.strip() for i in self.get_input_string_port('classes').split(',')]
        two_rounds = ast.literal_eval(self.get_input_string_port('two_rounds', default='True'))
        filter_geojson = ast.literal_eval(self.get_input_string_port('filter_geojson', default='True'))
        min_side_dim = int(self.get_input_string_port('min_side_dim', default='10'))
        max_side_dim = int(self.get_input_string_port('max_side_dim', default='125'))
        train_size = int(self.get_input_string_port('train_size', default='10000'))
        batch_size = int(self.get_input_string_port('batch_size', default='32'))
        nb_epoch = int(self.get_input_string_port('nb_epoch', default='35'))
        use_lowest_val_loss = ast.literal_eval(self.get_input_string_port('use_lowest_val_loss', default='True'))
        nb_epoch_2 = int(self.get_input_string_port('nb_epoch_2', default='8'))
        train_size_2 = int(self.get_input_string_port('train_size', default=int(0.5 * train_size)))
        test = ast.literal_eval(self.get_input_string_port('test', default='True'))
        test_size = int(self.get_input_string_port('test_size', default='5000'))
        lr_1 = float(self.get_input_string_port('learning_rate', default='0.001'))
        lr_2 = float(self.get_input_string_port('learning_rate_2', default='0.01'))
        bit_depth = int(self.get_input_string_port('bit_depth', default='8'))

        print '\nString args loaded. Running for {} seconds'.format(str(time.time() - start))

        # Get geojson input port
        shp_dir = self.get_input_data_port('geojson')
        shp_list = [f for f in os.listdir(shp_dir) if f[-8:] == '.geojson']

        if len(shp_list) != 1:
            raise Exception('Make sure there is exactly one geojson in image_dest s3 bucket')

        shp = os.path.join(shp_dir, shp_list[0])

        # Get image input port
        img_dir = self.get_input_data_port('images')
        os.makedirs(os.path.join(img_dir, 'models'))
        imgs = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img[-4:] == '.tif']

        # limit number of images to 5
        if len(imgs) > 5:
            raise Exception('There are too many images in the input image directory. ' \
                            'Please use a maximum of five image strips.')
        if len(imgs) == 0:
            raise Exception('No images were found in the input directory. Please ' \
                            'provide at lease one GeoTif image.')

        # determine input shape
        input_shape = [0,0,0]
        input_shape[0] = geoio.GeoImage(imgs[0]).shape[0]
        input_shape[1:] = [max_side_dim, max_side_dim]

        # Create output directories, organize file inputs
        trained_model = self.get_output_data_port('trained_model')
        model_weights = os.path.join(trained_model, 'model_weights')

        os.makedirs(trained_model)
        os.makedirs(model_weights)

        os.rename(shp, os.path.join(img_dir, 'orig_geojson.geojson'))
        shp = 'orig_geojson.geojson'
        os.chdir(img_dir)

        print '\nOutput directories created. Running for {} seconds'.format(str(time.time() - start))

        # Filter for appropriate polygon size
        if filter_geojson:
            gt.filter_polygon_size(shp, output_file = 'filtered_geojson.geojson',
                                   min_polygon_hw = min_side_dim,
                                   max_polygon_hw = max_side_dim)
            shp = 'filtered_geojson.geojson'
            print 'Polygons filtered. Running for {} seconds'.format(str(time.time() - start))

        # Check for number of remaining polygons
        with open(shp) as f:
            info = geojson.load(f)
            feats = info['features']
            poly_ct = len(feats)

        # Set aside test data
        train_test_prop = float(test_size) / poly_ct
        poly_ct -= (train_test_prop * poly_ct)

        print '\nMaking train/test data from ' + str(shp)
        gt.create_balanced_geojson(shp, output_file = 'filtered.geojson',
                                   balanced = False, train_test = train_test_prop)

        # Create balanced train data
        if two_rounds:
            # Make balanced data for round one
            gt.create_balanced_geojson('train_filtered.geojson',
                                       output_file = 'train_balanced.geojson')

            # Establish number of remaining train polygons
            with open('train_balanced.geojson') as f:
                info = geojson.load(f)['features']
                poly_ct = len(info)

            inp = 'train_balanced.geojson'

        else:
            inp = 'train_filtered.geojson'

        # Warn if not enough training data
        if poly_ct < train_size:
            raise Exception('There are only {} polygons that can be used as training ' \
                            'data, cannot train the network on {} samples. Please decrease' \
                            ' train_size or provide more polygons.'.format(str(poly_ct), str(train_size)))

        print 'Ready to create network instance and train network. Running for {}' \
              ' seconds.\n'.format(str(time.time() - start))

        # Create instance of PoolNet
        p = PoolNet(classes=classes, batch_size=batch_size, input_shape=input_shape,
                    min_chip_hw=min_side_dim, max_chip_hw=max_side_dim, learning_rate=lr_1,
                    bit_depth=bit_depth)

        # Fit network in large batches
        print 'training network...'
        if train_size < 5000:
            batches_per_epoch = 1
            chips_per_batch = train_size
        else:
            batches_per_epoch = int(train_size / 5000.)
            chips_per_batch = 5000

        hist = p.fit_from_geojson(inp, chips_per_batch=chips_per_batch, return_history=True,
                           batches_per_epoch=batches_per_epoch, nb_epoch=nb_epoch,
                           validation_split=0.1)
        print 'First round of training complete. Running for {} seconds'.format(str(time.time() - start))

        # Find lowest val_loss, load weights
        if use_lowest_val_loss:
            val_losses = [epoch['val_loss'][0] for epoch in hist]
            min_epoch = np.argmin(val_losses)
            min_loss = val_losses[min_epoch]
            print '\nAll validation losses: ' + str(val_losses)
            print 'min validation loss: ' + str(min_loss)
            min_weights = 'models/epoch' + str(min_epoch) + '_{0:.2f}.h5'.format(min_loss)
            print 'loading weights from: ' + min_weights
            p.model.load_weights(min_weights)
            print 'weights with lowest val loss now loaded'

        # Move model weights to output directory
        os.makedirs(os.path.join(model_weights, 'round_1'))
        weights = os.listdir('models/')
        for weights_file in weights:
            shutil.move('models/' + weights_file, os.path.join(model_weights, 'round_1'))

        # Retrain model
        if two_rounds:
            print 'training round 2...'
            if train_size_2 < 5000:
                batches_per_epoch = 1
                chips_per_batch = train_size_2
            else:
                batches_per_epoch = int(train_size_2 / 5000.)
                chips_per_batch = 5000

            p.fit_from_geojson('train_filtered.geojson', chips_per_batch=chips_per_batch,
                               nb_epoch=nb_epoch_2, batches_per_epoch=batches_per_epoch,
                               validation_split=0.1)
            print 'Second round of training complete. Running for {} seconds'.format(str(time.time() - start))

            # Move model weights to output dir
            os.makedirs(os.path.join(model_weights, 'round_2'))
            weights = os.listdir('models/')

            for weights_file in weights:
                shutil.move('models/' + weights_file,
                            os.path.join(model_weights, 'round_2'))

        # Test Net
        if test:
            print 'Generating test data...'
            test_gen = de.getIterData('test_filtered.geojson', batch_size=test_size,
                                      min_chip_hw=min_side_dim, max_chip_hw=max_side_dim,
                                      bit_depth=bit_depth, show_percentage=False)

            print 'predicting classes of test data...'
            x, y = test_gen.next()

            y_pred = p.model.predict_classes(x)
            ytrue = [i[1] for i in y]
            print 'getting test metrics:'
            test_size = len(y)
            print 'test size: ' + str(test_size)
            wrong, right = np.argwhere(y_pred != ytrue), np.argwhere(y_pred == ytrue)
            print '{} incorrectly classified, {} correctly classified'.format(str(len(wrong)), str(len(right)))
            fp = int(np.sum([y_pred[i] for i in wrong]))
            print 'fp: ' + str(fp)
            tp = int(np.sum([y_pred[i] for i in right]))
            print 'tp: ' + str(tp)
            fn, tn = int(len(wrong) - fp), int(len(right) - tp)

            # get accuracy metrics
            try:
                precision = float(tp) / (tp + fp)
            except (ZeroDivisionError):
                precision = 'N/A'
            try:
                recall = float(tp)/(tp + fn)
            except (ZeroDivisionError):
                recall = 'N/A'

            print 'fn, tn: ' + str(fn) + ' ' + str(tn)
            test_report = 'Test size: ' + str(test_size) + \
                          '\nFalse Positives: ' + str(fp) + \
                          '\nFalse Negatives: ' + str(fn) + \
                          '\nPrecision: ' + str(precision) + \
                          '\nRecall: ' + str(recall) + \
                          '\nAccuracy: ' + str(float(len(right))/test_size)
            print test_report

            # Record test results
            print 'writing test report file...'
            with open(os.path.join(trained_model, 'test_report.txt'), 'w') as tr:
                tr.write(test_report)
            print 'Testing complete. Running for {} seconds'.format(str(time.time() - start))

        # Save model architecture and weights to output directory
        os.chdir(trained_model)
        json_str = p.model.to_json()
        p.model.save_weights('model_weights.h5')
        with open('model_architecture.json', 'w') as arch:
            json.dump(json_str, arch)

        print 'Total run time: ' + str(time.time() - start)



if __name__ == '__main__':
    with TrainCnnClassifier() as task:
        task.invoke()
