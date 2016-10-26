FROM nvidia/cuda:7.5-cudnn4-devel

RUN apt-get update && apt-get install -y\
    git\
    vim\
    python \
    ipython\
    build-essential\
    python-software-properties\
    software-properties-common\
    python-pip\
    python-scipy\
    python-dev\
    gdal-bin\
    python-gdal\
    libgdal-dev


# install ml dependencies
RUN pip install gdal numpy ephem psycopg2 h5py theano keras
RUN pip install git+https://github.com/DigitalGlobe/mltools

ADD ./bin /
COPY .theanorc /root/.theanorc
COPY keras.json /root/.keras/keras.json
