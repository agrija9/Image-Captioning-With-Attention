import tensorflow as tf
import os
import json


class COCO(object):
    def __init__(self, annotations_folder, image_folder):
        print("[INFO] Instantiated COCO")
        self.annotations_folder = annotations_folder
        self.image_folder = image_folder

    def download_annotations(self):
        """
        Downloads caption annotation files from cocodataset.org
        """

        if not os.path.exists(os.path.abspath('.') + self.annotations_folder):
            print("[INFO] Downloading COCO annotations")
            annotation_zip = tf.keras.utils.get_file('captions.zip', cache_subdir=os.path.abspath('.'),
                                                     origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                     extract = True)

            annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
            os.remove(annotation_zip)

            return annotation_file

        else:
            print("[INFO] Annotations folder exists. Fetching annotations from PC.")
            annotation_file = 'annotations/captions_train2014.json'
            return annotation_file

    def download_images(self):
        """
        Downloads image files from cocodataset.org
        NOTE: this downloads 13 GB of data.
        """

        if not os.path.exists(os.path.abspath('.') + self.image_folder):
            print("[INFO] Downloading COCO images (13 GB)")
            image_zip = tf.keras.utils.get_file('train2014.zip', cache_subdir=os.path.abspath('.'),
                                                origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                                extract = True)

            PATH = os.path.dirname(image_zip) + self.image_folder
            os.remove(image_zip)

            return PATH

        else:
            print("[INFO] Train images folder exists. Fetching images from PC.")
            PATH = os.path.abspath('.') + self.image_folder
            return PATH
