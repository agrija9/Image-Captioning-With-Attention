import tensorflow as tf # 2.0.0
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time

from coco import COCO
from train import train_step
from evaluate import evaluate
from model import BahdanauAttention, CNN_Encoder, RNN_Decoder
from loss import loss_function
from utils import (load_image, save_InceptionV3_features,
                   calc_max_length, load_InceptionV3_features,
                   plot_loss, plot_attention)

EPOCHS = 10
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
UNITS = 512

def main(COCO):
    parser = argparse.ArgumentParser(description="Show, attend and tell")
    parser.add_argument('--annotations_dir', type=str, default="annotations/", help='annotations directory')
    parser.add_argument('--images_dir', type=str, default="train2014/")
    parser.add_argument('--cache_inception_features', type=bool, default=False)

    args = parser.parse_args()

    # folders to load/save COCO data (annotations and images)
    annotation_folder = args.annotations_dir
    image_folder = args.images_dir

    # flag to download InceptionV3 features
    cache_inception_features = args.cache_inception_features

    print("----------------------------")
    print("STARTED IMAGE CAPTIONING PIPELINE")
    print("----------------------------")

    print()
    print("EPOCHS", EPOCHS)
    print("BATCH SIZE:", BATCH_SIZE)

    print()
    print("Available files in dir", os.listdir())

    # print()
    # print("Using GPU card:", tf.config.list_physical_devices('GPU'))

    print()
    print("Image and annotation folders")
    print(image_folder)
    print(annotation_folder)
    print("Cache image features")
    print(cache_inception_features)

    COCO_annotations = COCO.download_annotations()
    with open(COCO_annotations, "r") as f:
        annotations = json.load(f)

    # this returns path to images
    COCO_PATH = COCO.download_images()
    print()
    # print("COCO PATH:", COCO_PATH)

    # limit size of training (30,000 images)
    # store captions and image names
    all_captions = []
    all_img_names = []

    for annotation in annotations["annotations"]:
        # generate captions using start/end of sentence
        caption = "<start>" + annotation["caption"] + "<end>"
        image_id = annotation["image_id"]
        full_coco_image_path = COCO_PATH + "COCO_train2014_" + "%012d.jpg" % (image_id)

        all_img_names.append(full_coco_image_path)
        all_captions.append(caption)

    # shuffle images/captions and set random state
    train_captions, train_img_names = shuffle(all_captions,
                                       all_img_names,
                                       random_state=1)

    # print("**********************************")
    # print("Resulting array with all captions ({})".format(len(all_captions)))
    # all_captions

    # select first n captions from shuffled dataset
    # num_examples = 60000
    num_examples = 30000
    train_captions = train_captions[:num_examples]
    train_img_names = train_img_names[:num_examples]

    print()
    print("num examples (total images):", num_examples)

    # print("********************************")
    # print("Size of downsampled training set")
    # print("Captions: {}".format(len(train_captions)))

    # Process images using InceptionV3 model
    print()
    print("[INFO] Instantiating InceptionV3 model")
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")

    print()
    print("[INFO] CNN network information")
    print("inputs of model:", image_model.input)
    print("shape of last hidden layer:", image_model.layers[-1].output)

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # save feature vectors obtained with InceptionV3
    encode_train = sorted(set(train_img_names)) # get unique images (paths)
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(10)

    if cache_inception_features:
        print()
        print("[INFO] Caching InceptionV3 features")
        # this fetches 25950 features for the given initial 30,000
        for img, path in image_dataset:
            print("extracting image feature:", path)
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())
    else:
        print()
        print("[INFO] Fetching computed features from PC")

    # pre-process and tokenize captions
    # tokenize by space, limit vocabulary up to 5000 words, create word-to-index and index-to-word mapping
    # pad all sequences
    # Choose the top 5000 words from the vocabulary
    print()
    print("[INFO] Creating keras tokenizer")
    # top_k = 5000
    top_k = 20000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # # compare train_seqs[0:6] and train_captions[0:6]
    #
    # pad each vector to the max length of captions
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=calc_max_length(train_seqs), padding="post")

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # Create training and validation sets using an 80-20 split
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(train_img_names,
                                                                        cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=0)

    # create tensor flow dataset for training
    vocab_size = top_k + 1
    num_steps = len(img_name_train) // BATCH_SIZE
    print("Vocabulary size:", vocab_size)
    print("Num steps:", num_steps)

    # Vector from InceptionV3 is (64, 2048)
    # these two variables represent that vector shape
    attention_features_shape = 64
    features_shape = 2048

    print()
    print("[INFO] Creating tf dataset")
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # use load method to load the numpy files in parallel
    # this loads image and caption
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
                          load_InceptionV3_features, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle data
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # define encoder / decoder
    encoder = CNN_Encoder(EMBEDDING_DIM)
    decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, vocab_size)

    # define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam()
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # create checkpoint to resume/load model
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    print()
    print("[INFO] Started training model")

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []

    for epoch in range(start_epoch, EPOCHS):
        print()
        print("Epoch:", epoch)
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target,
                                            encoder, decoder,
                                            tokenizer, optimizer)

            # print("batch", batch)
            # print("batch_loss", batch_loss)
            # print("epoch loss", t_loss)

            total_loss += t_loss

            if batch % 1 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                  epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # save loss plot
    plot_loss(loss_plot, save_plot=True)

    print()
    print("[INFO] Testing Caption generation")

    # generate captions on the validation set
    for i in range(30):
        # generate random id
        rid = np.random.randint(0, len(img_name_val))

        # fetch corresponding image and caption from id
        image = img_name_val[rid]
        real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])

        # evaluate image (result is the caption)
        result, attention_plot = evaluate(image, max_length, attention_features_shape, encoder, decoder,
                                          load_image, image_features_extract_model, tokenizer)

        print()
        print("----------------------------")
        print("Image id:", i)
        print ("Real Caption:", real_caption)
        print ("Predicted Caption:", " ".join(result))
        plot_attention(image, result, attention_plot, save_plot=True, index_image=i)

    print("---------------")
    print("|FINISHED MAIN|")
    print("---------------")

if __name__ == "__main__":
    main(COCO = COCO("/annotations/", '/train2014/'))
