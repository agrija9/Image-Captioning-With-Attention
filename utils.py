import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))

    # Preprocesses a tensor or Numpy array encoding a batch of images.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def save_InceptionV3_features(dataset, model):
    for img, path in dataset:
        # call model and get batches of features
        batch_features = model(img)
        batch_features = tf.reshape(batch_features,
                                   (batch_features.shape[0], -1,
                                   batch_features.shape[3]))

    # pickle features into numpy array
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

def load_InceptionV3_features(image_name, caption):
    # load numpy files (feature vectors InceptionV3)
    img_tensor = np.load(image_name.decode("utf-8")+".npy")
    return img_tensor, caption

def plot_loss(loss_plot, save_plot):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')

    if save_plot:
        try:
            os.makedirs("./results/")
        except FileExistsError:
            pass
        fig.savefig("./results/loss_plot.png", dpi=fig.dpi)
    else:
        plt.show()

def plot_attention(image, result, attention_plot, save_plot, index_image):
    temp_image = np.array(Image.open(image))
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)

    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))

        # if (len_result // 2) < 5:
        # ax = fig.add_subplot(10, 10, l+1) # 2, 2
        ax = fig.add_subplot((len_result//2)+1, (len_result//2)+1, l+1) # 2, 2
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    if save_plot:
        try:
            os.makedirs("./results/")
        except FileExistsError:
            pass
        fig.savefig("./results/attention_plot_image_" + str(index_image), bbox_inches='tight',dpi=800)
        plt.close()
    else:
        plt.show()
