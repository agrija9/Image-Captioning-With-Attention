import numpy as np
import tensorflow as tf

"""The evaluate function is similar to the training loop,
except you don't use teacher forcing here.
The input to the decoder at each time step is its previous predictions
along with the hidden state and the encoder output.
Stop predicting when the model predicts the end token.
And store the attention weights for every time step."""

def evaluate(image, max_length, attention_features_shape, encoder, decoder,
             load_image, image_features_extract_model, tokenizer):
    attention_plot = np.zeros((max_length, attention_features_shape))

    # reset to zeros decoder initial hidden state
    hidden = decoder.reset_state(batch_size=1)

    # compute image features
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    # pass into CNN encoder image features
    features = encoder(img_tensor_val)

    # generate input to decoder (<start>)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    # compute predictions with decoder
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

        if predicted_id in tokenizer.index_word:
            result.append(tokenizer.index_word[predicted_id])
        else:
            result.append("unk")

        # end predicting if we reach <end> token
        if predicted_id in tokenizer.index_word and tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]

    return result, attention_plot
