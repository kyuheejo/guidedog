import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# your model focuses on during captioning
from . import model
import pickle
from . import utils
import argparse


def main(image):  # path to the image
    # hyperparameters 
    embedding_dim = 256
    units = 512
    vocab_size = 5001
    max_length = 51
    units = 512

    # First, prepare model 
    # define image feature extraction  model 
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # define encoder and decoder model 
    encoder = model.CNN_Encoder(embedding_dim)
    decoder = model.RNN_Decoder(embedding_dim, units, vocab_size)
    encoder.load_weights("transfer/models/encoder")
    decoder.load_weights("transfer/models/decoder")

    # load tokenizer
    with open("transfer/tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)

    hidden = tf.zeros((1, units))

    # load image 
    temp_input = tf.expand_dims(utils.load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            print('Prediction Caption:', ' '.join(result))
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    print('Prediction Caption:', ' '.join(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Captioning Using Tensorflow')

    # arguments related to model training 
    parser.add_argument('--image_path', default='.', type=str, help='image to be analyzed')
    args = parser.parse_args()

    main(args.image_path)
