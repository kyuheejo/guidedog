import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# your model focuses on during captioning
import matplotlib.pyplot as plt

import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image

import utils
from model import CNN_Encoder, RNN_Decoder
import argparse


def evaluate(image, max_length, attention_features_shape, decoder, image_features_extract_model, encoder, tokenizer):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(utils.load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot, save_path, epoch):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(np.ceil(len_result/2), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    path_to_save = os.path.join(save_path, "examples", f"epoch{epoch}")
    os.makedirs(path_to_save, exist_ok=True)
    plt.savefig(os.path.join(path_to_save, image))

def annotation_to_list(annotations, PATH, num_data=None):

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = os.path.join(PATH, 'COCO_train2014_' +'%012d.jpg' % (val['image_id']))
        image_path_to_caption[image_path].append(caption)


    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    if num_data: 
        train_image_paths = image_paths[:num_data]
    else:
        train_image_paths = image_paths

    print(f"Number of data used: {len(train_image_paths)}")

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))
    return train_captions, img_name_vector

def extract_features(image_features_extract_model, img_name_vector):
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
    utils.load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(64)

    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

@tf.function
def train_step(encoder, decoder, tokenizer, optimizer, img_tensor, target, loss_function):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
          # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
           
            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

def train_model(args):

    BATCH_SIZE = args.batch_size
    BUFFER_SIZE = args.buffer_size
    embedding_dim = args.embedding_dim
    units = args.units 

    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = args.features_shape
    attention_features_shape = args.attention_features_shape

    annotation_folder = args.annotation_folder
    annotation_file = os.path.join(annotation_folder, "captions_train2014.json")
    image_folder = args.image_folder
    PATH = os.path.join(image_folder, "train2014")


    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    train_captions, img_name_vector = annotation_to_list(annotations, PATH, 10 if args.mode=="debug" else None)

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    if args.extract_features: 
        extract_features(img_name_vector)

    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
    tokenizer.fit_on_texts(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    max_length = utils.calc_max_length(train_seqs)

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    print(f"Training set: {len(img_name_train)}, {len(cap_train)}, \n  Validation set: {len(img_name_val)}, {len(cap_val)}")

    vocab_size = top_k + 1
    num_steps = len(img_name_train) // BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            utils.map_func, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    checkpoint_path = args.save_path
    ckpt = tf.train.Checkpoint(encoder=encoder,
                                decoder=decoder,
                                optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)  
    
    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []

    EPOCHS = args.epoch

    print("Start training...")

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(encoder, decoder, tokenizer, optimizer, img_tensor, target, loss_function)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy()/int(target.shape[1])
                print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        # captions on the validation set

        if epoch % 5 == 0:
            print("Start saving model")
            ckpt_manager.save()
            # save model
            os.makedirs(os.path.join(args.save_path, "models"), exist_ok=True)
            encoder.save_weights(os.path.join(args.save_path, "models", "decoder"))
            decoder.save_weights(os.path.join(args.save_path, "models", "decoder"))
            print('Model Saved!')
    
        print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

    print("end training...")

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.savefig(os.path.join(args.save_path, "loss.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Captioning Using Tensorflow')

    # arguments related to model training 
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--buffer_size', default=1000, type=int)
    parser.add_argument('--embedding_dim', default=256, type=int)
    parser.add_argument('--units', default=512, type=int)
    parser.add_argument('--features_shape', default=2048, type=int)
    parser.add_argument('--attention_features_shape', default=64, type=int)
    parser.add_argument('--epoch', default=20, type=int)

    # arguments related to data path
    parser.add_argument('--annotation_folder', default="/home-3/kjo3@jhu.edu/kyuhee/coco/annotations/", type=str)
    parser.add_argument('--image_folder', default="/home-3/kjo3@jhu.edu/kyuhee/coco/images", type=str)

    # arguments related to training 
    parser.add_argument('--extract_features', default=False, type=utils.bool_flag) 
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--save_path', default="./checkpoints", type=str)

    parser.add_argument('--mode', default="train", type=str, help="either train or debug")
    args = parser.parse_args()

    train_model(args)

   

    








