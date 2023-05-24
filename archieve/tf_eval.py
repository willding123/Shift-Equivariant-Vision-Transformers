#%%
import numpy as np
import tensorflow_addons as tfa
from vit_keras import vit

# Load the ViT model checkpoint
model = vit.vit_b16(
    image_size=224,
    pretrained=False,
    pretrained_top=False, 
    include_top=True,
    classes=1000
)

# Load pre-trained weights
weights_path = ('~/scratch.cmsc663/sam_ViT-B_16.npz')
model.load_weights(weights_path)
#%% 
import tensorflow_datasets as tfds

# Load the ImageNet test set
ds_test, ds_info = tfds.load(
    'imagenet2012',
    split='validation',
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
    batch_size=512,
    data_dir='~/scratch.cmsc663/'
)

# Preprocess the ImageNet test set
def preprocess_image(image, label):
    # Resize the image to the input size of the ViT model
    image = tf.image.resize(image, (224, 224))
    # Normalize the image pixels
    image = (image / 255.0 - 0.5) * 2.0
    return image, label

ds_test = ds_test.map(preprocess_image)

# Evaluate the model on the ImageNet test set
loss, accuracy = model.evaluate(ds_test)

print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
