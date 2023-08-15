# Import necessary packages
import tensorflow as tf
from tflite_model_maker.config import QuantizationConfig, ExportFormat
from tflite_model_maker import model_spec, object_detector
from tflite_model_maker.object_detector import DataLoader
from tflite_model_maker.object_detector import EfficientDetSpec
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train, train_lib
from absl import logging

# Ensure TensorFlow version 2.x
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

# Setup variables
batch_size = 64
epochs = 250
checkpoint_dir = "/path/checkpoints"

# Create EfficientDetLite0Spec object
spec = object_detector.EfficientDetLite0Spec(
    model_name='efficientdet-lite0',
    uri='https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1',
    hparams='',
    model_dir=checkpoint_dir,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_execution=1,
    moving_average_decay=0,
    var_freeze_expr='(efficientnet|fpn_cells|resample_p6)',
    tflite_max_detections=25,
    strategy=None,
    tpu=None,
    gcp_project=None,
    tpu_zone=None,
    use_xla=False,
    profile=False,
    debug=False,
    tf_random_seed=111111,
    verbose=1
)

# Load datasets
train_data = object_detector.DataLoader.from_pascal_voc('/your_path/images/train/data', '/your_path/images/train/labels', label_map={1:"car", 2:"person", 3:"bike", 4:"bus", 5:"other vehicle", 6:"motor", 7:"truck"})
validation_data = object_detector.DataLoader.from_pascal_voc('/your_path/images/validation/data', '/your_path/images/validation/labels', label_map={1:"car", 2:"person", 3:"bike", 4:"bus", 5:"other vehicle", 6:"motor", 7:"truck"})
test_data = object_detector.DataLoader.from_pascal_voc('/your_path/images/test/data', '/your_path/images/test/labels', label_map={1:"car", 2:"person", 3:"bike", 4:"bus", 5:"other vehicle", 6:"motor", 7:"truck"})

# Create the object detector
detector = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=batch_size,
    train_whole_model=True,
    validation_data=validation_data,
    epochs=epochs,
    do_train=False
)

# Convert datasets for training
train_ds, steps_per_epoch, _ = detector._get_dataset_and_steps(train_data, batch_size, is_training=True)
validation_ds, validation_steps, val_json_file = detector._get_dataset_and_steps(validation_data, batch_size, is_training=False)

# Get the internal Keras model
model = detector.create_model()

# Setup
config = spec.config
config.update(
    dict(
        steps_per_epoch=steps_per_epoch,
        eval_samples=batch_size * validation_steps,
        val_json_file=val_json_file,
        batch_size=batch_size
    )
)
train.setup_model(model, config)
model.summary()

# Restore the weights
try:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        completed_epochs = int(latest.split("/")[-1].split("-")[1])
        model.load_weights(latest)
        print("Checkpoint found {}".format(latest))
    else:
        completed_epochs = 0
        print("No checkpoints found.")
except Exception as e:
    print("Error loading checkpoint: ", e)
    completed_epochs = 0

# Get callbacks
all_callbacks = train_lib.get_callbacks(config.as_dict(), validation_ds)

# Train the model
model.fit(
    train_ds,
    epochs=epochs,
    initial_epoch=completed_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,
    validation_steps=validation_steps,
    callbacks=all_callbacks
)

# Export the trained model
export_dir = "/content"
quant_config = None
detector.model = model
detector.export(export_dir=export_dir, tflite_filename='model.tflite', quantization_config=quant_config)

# Evaluate the TFLite model
tflite_filepath = "/path/model.tflite"
train_results = detector.evaluate_tflite(tflite_filepath, train_data)
validation_results = detector.evaluate_tflite(tflite_filepath, validation_data)
test_results = detector.evaluate_tflite(tflite_filepath, test_data)

print('Train data:')
for metric, value in train_results.items():
    print(f"{metric}: {value}")

print('Validation data:')
for metric, value in validation_results.items():
    print(f"{metric}: {value}")

print('Test data:')
for metric, value in test_results.items():
    print(f"{metric}: {value}")

if __name__ == "__main__":
    # Insert the calls to the main parts of the script here.
    pass

