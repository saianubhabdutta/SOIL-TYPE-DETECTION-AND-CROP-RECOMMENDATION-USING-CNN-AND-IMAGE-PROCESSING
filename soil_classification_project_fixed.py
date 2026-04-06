import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from skimage.feature import local_binary_pattern
from tensorflow.keras.layers import Layer

# ✅ Crop model import
from crop_model.predict_crop import predict_from_soil

# ==============================
# Configuration
# ==============================

DATASET_DIR = "Orignal-Dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25                  # ✅ MODIFIED: kept same
FINE_TUNE_EPOCHS = 20        # ✅ MODIFIED: increased from 5 → 20
FINE_TUNE_LAYERS = 50        # ✅ MODIFIED: unfreeze last 50 layers (was 10)
NUM_CLASSES = None

# ==============================
# Image Preprocessing
# ==============================

def preprocess_image_for_resnet(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.bilateralFilter(img, 9, 75, 75)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    return rgb


def extract_lbp_features(img_gray):
    radius = 3
    n_points = 8 * radius

    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    lbp = lbp.astype('float32')

    maxv = lbp.max() if lbp.size else 0.0

    if maxv > 0:
        lbp = (lbp / maxv) * 255.0
    else:
        lbp = lbp * 0.0

    return lbp


# ==============================
# Load Dataset
# ==============================

print("Loading dataset...")

X = []
y = []

classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

for label in classes:

    class_dir = os.path.join(DATASET_DIR, label)

    for fname in os.listdir(class_dir):

        img_path = os.path.join(class_dir, fname)

        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = preprocess_image_for_resnet(img)

        gray = cv2.cvtColor(
            cv2.resize(img, (IMG_SIZE, IMG_SIZE)),
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        lbp = extract_lbp_features(gray)

        lbp = lbp[..., np.newaxis]

        rgb_pre = preprocess_input(rgb)

        combined = np.dstack([rgb_pre, lbp]).astype('float32')

        X.append(combined)
        y.append(label)


X = np.array(X, dtype='float32')

print("Loaded", len(X), "images with shape", X.shape)

# ==============================
# Label Encoding
# ==============================

le = LabelEncoder()

y_encoded = le.fit_transform(y)

NUM_CLASSES = len(le.classes_)

y_cat = to_categorical(y_encoded, num_classes=NUM_CLASSES)

# ==============================
# Train / Validation Split
# ==============================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Dataset split:", X_train.shape[0], "train,", X_val.shape[0], "validation")

# ==============================
# ✅ MODIFIED: Class Weights (handles imbalanced dataset)
# ==============================

y_train_labels = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# ==============================
# ✅ MODIFIED: Stronger Data Augmentation
# ==============================

datagen = ImageDataGenerator(
    rotation_range=40,          # ✅ increased from 15 → 40
    width_shift_range=0.2,      # ✅ increased from 0.1 → 0.2
    height_shift_range=0.2,     # ✅ increased from 0.1 → 0.2
    horizontal_flip=True,
    vertical_flip=True,         # ✅ NEW
    zoom_range=0.3,             # ✅ NEW
    shear_range=0.2,            # ✅ NEW
    brightness_range=[0.7, 1.3],# ✅ widened from [0.8,1.2]
    fill_mode='reflect'
)

datagen.fit(X_train)

# ==============================
# Build Model
# ==============================

base_input = Input(shape=(IMG_SIZE, IMG_SIZE, 4), name='input_4ch')

class ChannelSplit(Layer):
    def call(self, inputs):
        rgb = inputs[..., :3]
        lbp = inputs[..., 3:]
        return [rgb, lbp]

rgb_tensor, lbp_tensor = ChannelSplit()(base_input)

resnet_base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=rgb_tensor
)

for layer in resnet_base.layers:
    layer.trainable = False

x = GlobalAveragePooling2D(name='gap')(resnet_base.output)
lbp_pool = GlobalAveragePooling2D(name='lbp_gap')(lbp_tensor)

x = tf.keras.layers.Concatenate(name='concat')([x, lbp_pool])

# ✅ MODIFIED: Deeper head with BatchNormalization
x = Dense(256, activation='relu', name='dense_256')(x)      # ✅ NEW: added 256 layer
x = BatchNormalization(name='bn_256')(x)                     # ✅ NEW
x = Dropout(0.4, name='dropout_1')(x)                       # ✅ MODIFIED: 0.5 → 0.4

x = Dense(128, activation='relu', name='dense_128')(x)
x = BatchNormalization(name='bn_128')(x)                     # ✅ NEW
x = Dropout(0.3, name='dropout_2')(x)                       # ✅ NEW: second dropout

output = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model = Model(inputs=base_input, outputs=output, name='soil_resnet_lbp')

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# ✅ MODIFIED: Callbacks (added ReduceLROnPlateau)
# ==============================

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    patience=7,                  # ✅ MODIFIED: increased patience 5 → 7
    restore_best_weights=True
)

# ✅ NEW: Reduce LR when val_loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ==============================
# Training
# ==============================

print("Starting training...")

model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch= len(X_train) // BATCH_SIZE+1,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    class_weight=class_weight_dict,     # ✅ NEW: class weights applied
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# ==============================
# ✅ MODIFIED: Fine-tuning (50 layers, 20 epochs, lower LR)
# ==============================

print(f"Fine-tuning top {FINE_TUNE_LAYERS} layers...")

for layer in resnet_base.layers[-FINE_TUNE_LAYERS:]:   # ✅ 50 layers (was 10)
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),                # ✅ same low LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ NEW: Fresh callbacks for fine-tuning phase
ft_checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

ft_earlystop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

ft_reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1
)

model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch= len(X_train) // BATCH_SIZE+1,
    validation_data=(X_val, y_val),
    epochs=FINE_TUNE_EPOCHS,                           # ✅ 20 epochs (was 5)
    class_weight=class_weight_dict,                    # ✅ NEW
    callbacks=[ft_checkpoint, ft_earlystop, ft_reduce_lr]
)

model.save("best_model.keras")

print("Training finished. Model saved as best_model.keras")

# ==============================
# CONFUSION MATRIX
# ==============================

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Generating confusion matrix...")

y_val_true = np.argmax(y_val, axis=1)
y_val_pred = np.argmax(model.predict(X_val), axis=1)

cm = confusion_matrix(y_val_true, y_val_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix saved as confusion_matrix.png")

# ==============================
# Prediction Function
# ==============================

def predict_soil_type(model, img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Image not found")

    rgb = preprocess_image_for_resnet(img)

    gray = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    lbp = extract_lbp_features(gray)
    lbp = lbp[..., np.newaxis].astype('float32')

    rgb_pre = preprocess_input(rgb)

    input_img = np.dstack([rgb_pre, lbp]).astype('float32')
    input_img = np.expand_dims(input_img, axis=0)

    pred = model.predict(input_img)
    class_idx = np.argmax(pred, axis=1)[0]

    soil_type = le.inverse_transform([class_idx])[0]

    return soil_type


# ==============================
# FINAL TEST
# ==============================

if __name__ == "__main__":

    model_path = "best_model.keras"
    test_image = "test_image.jpg"

    if os.path.exists(model_path):

        print("\nLoading trained model...")

        best_model = load_model(
            model_path,
            custom_objects={"ChannelSplit": ChannelSplit},
            compile=False
        )

        if os.path.exists(test_image):

            print("Running prediction on test image...")

            soil = predict_soil_type(best_model, test_image)

            # ✅ ML Crop Prediction
            crops = predict_from_soil(soil)

            print("\n==============================")
            print("Predicted Soil:", soil)
            print("Top 3 Recommended Crops:", ", ".join(crops))
            print("==============================")

        else:
            print("Test image not found")

    else:
        print("Model not found. Train first.")