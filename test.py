from face_model_training.datasets.face_window_dataset import RandomFaceWindowDataset
from face_model_training.models.combined import Combined
from face_model_training.training import train


data = RandomFaceWindowDataset(epoch_size=10, augment_scale=0)
model = Combined()
train_data, test_data = data.train_test_split(0.6)
try:
    train(model=model, train_dataset=train_data, val_dataset=test_data, epochs=100)
    breakpoint()
except KeyboardInterrupt:
    breakpoint()