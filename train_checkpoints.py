import os
import sys
import torch
import backend
import models
import train

# Set utf-8 encoding for stdout
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Tat do hoa khi huan luyen
backend.use_graphics = False

os.makedirs("checkpoints", exist_ok=True)

print("=== 1. Huan luyen mo hinh MLP (DigitClassificationModel) ===")
mlp_model = models.DigitClassificationModel()
mlp_dataset = backend.DigitClassificationDataset(mlp_model)
train.train_digitclassifier(mlp_model, mlp_dataset)
mlp_val_acc = mlp_dataset.get_validation_accuracy()
print(f"MLP Validation Accuracy: {mlp_val_acc:.2%}")

mlp_path = os.path.join("checkpoints", "mlp_mnist.pt")
torch.save(mlp_model.state_dict(), mlp_path)
print(f"-> Da luu trong so MLP tai: {mlp_path}")

print("\n=== 2. Huan luyen mo hinh CNN (DigitConvolutionalModel) ===")
cnn_model = models.DigitConvolutionalModel()
cnn_dataset = backend.DigitClassificationDataset2(cnn_model)
train.Train_DigitConvolution(cnn_model, cnn_dataset)

# Danh gia do chinh xac cua CNN tren tap test
test_logits = cnn_model(torch.tensor(cnn_dataset.test_images)).data
test_pred = torch.argmax(test_logits, dim=1).numpy()
cnn_acc = (test_pred == cnn_dataset.test_labels).mean()
print(f"CNN Test Accuracy: {cnn_acc:.2%}")

cnn_path = os.path.join("checkpoints", "cnn_mnist.pt")
torch.save(cnn_model.state_dict(), cnn_path)
print(f"-> Da luu trong so CNN tai: {cnn_path}")

print("\nHoan tat luu checkpoints thanh cong!")
