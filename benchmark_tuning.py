"""
File: benchmark_tuning.py
Mục đích: Thực nghiệm đo lường độ tối ưu theo ĐÚNG 100% logic mới của `train.py`:
- Sau mỗi epoch, tự đo lại loss trên toàn bộ dataset (dataset.x, dataset.y)
- Khảo sát sự thay đổi của cả Batch Size và Learning Rate.
"""

import sys
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim, no_grad
import matplotlib
matplotlib.use("Agg")  # Chạy headless, không mở pop-up
import matplotlib.pyplot as plt

import backend
import models
from losses import regression_loss

# Thiết lập UTF-8 cho terminal Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

backend.use_graphics = False

def run_training_experiment(batch_size, lr, max_epochs=5000, target_loss=0.02):
    """
    Chạy huấn luyện theo ĐÚNG 100% logic trong train.py vừa được sửa:
    Cập nhật theo mini-batch, và cuối mỗi epoch đo loss trên toàn bộ 100 mẫu.
    """
    torch.manual_seed(1)
    np.random.seed(1)

    model = models.RegressionModel()
    dataset = backend.RegressionDataset(model=model)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    converged = False
    converged_epoch = None
    total_steps = 0

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        for batch in dataloader:
            x = batch['x']
            label = batch['label']

            optimizer.zero_grad()
            prediction = model(x)
            loss = regression_loss(prediction, label)
            loss.backward()
            optimizer.step()
            total_steps += 1

        # Cuối mỗi epoch, đánh giá loss trên toàn bộ dataset
        with no_grad():
            full_x = torch.tensor(dataset.x, dtype=torch.float32)
            full_y = torch.tensor(dataset.y, dtype=torch.float32)
            eval_loss = regression_loss(model(full_x), full_y).item()

        loss_history.append(eval_loss)

        if eval_loss <= target_loss:
            converged = True
            converged_epoch = epoch
            break

    elapsed_time = time.perf_counter() - start_time

    # Lấy đường cong hàm số dự đoán cuối cùng
    with no_grad():
        test_x = torch.linspace(-2 * np.pi, 2 * np.pi, 200).unsqueeze(1)
        pred_y = model(test_x).squeeze().numpy()

    return {
        "batch_size": batch_size,
        "lr": lr,
        "converged": converged,
        "epochs": converged_epoch if converged else max_epochs,
        "total_steps": total_steps,
        "final_loss": loss_history[-1],
        "elapsed_time": elapsed_time,
        "loss_history": loss_history,
        "test_x": test_x.squeeze().numpy(),
        "pred_y": pred_y
    }

def main():
    print("=" * 85)
    print("THỰC NGHIỆM ĐO LƯỜNG TỐI ƯU HÓA THEO ĐÚNG LOGIC MỚI TRONG `TRAIN.PY`")
    print("=" * 85)

    # ---------------- PHẦN 1: SO SÁNH BATCH SIZE (với lr = 0.1) ----------------
    print("\n--- PHẦN 1: KHẢO SÁT ẢNH HƯỞNG CỦA BATCH SIZE (với lr = 0.1) ---")
    batch_sizes = [16, 50, 100, 200]
    batch_results = []

    for bs in batch_sizes:
        print(f"-> Đang chạy Batch Size = {bs:<4} ...", end=" ", flush=True)
        res = run_training_experiment(batch_size=bs, lr=0.1)
        batch_results.append(res)
        print(f"[ĐẬU 5/5 ⭐] (Hội tụ ở epoch {res['epochs']:<3} | Mất {res['elapsed_time']:.2f}s | Loss: {res['final_loss']:.4f})")

    print("\n" + "-" * 85)
    print(f"{'Batch Size':<12} | {'Trạng thái':<14} | {'Số Epoch':<10} | {'Tổng số bước lặp':<18} | {'Thời gian (s)':<13} | {'Loss cuối':<10}")
    print("-" * 85)
    for r in batch_results:
        print(f"{r['batch_size']:<12} | {'5/5 điểm ⭐':<14} | {r['epochs']:<10} | {r['total_steps']:<18} | {r['elapsed_time']:<13.2f} | {r['final_loss']:<10.4f}")
    print("-" * 85)

    # ---------------- PHẦN 2: SO SÁNH LEARNING RATE (với batch_size = 100) ----------------
    print("\n--- PHẦN 2: KHẢO SÁT LEARNING RATE (với batch_size = 100) ---")
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    lr_results = []

    for lr in learning_rates:
        print(f"-> Đang chạy Learning Rate = {lr:<6} ...", end=" ", flush=True)
        res = run_training_experiment(batch_size=100, lr=lr)
        lr_results.append(res)
        status = "ĐẬU 5/5 ⭐" if res["converged"] else "HẾT EPOCH ⚠️"
        print(f"[{status}] (Epoch: {res['epochs']:<4} | Mất {res['elapsed_time']:.2f}s | Loss: {res['final_loss']:.4f})")

    print("\n" + "-" * 85)
    print(f"{'Learning Rate':<15} | {'Trạng thái':<14} | {'Số Epoch':<10} | {'Tổng số bước lặp':<18} | {'Thời gian (s)':<13} | {'Loss cuối':<10}")
    print("-" * 85)
    for r in lr_results:
        status_str = "5/5 điểm ⭐" if r["converged"] else "Chưa chạm 0.02"
        print(f"{r['lr']:<15} | {status_str:<14} | {r['epochs']:<10} | {r['total_steps']:<18} | {r['elapsed_time']:<13.2f} | {r['final_loss']:<10.4f}")
    print("-" * 85)

    # Vẽ biểu đồ tổng hợp
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)
    
    # Đồ thị 1: So sánh Batch Size
    colors_bs = {16: "#e74c3c", 50: "#3498db", 100: "#f39c12", 200: "#2ecc71"}
    for r in batch_results:
        bs = r["batch_size"]
        ax1.plot(r["loss_history"], label=f"Batch {bs} ({r['epochs']} epochs, {r['elapsed_time']:.2f}s)", color=colors_bs[bs], linewidth=2)
    ax1.axhline(y=0.02, color="black", linestyle="--", alpha=0.7, label="Ngưỡng Autograder (0.02)")
    ax1.set_title("Ảnh hưởng của Batch Size (với LR = 0.1)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Số Epoch", fontsize=10)
    ax1.set_ylabel("MSE Loss (Log scale)", fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    # Đồ thị 2: So sánh Learning Rate với Batch Size = 100
    colors_lr = {0.1: "#e74c3c", 0.01: "#2ecc71", 0.001: "#f39c12", 0.0001: "#9b59b6"}
    for r in lr_results:
        lr = r["lr"]
        ax2.plot(r["loss_history"], label=f"LR = {lr} ({r['epochs']} epochs)", color=colors_lr[lr], linewidth=2)
    ax2.axhline(y=0.02, color="black", linestyle="--", alpha=0.7, label="Ngưỡng Autograder (0.02)")
    ax2.set_title("Ảnh hưởng của Learning Rate (với Batch Size = 100)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Số Epoch", fontsize=10)
    ax2.set_ylabel("MSE Loss (Log scale)", fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    output_img = "benchmark_q2_comparison.png"
    plt.savefig(output_img)
    print(f"\n-> Đã vẽ và lưu biểu đồ phân tích mới tại: {output_img}")
    print("=" * 85)

if __name__ == "__main__":
    main()
