"""
File: benchmark_q3.py
Mục đích: Thực nghiệm đo lường tối ưu hóa siêu tham số Q3 (MNIST MLP) theo ĐÚNG 100% logic mới của `train.py`:
- Sau mỗi epoch đo validation accuracy bằng `dataset.get_validation_accuracy()`
- Dừng ngay khi đạt ngưỡng an toàn: `if val_accuracy >= 0.975: break`
- In tiến trình từng epoch để trực tiếp quan sát mô hình học
- Đánh giá trên tập Test độc lập (đúng chuẩn Autograder)
"""

import sys
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim, no_grad
import matplotlib
matplotlib.use("Agg")  # Chạy headless
import matplotlib.pyplot as plt

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import backend
import models
from losses import digitclassifier_loss

backend.use_graphics = False

def run_experiment(batch_size, lr, max_epochs=10, target_val_acc=0.975, verbose=True):
    """
    Huấn luyện theo đúng chuẩn 100% logic train_digitclassifier:
    Duyệt mini-batch, cuối mỗi epoch đo Validation và break khi chạm ngưỡng 97.5%.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    model = models.DigitClassificationModel()
    dataset = backend.DigitClassificationDataset(model=model)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_acc_history = []
    loss_history = []
    converged = False
    converged_epoch = None

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in dataloader:
            x = batch['x']
            label = batch['label']

            # 5 bước huấn luyện chuẩn
            optimizer.zero_grad()
            prediction = model(x)
            loss = digitclassifier_loss(prediction, label)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)

        # Đánh giá sau mỗi epoch trên tập Validation
        model.eval()
        with no_grad():
            val_accuracy = dataset.get_validation_accuracy()
        val_acc_history.append(val_accuracy)

        if verbose:
            print(f"      [Epoch {epoch:2d}/{max_epochs}] Loss: {avg_loss:.4f} | Val Acc: {val_accuracy*100:6.2f}%", flush=True)

        # ĐÚNG LOGIC CỦA BẠN: Ngưỡng an toàn 97.5% để dừng sớm
        if val_accuracy >= target_val_acc:
            converged = True
            converged_epoch = epoch
            if verbose:
                print(f"      -> [ĐÃ HỘI TỤ] Vượt ngưỡng an toàn {target_val_acc*100:.1f}% tại Epoch {epoch}! Dừng huấn luyện.", flush=True)
            break

    elapsed_time = time.perf_counter() - start_time

    # Đo độ chính xác trên tập Test độc lập 
    model.eval()
    with no_grad():
        test_logits = model(torch.tensor(dataset.test_images, dtype=torch.float32)).data
        test_predicted = np.argmax(test_logits, axis=1).detach().numpy()
        test_acc = float(np.mean(test_predicted == dataset.test_labels))

    return {
        "batch_size": batch_size,
        "lr": lr,
        "converged": converged,
        "epochs": converged_epoch if converged else max_epochs,
        "final_val_acc": val_acc_history[-1],
        "final_test_acc": test_acc,
        "elapsed_time": elapsed_time,
        "val_acc_history": val_acc_history,
        "loss_history": loss_history
    }

def main():
    print("=" * 85)
    print("THỰC NGHIỆM ĐO LƯỜNG TỐI ƯU HÓA Q3 (THEO ĐÚNG LOGIC DỪNG AN TOÀN >= 97.5%)")
    print("=" * 85)

    # ---------------- PHẦN 1: SO SÁNH LEARNING RATE (với batch_size = 100) ----------------
    print("\n--- PHẦN 1: KHẢO SÁT ẢNH HƯỞNG CỦA LEARNING RATE (Batch Size = 100) ---")
    learning_rates = [0.01, 0.001, 0.0001]
    lr_results = []

    for lr in learning_rates:
        print(f"\n-> Bắt đầu thử nghiệm với Learning Rate = {lr}:", flush=True)
        res = run_experiment(batch_size=100, lr=lr, max_epochs=20, target_val_acc=0.975)
        lr_results.append(res)
        pass_test = "5/5 ĐIỂM ⭐" if res["final_test_acc"] >= 0.97 else "DƯỚI 97% ⚠️"
        print(f"   => KẾT QUẢ: [{pass_test}] | Test Acc: {res['final_test_acc']*100:.2f}% | Tổng thời gian: {res['elapsed_time']:.1f}s")

    print("\n" + "-" * 85)
    print(f"{'Learning Rate':<15} | {'Trạng thái':<15} | {'Số Epoch':<10} | {'Val Acc (%)':<13} | {'Test Acc (%)':<13} | {'Thời gian (s)':<12}")
    print("-" * 85)
    for r in lr_results:
        status_str = "5/5 điểm ⭐" if r["final_test_acc"] >= 0.97 else "Chưa đạt 97%"
        print(f"{r['lr']:<15} | {status_str:<15} | {r['epochs']:<10} | {r['final_val_acc']*100:<13.2f} | {r['final_test_acc']*100:<13.2f} | {r['elapsed_time']:<12.1f}")
    print("-" * 85)

    # ---------------- PHẦN 2: SO SÁNH BATCH SIZE (với lr = 0.001) ----------------
    print("\n--- PHẦN 2: KHẢO SÁT ẢNH HƯỞNG CỦA BATCH SIZE (với LR = 0.001) ---")
    batch_sizes = [50, 100, 200]
    bs_results = []

    for bs in batch_sizes:
        print(f"\n-> Bắt đầu thử nghiệm với Batch Size = {bs}:", flush=True)
        res = run_experiment(batch_size=bs, lr=0.001, max_epochs=20, target_val_acc=0.975)
        bs_results.append(res)
        pass_test = "5/5 ĐIỂM ⭐" if res["final_test_acc"] >= 0.97 else "DƯỚI 97% ⚠️"
        print(f"   => KẾT QUẢ: [{pass_test}] | Test Acc: {res['final_test_acc']*100:.2f}% | Tổng thời gian: {res['elapsed_time']:.1f}s")

    print("\n" + "-" * 85)
    print(f"{'Batch Size':<15} | {'Trạng thái':<15} | {'Số Epoch':<10} | {'Val Acc (%)':<13} | {'Test Acc (%)':<13} | {'Thời gian (s)':<12}")
    print("-" * 85)
    for r in bs_results:
        status_str = "5/5 điểm ⭐" if r["final_test_acc"] >= 0.97 else "Chưa đạt 97%"
        print(f"{r['batch_size']:<15} | {status_str:<15} | {r['epochs']:<10} | {r['final_val_acc']*100:<13.2f} | {r['final_test_acc']*100:<13.2f} | {r['elapsed_time']:<12.1f}")
    print("-" * 85)

    # ---------------- PHẦN 3: VẼ BIỂU ĐỒ ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    # Đồ thị 1: Learning Rate
    colors_lr = {0.01: "#e74c3c", 0.001: "#2ecc71", 0.0001: "#3498db"}
    for r in lr_results:
        lr = r["lr"]
        epochs = list(range(1, len(r["val_acc_history"]) + 1))
        acc_pct = [acc * 100 for acc in r["val_acc_history"]]
        ax1.plot(epochs, acc_pct, marker='o', label=f"LR = {lr} ({r['epochs']} epochs, Test: {r['final_test_acc']*100:.2f}%)", color=colors_lr[lr], linewidth=2)
    ax1.axhline(y=97.5, color="red", linestyle=":", alpha=0.9, label="Ngưỡng dừng an toàn (97.5%)")
    ax1.axhline(y=97.0, color="black", linestyle="--", alpha=0.7, label="Ngưỡng Autograder (97.0%)")
    ax1.set_title("Hội tụ Validation theo Learning Rate (Batch = 100)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Số Epoch", fontsize=10)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax1.set_ylim(85, 99)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9, loc="lower right")

    # Đồ thị 2: Batch Size
    colors_bs = {50: "#9b59b6", 100: "#2ecc71", 200: "#f39c12"}
    for r in bs_results:
        bs = r["batch_size"]
        epochs = list(range(1, len(r["val_acc_history"]) + 1))
        acc_pct = [acc * 100 for acc in r["val_acc_history"]]
        ax2.plot(epochs, acc_pct, marker='s', label=f"Batch {bs} ({r['epochs']} epochs, Test: {r['final_test_acc']*100:.2f}%)", color=colors_bs[bs], linewidth=2)
    ax2.axhline(y=97.5, color="red", linestyle=":", alpha=0.9, label="Ngưỡng dừng an toàn (97.5%)")
    ax2.axhline(y=97.0, color="black", linestyle="--", alpha=0.7, label="Ngưỡng Autograder (97.0%)")
    ax2.set_title("Hội tụ Validation theo Batch Size (LR = 0.001)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Số Epoch", fontsize=10)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax2.set_ylim(85, 99)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    output_img = "benchmark_q3_comparison.png"
    plt.savefig(output_img)
    print(f"\n-> Đã vẽ và cập nhật biểu đồ phân tích mới tại: {output_img}")
    print("=" * 85)

if __name__ == "__main__":
    main()
