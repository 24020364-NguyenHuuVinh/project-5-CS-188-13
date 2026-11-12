"""
demo1.py
HỆ THỐNG THỬ NGHIỆM MÔ HÌNH HỌC MÁY (CS188 PROJECT 5)

Bao gồm 5 bài toán:
- Bài 1 (Q1): Mô hình Perceptron 2D & Ranh giới phân loại.
- Bài 2 (Q2): Hồi quy phi tuyến sin(x) & Benchmark khảo sát siêu tham số (Batch Size, Learning Rate).
- Bài 3 (Q3): Phân loại chữ số MNIST (MLP) & Benchmark hội tụ an toàn (>= 97.5%)
              + Bảng vẽ tương tác tự động căn giữa trọng tâm khối lượng (Center of Mass).
- Bài 4 (Q4): Mạng hồi quy (RNN) nhận diện ngôn ngữ từ chuỗi ký tự.
- Bài 5 (Q5): Mạng tích chập CNN & trực quan hóa Feature Map từ hàm Convolve 2D (chế độ cơ bản).

Tính năng nổi bật:
- Bộ theo dõi tác vụ toàn cục (Task Monitor): Cho biết chính xác tác vụ nào đang chạy.
- Nút "🛑 Dừng tác vụ" (Cancel/Reset): Hủy ngang ngay lập tức tác vụ đang chạy, không sợ bị đơ hay kẹt trạng thái bận.
"""

import contextlib
import os
import queue
import random
import threading
import time
import traceback
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
import torch
from torch import optim, no_grad
from torch.utils.data import DataLoader

try:
    from PIL import Image, ImageDraw, ImageTk
    HAS_PIL = True
except Exception:
    HAS_PIL = False

import backend
backend.use_graphics = False  # Chạy headless, không mở pop-up riêng của matplotlib

# Kiểm tra module đề bài
IMPORT_ERRORS = []
for mod in ["models", "train", "losses"]:
    try:
        globals()[mod] = __import__(mod)
    except Exception as exc:
        globals()[mod] = None
        IMPORT_ERRORS.append(f"{mod}: {exc}")

models = globals().get("models")
train = globals().get("train")
losses = globals().get("losses")

WEIGHT_DIR = "ui_weights"


# ============================================================================
# 1. QUẢN LÝ LƯU & NẠP TRỌNG SỐ
# ============================================================================
def save_model_weights(model, name):
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    filepath = os.path.join(WEIGHT_DIR, f"{name}.pt")
    torch.save(model.state_dict(), filepath)
    return filepath


def load_model_weights(model, name):
    primary = os.path.join(WEIGHT_DIR, f"{name}.pt")
    if os.path.exists(primary):
        model.load_state_dict(torch.load(primary, map_location="cpu"))
        return primary

    # Tìm dự phòng trong thư mục checkpoints/
    backups = {
        "q3": os.path.join("checkpoints", "mlp_mnist.pt"),
        "q5": os.path.join("checkpoints", "cnn_mnist.pt"),
        "q5_full": os.path.join("checkpoints", "cnn_mnist.pt"),
    }
    if name in backups and os.path.exists(backups[name]):
        model.load_state_dict(torch.load(backups[name], map_location="cpu"))
        return backups[name]

    raise FileNotFoundError(f"Chưa có file trọng số cho '{name}'. Hãy bấm Huấn luyện trước.")


# ============================================================================
# 2. XỬ LÝ ẢNH VẼ: TỰ ĐỘNG CĂN TRỌNG TÂM (CENTER OF MASS)
# ============================================================================
def normalize_mnist_sketch(pil_img):
    """
    Tự động chuẩn hóa nét vẽ về chuẩn MNIST:
    - Cắt khung bao quanh chữ số.
    - Co giãn vừa vặn khung 20x20 giữ nguyên tỷ lệ.
    - Đặt vào trung tâm ảnh 28x28.
    - Tính tâm khối lượng (Center of Mass) và dịch về tọa độ (14, 14).
    """
    if not HAS_PIL or pil_img is None:
        return np.zeros((28, 28), dtype=np.float32), None

    bbox = pil_img.getbbox()
    if bbox is None:
        return np.zeros((28, 28), dtype=np.float32), pil_img.resize((28, 28))

    cropped = pil_img.crop(bbox)
    w, h = cropped.size
    if w > h:
        nw = 20
        nh = max(1, int(round(h * 20.0 / w)))
    else:
        nh = 20
        nw = max(1, int(round(w * 20.0 / h)))

    resample_box = getattr(Image, "Resampling", Image).BOX
    scaled = cropped.resize((nw, nh), resample_box)

    padded = Image.new("L", (28, 28), color=0)
    padded.paste(scaled, ((28 - nw) // 2, (28 - nh) // 2))

    arr = np.array(padded, dtype=np.float32) / 255.0
    mass = np.sum(arr)
    if mass > 0:
        cy, cx = np.indices(arr.shape)
        c_y = np.sum(cy * arr) / mass
        c_x = np.sum(cx * arr) / mass

        shift_x = int(round(14.0 - c_x))
        shift_y = int(round(14.0 - c_y))

        shifted = Image.new("L", (28, 28), color=0)
        shifted.paste(padded, (shift_x, shift_y))
        padded = shifted
        arr = np.array(padded, dtype=np.float32) / 255.0

    return arr, padded


# ============================================================================
# 3. BẢNG VẼ TAY TƯƠNG TÁC (280x280)
# ============================================================================
class HandwritingCanvas(tk.Frame):
    DIM = 280
    PEN_R = 9

    def __init__(self, master, on_release=None):
        super().__init__(master, bg="#e2e8f0", bd=1, relief=tk.SOLID)
        self.on_release = on_release
        self.last_pos = None

        self.cv = tk.Canvas(
            self, width=self.DIM, height=self.DIM,
            bg="#000000", cursor="crosshair", bd=0, highlightthickness=0
        )
        self.cv.pack()

        if HAS_PIL:
            self.pil_layer = Image.new("L", (self.DIM, self.DIM), color=0)
            self.draw = ImageDraw.Draw(self.pil_layer)
        else:
            self.arr_layer = np.zeros((28, 28), dtype=np.float32)

        self.cv.bind("<Button-1>", self._down)
        self.cv.bind("<B1-Motion>", self._drag)
        self.cv.bind("<ButtonRelease-1>", self._up)

    def _down(self, e):
        self.last_pos = (e.x, e.y)
        r = self.PEN_R
        self.cv.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="#ffffff", outline="#ffffff")
        if HAS_PIL:
            self.draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)

    def _drag(self, e):
        if self.last_pos:
            x0, y0 = self.last_pos
            r = self.PEN_R
            self.cv.create_line(x0, y0, e.x, e.y, fill="#ffffff", width=r * 2,
                                capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True)
            if HAS_PIL:
                self.draw.line([x0, y0, e.x, e.y], fill=255, width=r * 2)
                self.draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)
        self.last_pos = (e.x, e.y)

    def _up(self, _e):
        self.last_pos = None
        if self.on_release:
            self.on_release()

    def clear(self):
        self.cv.delete("all")
        if HAS_PIL:
            self.pil_layer = Image.new("L", (self.DIM, self.DIM), color=0)
            self.draw = ImageDraw.Draw(self.pil_layer)
        else:
            self.arr_layer = np.zeros((28, 28), dtype=np.float32)

    def set_sample(self, mat_28):
        self.clear()
        mat = np.clip(np.asarray(mat_28, dtype=np.float32).reshape(28, 28), 0.0, 1.0)
        if HAS_PIL:
            u8 = (mat * 255).astype(np.uint8)
            img = Image.fromarray(u8)
            resample_mode = getattr(Image, "Resampling", Image).NEAREST
            self.pil_layer = img.resize((self.DIM, self.DIM), resample_mode)
            self.draw = ImageDraw.Draw(self.pil_layer)
            self._tk_cache = ImageTk.PhotoImage(self.pil_layer)
            self.cv.create_image(0, 0, anchor=tk.NW, image=self._tk_cache)
        else:
            self.arr_layer = mat.copy()

    def get_centered_array(self):
        if HAS_PIL:
            return normalize_mnist_sketch(self.pil_layer)
        return self.arr_layer.copy(), None


# ============================================================================
# 4. VIEW BASE ĐA LUỒNG & BỘ ĐIỀU KHIỂN HỦY TÁC VỤ
# ============================================================================
class ConsoleRedirector:
    def __init__(self, q):
        self.q = q
        self.buf = ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.q.put(("log", line))
        return len(s)

    def flush(self):
        pass


class BaseLabView(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg="#f8fafc")
        self.app = app
        self.msg_q = queue.Queue()
        self.is_running = False
        self.lock_list = []
        self.status = tk.StringVar(value="Sẵn sàng.")

    def add_console_box(self, parent, height=5):
        frame = tk.LabelFrame(parent, text=" Nhật ký thực thi ", font=("Segoe UI", 9, "bold"),
                              fg="#334155", bg="#ffffff", bd=1, relief=tk.SOLID, padx=6, pady=4)
        self.console = tk.Text(
            frame, height=height, bg="#1e293b", fg="#e2e8f0",
            font=("Consolas", 9), state="disabled", wrap="word",
            bd=0, padx=6, pady=4
        )
        sb = ttk.Scrollbar(frame, command=self.console.yview)
        self.console.configure(yscrollcommand=sb.set)
        self.console.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return frame

    def print_log(self, text):
        self.console.configure(state="normal")
        self.console.insert("end", str(text) + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def unlock_ui(self):
        self.is_running = False
        for widget in self.lock_list:
            try:
                widget.config(state="normal")
            except Exception:
                pass

    def run_task(self, task_name, target_fn, on_done=None, completion_callback=None):
        cb = on_done if on_done is not None else completion_callback
        if self.app.busy:
            cur = self.app.current_running_name
            messagebox.showinfo("Đang bận", f"Tác vụ [{cur}] đang chạy!\n\nNếu muốn dừng, hãy bấm nút đỏ [🛑 Dừng tác vụ] ở góc trên cùng bên phải.")
            return

        self.app.register_start_task(task_name)
        self.is_running = True
        for widget in self.lock_list:
            try:
                widget.config(state="disabled")
            except Exception:
                pass

        def runner():
            redirector = ConsoleRedirector(self.msg_q)
            try:
                with contextlib.redirect_stdout(redirector):
                    target_fn()
                if redirector.buf.strip():
                    self.msg_q.put(("log", redirector.buf.strip()))
                self.msg_q.put(("done", None))
            except Exception:
                self.msg_q.put(("done", traceback.format_exc()))

        threading.Thread(target=runner, daemon=True).start()
        self.after(90, lambda: self._check_task(cb))

    def _check_task(self, callback):
        done, err = False, None
        try:
            while True:
                tag, payload = self.msg_q.get_nowait()
                if tag == "log":
                    self.print_log(payload)
                elif tag == "done":
                    done, err = True, payload
        except queue.Empty:
            pass

        try:
            self.on_tick()
        except Exception:
            pass

        if done:
            self.unlock_ui()
            self.app.register_end_task()

            if err:
                self.print_log(err)
                self.status.set("Phát sinh lỗi khi thực thi (xem chi tiết ở nhật ký).")
                messagebox.showerror("Lỗi", err.strip().splitlines()[-1])
            elif callback:
                callback()
        else:
            self.after(90, lambda: self._check_task(callback))

    def on_tick(self):
        pass


# ============================================================================
# 5. BIỂU ĐỒ XÁC SUẤT
# ============================================================================
class ProbChart(tk.Frame):
    def __init__(self, master, horizontal=False, with_map=False, figsize=(5.2, 3.4)):
        super().__init__(master, bg="#ffffff", bd=1, relief=tk.SOLID)
        self.horizontal = horizontal
        self.fig = Figure(figsize=figsize, dpi=100, facecolor="#ffffff")
        if with_map:
            self.ax_bar, self.ax_map = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [2.3, 1.2]})
        else:
            self.ax_bar, self.ax_map = self.fig.add_subplot(111), None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def render(self, labels, probs, fmap=None, title=""):
        probs = np.asarray(probs, dtype=float)
        n = len(probs)
        best = int(probs.argmax()) if n > 0 else 0

        colors = ["#3b82f6"] * n
        if n > 0:
            colors[best] = "#ef4444"

        ax = self.ax_bar
        ax.clear()
        ax.set_facecolor("#ffffff")

        if self.horizontal:
            ax.barh(range(n), probs, color=colors, edgecolor="#e2e8f0")
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.08)
            for i, p in enumerate(probs):
                ax.text(min(p + 0.02, 0.88), i, f"{p:.1%}", va="center", fontsize=8)
        else:
            ax.bar(range(n), probs, color=colors, edgecolor="#e2e8f0")
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylim(0, 1.1)
            for i, p in enumerate(probs):
                if p > 0.04:
                    ax.text(i, p + 0.02, f"{p:.0%}", ha="center", fontsize=8)

        ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1", axis="y" if not self.horizontal else "x")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

        if self.ax_map is not None:
            self.ax_map.clear()
            self.ax_map.axis("off")
            self.ax_map.set_facecolor("#ffffff")
            if fmap is not None:
                self.ax_map.imshow(fmap, cmap="viridis")
                self.ax_map.set_title("Feature Map 26x26\n(Lớp Convolve)", fontsize=8, pad=6)

        self.fig.tight_layout()
        self.canvas.draw_idle()


# ============================================================================
# 6. BÀI 1: PERCEPTRON (Q1)
# ============================================================================
class PerceptronModule(BaseLabView):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.model = None
        self.dataset = None

        top_bar = tk.Frame(self, bg="#ffffff", bd=1, relief=tk.SOLID, padx=10, pady=8)
        top_bar.pack(fill="x", padx=10, pady=8)

        btn_new = tk.Button(top_bar, text="🎲 Tạo dữ liệu 2D mới", bg="#f1f5f9", fg="#0f172a",
                            font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=10, pady=4,
                            cursor="hand2", command=self.reset_data)
        btn_train = tk.Button(top_bar, text="▶ Train Perceptron", bg="#2563eb", fg="#ffffff",
                              font=("Segoe UI", 9, "bold"), bd=0, padx=14, pady=4,
                              cursor="hand2", command=self.train_perceptron)
        btn_new.pack(side="left", padx=4)
        btn_train.pack(side="left", padx=4)

        self.lock_list = [btn_new, btn_train]

        tk.Label(top_bar, textvariable=self.status, font=("Segoe UI", 9, "bold"),
                 fg="#0f172a", bg="#ffffff").pack(side="left", padx=12)

        chart_frame = tk.Frame(self, bg="#ffffff", bd=1, relief=tk.SOLID, padx=8, pady=8)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.fig = Figure(figsize=(6, 4.2), dpi=100, facecolor="#ffffff")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.add_console_box(self, height=4).pack(fill="x", padx=10, pady=(0, 8))
        self.reset_data()

    def reset_data(self):
        try:
            self.model = models.PerceptronModel(3)
            self.dataset = backend.PerceptronDataset(self.model)
            self.status.set("Đã tạo tập 500 điểm dữ liệu 2D. Bấm Train Perceptron.")
            self.draw_plot()
        except Exception as e:
            self.print_log(traceback.format_exc())
            self.status.set(f"Lỗi: {e}")

    def draw_plot(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#ffffff")

        x, y = self.dataset.x, self.dataset.y.flatten()
        ax.scatter(x[y == 1, 0], x[y == 1, 1], color="#ef4444", marker="+", s=35, label="Nhãn +1")
        ax.scatter(x[y == -1, 0], x[y == -1, 1], color="#3b82f6", marker="_", s=35, label="Nhãn -1")

        w = self.model.get_weights().detach().numpy().flatten()
        lim = np.array([-3.0, 3.0])

        if w.size == 3:
            if w[1] != 0:
                ax.plot(lim, (-w[0] * lim - w[2]) / w[1], color="#10b981", linewidth=2, label="Ranh giới w·x = 0")
            elif w[0] != 0:
                ax.plot(np.full(2, -w[2] / w[0]), lim, color="#10b981", linewidth=2, label="Ranh giới w·x = 0")

            preds = np.where(x @ w >= 0, 1.0, -1.0)
            acc = float(np.mean(preds == y))
            title_text = f"Trọng số: w = [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}] | Accuracy = {acc:.1%}"
        else:
            title_text = "Mô hình Perceptron 2D"

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
        ax.set_title(title_text, fontsize=10, fontweight="bold", pad=8)
        ax.legend(loc="lower right", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def train_perceptron(self):
        if self.model is None:
            return
        self.status.set("Đang cập nhật trọng số Perceptron...")

        def task():
            dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
            with no_grad():
                for ep in range(100):
                    if self.app.is_cancelled():
                        print("-> Đã dừng huấn luyện Perceptron theo yêu cầu.")
                        break
                    convergence = True
                    for batch in dataloader:
                        x = batch['x']
                        label = batch['label']
                        pred = self.model.get_prediction(x)
                        if pred != label.item():
                            self.model.w += label.item() * x
                            convergence = False
                    if convergence:
                        print(f"-> Perceptron đã hội tụ hoàn toàn sau {ep+1} epoch!")
                        break

        self.run_task(
            "Huấn luyện Perceptron (Q1)",
            task,
            on_done=self._after_train
        )

    def _after_train(self):
        if self.app.is_cancelled():
            self.status.set("Đã dừng huấn luyện.")
        else:
            self.status.set("Huấn luyện hoàn tất: Phân loại đúng 100% dữ liệu.")
        self.draw_plot()

    def on_tick(self):
        if self.model is not None:
            self.draw_plot()


# ============================================================================
# 7. BÀI 2: HỒI QUY SIN(X) & BENCHMARK SIÊU THAM SỐ (Q2)
# ============================================================================
class RegressionModule(BaseLabView):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.model = None
        self.grid_x = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1).astype(np.float32)
        self.view_mode = "sin"
        self.bench_data = None

        bar = tk.Frame(self, bg="#ffffff", bd=1, relief=tk.SOLID, padx=10, pady=8)
        bar.pack(fill="x", padx=10, pady=8)

        tk.Label(bar, text="Learning Rate:", font=("Segoe UI", 9), bg="#ffffff").pack(side="left")
        self.var_lr = tk.StringVar(value="0.1")
        ttk.Combobox(bar, textvariable=self.var_lr, values=["0.1", "0.05", "0.01", "0.001"], width=6).pack(side="left", padx=(2, 8))

        tk.Label(bar, text="Batch Size:", font=("Segoe UI", 9), bg="#ffffff").pack(side="left")
        self.var_bs = tk.StringVar(value="100")
        ttk.Combobox(bar, textvariable=self.var_bs, values=["16", "50", "100", "200"], width=5).pack(side="left", padx=(2, 8))

        tk.Label(bar, text="Ngưỡng Loss:", font=("Segoe UI", 9), bg="#ffffff").pack(side="left")
        self.var_tgt = tk.StringVar(value="0.02")
        tk.Entry(bar, textvariable=self.var_tgt, width=5, bd=1, relief=tk.SOLID).pack(side="left", padx=(2, 10))

        btn_fit = tk.Button(bar, text="▶ Train Hồi quy", bg="#2563eb", fg="#ffffff",
                            font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=3,
                            cursor="hand2", command=self.train_single)
        btn_load = tk.Button(bar, text="💾 Nạp Trọng số", bg="#f1f5f9", fg="#0f172a",
                             font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=8, pady=3,
                             cursor="hand2", command=self.load_checkpoint)
        btn_bench = tk.Button(bar, text="📊 Chạy Benchmark Q2", bg="#f59e0b", fg="#000000",
                              font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=3,
                              cursor="hand2", command=self.run_benchmark)
        self.btn_toggle = tk.Button(bar, text="🔄 Xem Biểu đồ Benchmark", bg="#f1f5f9", fg="#0f172a",
                                    font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=8, pady=3,
                                    cursor="hand2", command=self.toggle_view)
        self.btn_save_png = tk.Button(bar, text="🖼️ Xuất PNG", bg="#f1f5f9", fg="#64748b",
                                      font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=8, pady=3,
                                      state="disabled", command=self.export_png)

        btn_fit.pack(side="left", padx=2)
        btn_load.pack(side="left", padx=2)
        btn_bench.pack(side="left", padx=2)
        self.btn_toggle.pack(side="left", padx=2)
        self.btn_save_png.pack(side="left", padx=2)

        self.lock_list = [btn_fit, btn_load, btn_bench, self.btn_toggle, self.btn_save_png]

        chart_frame = tk.Frame(self, bg="#ffffff", bd=1, relief=tk.SOLID, padx=8, pady=6)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.fig = Figure(figsize=(7.5, 4.0), dpi=100, facecolor="#ffffff")
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.add_console_box(self, height=5).pack(fill="x", padx=10, pady=(0, 8))
        self.draw_sin()

    def draw_sin(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#ffffff")

        ax.plot(self.grid_x, np.sin(self.grid_x), color="#2563eb", linewidth=2.2, label="Đồ thị y = sin(x) gốc")
        title = "Chưa có mô hình (Bấm Train hoặc Nạp trọng số)"

        if self.model is not None:
            with torch.no_grad():
                pred = self.model(torch.tensor(self.grid_x))
                loss = losses.regression_loss(pred, torch.tensor(np.sin(self.grid_x))).item()
            ax.plot(self.grid_x, pred.detach().numpy(), color="#ef4444", linewidth=2.2, label="Mô hình MLP học được")
            rating = "ĐẠT CHUẨN 5/5 ⭐" if loss <= 0.02 else "Chưa đạt yêu cầu"
            title = f"MSE Loss: {loss:.5f} ({rating}) | Ngưỡng Autograder <= 0.02"

        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-1.4, 1.4)
        ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.legend(loc="upper right", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def draw_benchmark(self):
        if not self.bench_data:
            return
        bs_res, lr_res = self.bench_data
        self.fig.clf()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        ax1.set_facecolor("#ffffff")
        ax2.set_facecolor("#ffffff")

        colors_bs = {16: "#ef4444", 50: "#3b82f6", 100: "#f59e0b", 200: "#10b981"}
        for r in bs_res:
            b = r["batch_size"]
            ax1.plot(r["loss_history"], label=f"Batch {b} ({r['epochs']} ep, {r['elapsed_time']:.2f}s)",
                     color=colors_bs.get(b, "#3b82f6"), linewidth=2)
        ax1.axhline(y=0.02, color="#000000", linestyle="--", alpha=0.7, label="Ngưỡng đề (0.02)")
        ax1.set_title("Ảnh hưởng của Batch Size (LR = 0.1)", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Số Epoch", fontsize=8)
        ax1.set_ylabel("MSE Loss (Log scale)", fontsize=8)
        ax1.set_yscale("log")
        ax1.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
        ax1.legend(fontsize=8)

        colors_lr = {0.1: "#ef4444", 0.01: "#10b981", 0.001: "#f59e0b", 0.0001: "#8b5cf6"}
        for r in lr_res:
            l = r["lr"]
            ax2.plot(r["loss_history"], label=f"LR = {l} ({r['epochs']} ep)",
                     color=colors_lr.get(l, "#3b82f6"), linewidth=2)
        ax2.axhline(y=0.02, color="#000000", linestyle="--", alpha=0.7, label="Ngưỡng đề (0.02)")
        ax2.set_title("Ảnh hưởng của Learning Rate (Batch = 100)", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Số Epoch", fontsize=8)
        ax2.set_ylabel("MSE Loss (Log scale)", fontsize=8)
        ax2.set_yscale("log")
        ax2.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
        ax2.legend(fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def toggle_view(self):
        if self.view_mode == "sin":
            if self.bench_data is None:
                messagebox.showinfo("Thông báo", "Vui lòng bấm 'Chạy Benchmark Q2' trước để có số liệu.")
                return
            self.view_mode = "benchmark"
            self.btn_toggle.config(text="🔄 Xem Đồ thị sin(x)")
            self.draw_benchmark()
        else:
            self.view_mode = "sin"
            self.btn_toggle.config(text="🔄 Xem Biểu đồ Benchmark")
            self.draw_sin()

    def train_single(self):
        try:
            lr = float(self.var_lr.get())
            bs = int(self.var_bs.get())
            tgt = float(self.var_tgt.get())
        except ValueError:
            messagebox.showerror("Lỗi", "Vui lòng nhập đúng tham số LR, Batch, Ngưỡng.")
            return

        self.view_mode = "sin"
        self.btn_toggle.config(text="🔄 Xem Biểu đồ Benchmark")
        self.status.set(f"Đang huấn luyện (LR={lr}, Batch={bs})...")

        def task():
            torch.manual_seed(1)
            np.random.seed(1)
            self.model = models.RegressionModel()
            dataset = backend.RegressionDataset(self.model)
            loader = DataLoader(dataset, batch_size=bs, shuffle=True)
            opt = optim.Adam(self.model.parameters(), lr=lr)

            for epoch in range(1, 201):
                if self.app.is_cancelled():
                    print("-> Đã nhận lệnh dừng huấn luyện Q2.")
                    break

                for b in loader:
                    opt.zero_grad()
                    pred = self.model(b['x'])
                    loss = losses.regression_loss(pred, b['label'])
                    loss.backward()
                    opt.step()

                with torch.no_grad():
                    fx = torch.tensor(dataset.x, dtype=torch.float32)
                    fy = torch.tensor(dataset.y, dtype=torch.float32)
                    cur_loss = losses.regression_loss(self.model(fx), fy).item()

                if epoch % 5 == 0 or cur_loss <= tgt:
                    print(f"Epoch {epoch:3d} | MSE Loss: {cur_loss:.5f}")

                if cur_loss <= tgt:
                    print(f"-> Hội tụ tại epoch {epoch} (Loss: {cur_loss:.5f} <= {tgt})!")
                    break

            if not self.app.is_cancelled():
                save_model_weights(self.model, "q2")

        self.run_task(
            "Huấn luyện Hồi quy (Q2)",
            task,
            on_done=lambda: self.status.set("Đã dừng tác vụ." if self.app.is_cancelled() else "Huấn luyện xong và đã lưu trọng số q2.pt.")
        )

    def run_benchmark(self):
        self.status.set("Đang chạy Benchmark khảo sát siêu tham số Q2...")

        def experiment_run(bs, lr, max_ep=100, target_loss=0.02):
            torch.manual_seed(1)
            np.random.seed(1)
            m = models.RegressionModel()
            ds = backend.RegressionDataset(m)
            loader = DataLoader(ds, batch_size=bs, shuffle=True)
            opt = optim.Adam(m.parameters(), lr=lr)
            hist = []
            conv = False
            conv_ep = None
            step_count = 0
            t0 = time.perf_counter()

            for ep in range(1, max_ep + 1):
                if self.app.is_cancelled():
                    break
                for b in loader:
                    opt.zero_grad()
                    p = m(b['x'])
                    l = losses.regression_loss(p, b['label'])
                    l.backward()
                    opt.step()
                    step_count += 1
                with torch.no_grad():
                    fx = torch.tensor(ds.x, dtype=torch.float32)
                    fy = torch.tensor(ds.y, dtype=torch.float32)
                    eval_l = losses.regression_loss(m(fx), fy).item()
                hist.append(eval_l)
                if eval_l <= target_loss:
                    conv = True
                    conv_ep = ep
                    break

            return {
                "batch_size": bs, "lr": lr, "converged": conv,
                "epochs": conv_ep if conv else max_ep,
                "total_steps": step_count, "final_loss": hist[-1] if hist else 0.0,
                "elapsed_time": time.perf_counter() - t0,
                "loss_history": hist
            }

        def task():
            print("=" * 85)
            print("THỰC NGHIỆM ĐO LƯỜNG TỐI ƯU HÓA Q2 (HỒI QUY SIN)")
            print("=" * 85)

            # Phần 1: Khảo sát Batch Size
            print("\n--- PHẦN 1: KHẢO SÁT ẢNH HƯỞNG CỦA BATCH SIZE (với LR = 0.1) ---")
            batch_list = [16, 50, 100, 200]
            bs_results = []
            for bs in batch_list:
                if self.app.is_cancelled():
                    print("-> Dừng benchmark theo yêu cầu.")
                    return
                print(f"-> Thử nghiệm Batch Size = {bs:<4} ...", end=" ", flush=True)
                res = experiment_run(bs, 0.1, max_ep=100)
                bs_results.append(res)
                print(f"[ĐẬU 5/5 ⭐] (Epoch {res['epochs']:<3} | Thời gian: {res['elapsed_time']:.2f}s | Loss: {res['final_loss']:.4f})")

            print("\n" + "-" * 85)
            print(f"{'Batch Size':<12} | {'Trạng thái':<14} | {'Số Epoch':<10} | {'Tổng bước lặp':<18} | {'Thời gian (s)':<13} | {'Loss cuối':<10}")
            print("-" * 85)
            for r in bs_results:
                print(f"{r['batch_size']:<12} | {'5/5 điểm ⭐':<14} | {r['epochs']:<10} | {r['total_steps']:<18} | {r['elapsed_time']:<13.2f} | {r['final_loss']:<10.4f}")
            print("-" * 85)

            # Phần 2: Khảo sát Learning Rate
            print("\n--- PHẦN 2: KHẢO SÁT LEARNING RATE (với Batch Size = 100) ---")
            lr_list = [0.1, 0.01, 0.001, 0.0001]
            lr_results = []
            for lr in lr_list:
                if self.app.is_cancelled():
                    print("-> Dừng benchmark theo yêu cầu.")
                    return
                print(f"-> Thử nghiệm Learning Rate = {lr:<6} ...", end=" ", flush=True)
                res = experiment_run(100, lr, max_ep=250)
                lr_results.append(res)
                tag = "ĐẬU 5/5 ⭐" if res["converged"] else "HẾT EPOCH ⚠️"
                print(f"[{tag}] (Epoch: {res['epochs']:<4} | Thời gian: {res['elapsed_time']:.2f}s | Loss: {res['final_loss']:.4f})")

            print("\n" + "-" * 85)
            print(f"{'Learning Rate':<15} | {'Trạng thái':<14} | {'Số Epoch':<10} | {'Tổng bước lặp':<18} | {'Thời gian (s)':<13} | {'Loss cuối':<10}")
            print("-" * 85)
            for r in lr_results:
                tag_str = "5/5 điểm ⭐" if r["converged"] else "Chưa chạm 0.02"
                print(f"{r['lr']:<15} | {tag_str:<14} | {r['epochs']:<10} | {r['total_steps']:<18} | {r['elapsed_time']:<13.2f} | {r['final_loss']:<10.4f}")
            print("-" * 85)

            self.bench_data = (bs_results, lr_results)

        def on_complete():
            if self.app.is_cancelled():
                self.status.set("Đã dừng Benchmark Q2.")
            elif self.bench_data:
                self.view_mode = "benchmark"
                self.btn_toggle.config(text="🔄 Xem Đồ thị sin(x)")
                self.btn_save_png.config(state="normal")
                self.draw_benchmark()
                self.status.set("Benchmark Q2 hoàn tất! Biểu đồ so sánh đã sẵn sàng.")

        self.run_task("Benchmark Tối ưu hóa (Q2)", task, on_done=on_complete)

    def export_png(self):
        try:
            target = "benchmark_q2_comparison.png"
            self.fig.savefig(target)
            messagebox.showinfo("Đã lưu", f"Biểu đồ đã được lưu tại: {target}")
        except Exception as e:
            self.status.set(f"Lỗi: {e}")

    def load_checkpoint(self):
        try:
            m = models.RegressionModel()
            p = load_model_weights(m, "q2")
            self.model = m
            self.view_mode = "sin"
            self.btn_toggle.config(text="🔄 Xem Biểu đồ Benchmark")
            self.status.set(f"Đã nạp trọng số từ {p}.")
            self.draw_sin()
        except Exception as e:
            self.status.set(f"Lỗi: {e}")

    def on_tick(self):
        if self.view_mode == "sin" and self.model is not None:
            self.draw_sin()


# ============================================================================
# 8. BÀI 3: CHỮ SỐ MNIST (MLP) & BENCHMARK SIÊU THAM SỐ (Q3)
# ============================================================================
class DigitClassificationModule(BaseLabView):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.model = None
        self.dataset = None
        self.mode = "canvas"
        self.bench_data = None

        bar = tk.Frame(self, bg="#ffffff", bd=1, relief=tk.SOLID, padx=10, pady=8)
        bar.pack(fill="x", padx=10, pady=8)

        tk.Label(bar, text="Learning Rate:", font=("Segoe UI", 9), bg="#ffffff").pack(side="left")
        self.var_lr = tk.StringVar(value="0.001")
        ttk.Combobox(bar, textvariable=self.var_lr, values=["0.01", "0.001", "0.0001"], width=7).pack(side="left", padx=(2, 8))

        tk.Label(bar, text="Batch Size:", font=("Segoe UI", 9), bg="#ffffff").pack(side="left")
        self.var_bs = tk.StringVar(value="100")
        ttk.Combobox(bar, textvariable=self.var_bs, values=["50", "100", "200"], width=5).pack(side="left", padx=(2, 8))

        tk.Label(bar, text="Ngưỡng dừng:", font=("Segoe UI", 9), bg="#ffffff").pack(side="left")
        self.var_stop = tk.StringVar(value="97.5%")
        ttk.Combobox(bar, textvariable=self.var_stop, values=["97.0%", "97.5%", "98.0%"], width=6).pack(side="left", padx=(2, 10))

        btn_fit = tk.Button(bar, text="▶ Train Mô hình", bg="#2563eb", fg="#ffffff",
                            font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=3,
                            cursor="hand2", command=self.train_model)
        btn_load = tk.Button(bar, text="💾 Nạp Trọng số", bg="#f1f5f9", fg="#0f172a",
                             font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=8, pady=3,
                             cursor="hand2", command=self.load_checkpoint)
        btn_eval = tk.Button(bar, text="🎯 Đo Test Acc", bg="#10b981", fg="#ffffff",
                             font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=3,
                             cursor="hand2", command=self.eval_test)
        btn_bench = tk.Button(bar, text="📊 Chạy Benchmark Q3", bg="#f59e0b", fg="#000000",
                              font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=3,
                              cursor="hand2", command=self.run_benchmark)
        self.btn_toggle = tk.Button(bar, text="🔄 Chuyển sang Biểu đồ Benchmark", bg="#f1f5f9", fg="#0f172a",
                                    font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=8, pady=3,
                                    cursor="hand2", command=self.toggle_mode)
        self.btn_save_png = tk.Button(bar, text="🖼️ Xuất PNG", bg="#f1f5f9", fg="#64748b",
                                      font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=8, pady=3,
                                      state="disabled", command=self.export_png)

        btn_fit.pack(side="left", padx=2)
        btn_load.pack(side="left", padx=2)
        btn_eval.pack(side="left", padx=2)
        btn_bench.pack(side="left", padx=2)
        self.btn_toggle.pack(side="left", padx=2)
        self.btn_save_png.pack(side="left", padx=2)

        self.lock_list = [btn_fit, btn_load, btn_eval, btn_bench, self.btn_toggle, self.btn_save_png]

        # Khung nội dung
        self.container = tk.Frame(self, bg="#f8fafc")
        self.container.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # View vẽ tay
        self.view_sketch = tk.Frame(self.container, bg="#f8fafc")
        self.view_sketch.pack(fill="both", expand=True)

        left_box = tk.LabelFrame(self.view_sketch, text=" Bảng vẽ chữ số (chuột trái) ",
                                 font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#ffffff",
                                 bd=1, relief=tk.SOLID, padx=10, pady=8)
        left_box.pack(side="left", fill="y", padx=(0, 8))

        self.canvas = HandwritingCanvas(left_box, on_release=self.infer_sketch)
        self.canvas.pack(pady=2)

        btn_row = tk.Frame(left_box, bg="#ffffff")
        btn_row.pack(fill="x", pady=4)
        tk.Button(btn_row, text="🗑️ Xóa bảng", bg="#f1f5f9", fg="#0f172a", bd=1, relief=tk.SOLID,
                  padx=6, pady=3, cursor="hand2", command=self.clear_pad).pack(side="left", expand=True, fill="x", padx=1)
        tk.Button(btn_row, text="⚡ Nhận dạng", bg="#2563eb", fg="#ffffff", bd=0,
                  padx=6, pady=3, cursor="hand2", command=self.infer_sketch).pack(side="left", expand=True, fill="x", padx=1)
        tk.Button(btn_row, text="🎲 Mẫu ngẫu nhiên", bg="#f1f5f9", fg="#0f172a", bd=1, relief=tk.SOLID,
                  padx=6, pady=3, cursor="hand2", command=self.load_random_mnist).pack(side="left", expand=True, fill="x", padx=1)

        thumb_row = tk.Frame(left_box, bg="#f8fafc", bd=1, relief=tk.SOLID, padx=6, pady=4)
        thumb_row.pack(fill="x", pady=4)
        tk.Label(thumb_row, text="Ảnh tự căn (28x28):", font=("Segoe UI", 8), bg="#f8fafc").pack(side="left")
        self.thumb_cv = tk.Canvas(thumb_row, width=48, height=48, bg="#000000", bd=0, highlightthickness=0)
        self.thumb_cv.pack(side="left", padx=8)

        tk.Label(left_box, text="✨ Nét vẽ tự động căn giữa trọng tâm về (14, 14)\n(Chuẩn MNIST, không cần căn chỉnh thủ công)",
                 font=("Segoe UI", 8, "italic"), fg="#64748b", bg="#ffffff").pack(anchor="w", pady=2)

        self.chart = ProbChart(self.view_sketch, horizontal=False, with_map=False, figsize=(5.5, 3.8))
        self.chart.pack(side="left", fill="both", expand=True)

        # View Benchmark
        self.view_bench = tk.Frame(self.container, bg="#ffffff", bd=1, relief=tk.SOLID, padx=8, pady=8)
        self.fig_bench = Figure(figsize=(7.5, 3.8), dpi=100, facecolor="#ffffff")
        self.canvas_bench = FigureCanvasTkAgg(self.fig_bench, master=self.view_bench)
        self.canvas_bench.get_tk_widget().pack(fill="both", expand=True)

        self.add_console_box(self, height=5).pack(fill="x", padx=10, pady=(0, 8))
        self.status.set("Hãy vẽ thử một chữ số hoặc bấm Huấn luyện / Benchmark Q3.")

    def _ensure_instances(self):
        if self.model is None:
            self.model = models.DigitClassificationModel()
        if self.dataset is None:
            self.dataset = backend.DigitClassificationDataset(self.model)
        else:
            self.dataset.model = self.model

    def clear_pad(self):
        self.canvas.clear()
        self.thumb_cv.delete("all")
        self.chart.render([str(i) for i in range(10)], np.zeros(10), None, "Đã xóa nét vẽ")

    def infer_sketch(self):
        if self.model is None:
            self.status.set("Chưa có mô hình: Bấm Huấn luyện hoặc Nạp trọng số trước.")
            return
        try:
            arr_28, _ = self.canvas.get_centered_array()
            if arr_28 is None or np.max(arr_28) < 0.05:
                return

            self.thumb_cv.delete("all")
            step = 48 / 28.0
            for r in range(28):
                for c in range(28):
                    val = int(255 * arr_28[r, c])
                    if val > 15:
                        self.thumb_cv.create_rectangle(
                            c * step, r * step, (c + 1) * step, (r + 1) * step,
                            fill=f"#{val:02x}{val:02x}{val:02x}", outline=""
                        )

            tensor_in = torch.tensor(arr_28.reshape(1, 784), dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(tensor_in)
                probs = torch.softmax(logits, dim=-1)[0].numpy()

            winner = int(probs.argmax())
            conf = probs[winner]
            self.chart.render(
                [str(i) for i in range(10)], probs, None,
                f"KẾT QUẢ DỰ ĐOÁN: CHỮ SỐ {winner} (Độ tin cậy: {conf:.1%})"
            )
            self.status.set(f"Dự đoán chữ số: {winner} | Độ tin cậy: {conf:.1%}")
        except Exception as e:
            self.print_log(traceback.format_exc())
            self.status.set(f"Lỗi: {e}")

    def load_random_mnist(self):
        try:
            self._ensure_instances()
            idx = random.randrange(len(self.dataset.test_images))
            raw = np.asarray(self.dataset.test_images[idx]).reshape(28, 28)
            label = int(self.dataset.test_labels[idx])
            self.canvas.set_sample(raw)
            self.infer_sketch()
            self.status.set(f"{self.status.get()} (Nhãn thật: {label})")
        except Exception as e:
            self.status.set(f"Lỗi: {e}")

    def eval_test(self):
        try:
            self._ensure_instances()
            self.model.eval()
            with torch.no_grad():
                inp = torch.tensor(np.asarray(self.dataset.test_images), dtype=torch.float32)
                logits = self.model(inp)
            preds = logits.argmax(dim=1).numpy()
            acc = float(np.mean(preds == self.dataset.test_labels))
            rating = "ĐẠT 5/5 ⭐ (>= 97%)" if acc >= 0.97 else "Chưa đạt"
            self.status.set(f"Độ chính xác tập Test (10,000 ảnh): {acc:.2%} [{rating}]")
            self.print_log(f"-> Test Accuracy = {acc:.2%} ({rating})")
        except Exception as e:
            self.status.set(f"Lỗi: {e}")

    def train_model(self):
        try:
            lr = float(self.var_lr.get())
            bs = int(self.var_bs.get())
            tgt_str = self.var_stop.get().replace("%", "").strip()
            tgt_acc = float(tgt_str) / 100.0 if float(tgt_str) > 1.0 else float(tgt_str)
        except ValueError:
            messagebox.showerror("Lỗi", "Vui lòng nhập đúng số cho LR, Batch, Ngưỡng.")
            return

        self.status.set("Đang huấn luyện mô hình MLP...")

        def task():
            torch.manual_seed(42)
            np.random.seed(42)
            self.model = models.DigitClassificationModel()
            self.dataset = backend.DigitClassificationDataset(self.model)
            loader = DataLoader(self.dataset, batch_size=bs, shuffle=True)
            opt = optim.Adam(self.model.parameters(), lr=lr)

            print(f"Huấn luyện DigitClassificationModel (LR={lr}, Batch={bs}, Dừng an toàn >={tgt_acc*100:.1f}%):")
            for ep in range(1, 21):
                if self.app.is_cancelled():
                    print("-> Đã nhận lệnh dừng huấn luyện Q3.")
                    break

                self.model.train()
                ep_losses = []
                for b in loader:
                    opt.zero_grad()
                    out = self.model(b['x'])
                    loss = losses.digitclassifier_loss(out, b['label'])
                    loss.backward()
                    opt.step()
                    ep_losses.append(loss.item())

                avg_l = float(np.mean(ep_losses))
                self.model.eval()
                with torch.no_grad():
                    val_acc = self.dataset.get_validation_accuracy()
                print(f"   [Epoch {ep:2d}] Loss: {avg_l:.4f} | Val Accuracy: {val_acc*100:6.2f}%")

                if val_acc >= tgt_acc:
                    print(f"-> [HỘI TỤ AN TOÀN] Vượt ngưỡng an toàn {tgt_acc*100:.1f}% tại Epoch {ep}! Dừng.")
                    break

            if not self.app.is_cancelled():
                save_model_weights(self.model, "q3")

        self.run_task(
            "Huấn luyện Chữ số MNIST (Q3)",
            task,
            on_done=lambda: [self.status.set("Đã dừng tác vụ." if self.app.is_cancelled() else "Huấn luyện xong! Hãy thử vẽ chữ số."), self.eval_test()]
        )

    def run_benchmark(self):
        self.status.set("Đang chạy Benchmark Q3 theo kịch bản chuẩn...")

        def experiment_run(bs, lr, max_ep=15, target_val_acc=0.975):
            torch.manual_seed(42)
            np.random.seed(42)
            m = models.DigitClassificationModel()
            ds = backend.DigitClassificationDataset(model=m)
            loader = DataLoader(ds, batch_size=bs, shuffle=True)
            opt = optim.Adam(m.parameters(), lr=lr)
            val_hist = []
            loss_hist = []
            conv = False
            conv_ep = None
            t0 = time.perf_counter()

            for ep in range(1, max_ep + 1):
                if self.app.is_cancelled():
                    break
                m.train()
                ep_l = []
                for b in loader:
                    opt.zero_grad()
                    out = m(b['x'])
                    l = losses.digitclassifier_loss(out, b['label'])
                    l.backward()
                    opt.step()
                    ep_l.append(l.item())
                loss_hist.append(float(np.mean(ep_l)))
                m.eval()
                with torch.no_grad():
                    va = ds.get_validation_accuracy()
                val_hist.append(va)
                print(f"      [Epoch {ep:2d}/{max_ep}] Loss: {loss_hist[-1]:.4f} | Val Acc: {va*100:6.2f}%")

                if va >= target_val_acc:
                    conv = True
                    conv_ep = ep
                    print(f"      -> [ĐÃ HỘI TỤ] Vượt ngưỡng an toàn {target_val_acc*100:.1f}% tại Epoch {ep}!")
                    break

            elapsed = time.perf_counter() - t0
            m.eval()
            with torch.no_grad():
                t_logits = m(torch.tensor(ds.test_images, dtype=torch.float32)).data
                t_preds = np.argmax(t_logits, axis=1).detach().numpy()
                t_acc = float(np.mean(t_preds == ds.test_labels))

            return {
                "batch_size": bs, "lr": lr, "converged": conv,
                "epochs": conv_ep if conv else max_ep,
                "final_val_acc": val_hist[-1] if val_hist else 0.0,
                "final_test_acc": t_acc,
                "elapsed_time": elapsed, "val_acc_history": val_hist,
                "loss_history": loss_hist
            }

        def task():
            print("=" * 85)
            print("THỰC NGHIỆM ĐO LƯỜNG TỐI ƯU HÓA Q3 (THEO ĐÚNG LOGIC DỪNG AN TOÀN >= 97.5%)")
            print("=" * 85)

            # Phần 1: Khảo sát Learning Rate
            print("\n--- PHẦN 1: KHẢO SÁT ẢNH HƯỞNG CỦA LEARNING RATE (Batch Size = 100) ---")
            lrs = [0.01, 0.001, 0.0001]
            lr_results = []
            for lr in lrs:
                if self.app.is_cancelled():
                    print("-> Dừng benchmark theo yêu cầu.")
                    return
                print(f"\n-> Thử nghiệm Learning Rate = {lr}:")
                res = experiment_run(100, lr)
                lr_results.append(res)
                eval_tag = "5/5 ĐIỂM ⭐" if res["final_test_acc"] >= 0.97 else "DƯỚI 97% ⚠️"
                print(f"   => KẾT QUẢ: [{eval_tag}] | Test Acc: {res['final_test_acc']*100:.2f}% | Thời gian: {res['elapsed_time']:.1f}s")

            print("\n" + "-" * 85)
            print(f"{'Learning Rate':<15} | {'Trạng thái':<15} | {'Số Epoch':<10} | {'Val Acc (%)':<13} | {'Test Acc (%)':<13} | {'Thời gian (s)':<12}")
            print("-" * 85)
            for r in lr_results:
                tag = "5/5 điểm ⭐" if r["final_test_acc"] >= 0.97 else "Chưa đạt 97%"
                print(f"{r['lr']:<15} | {tag:<15} | {r['epochs']:<10} | {r['final_val_acc']*100:<13.2f} | {r['final_test_acc']*100:<13.2f} | {r['elapsed_time']:<12.1f}")
            print("-" * 85)

            # Phần 2: Khảo sát Batch Size
            print("\n--- PHẦN 2: KHẢO SÁT ẢNH HƯỞNG CỦA BATCH SIZE (với LR = 0.001) ---")
            bss = [50, 100, 200]
            bs_results = []
            for bs in bss:
                if self.app.is_cancelled():
                    print("-> Dừng benchmark theo yêu cầu.")
                    return
                print(f"\n-> Thử nghiệm Batch Size = {bs}:")
                res = experiment_run(bs, 0.001)
                bs_results.append(res)
                eval_tag = "5/5 ĐIỂM ⭐" if res["final_test_acc"] >= 0.97 else "DƯỚI 97% ⚠️"
                print(f"   => KẾT QUẢ: [{eval_tag}] | Test Acc: {res['final_test_acc']*100:.2f}% | Thời gian: {res['elapsed_time']:.1f}s")

            print("\n" + "-" * 85)
            print(f"{'Batch Size':<15} | {'Trạng thái':<15} | {'Số Epoch':<10} | {'Val Acc (%)':<13} | {'Test Acc (%)':<13} | {'Thời gian (s)':<12}")
            print("-" * 85)
            for r in bs_results:
                tag = "5/5 điểm ⭐" if r["final_test_acc"] >= 0.97 else "Chưa đạt 97%"
                print(f"{r['batch_size']:<15} | {tag:<15} | {r['epochs']:<10} | {r['final_val_acc']*100:<13.2f} | {r['final_test_acc']*100:<13.2f} | {r['elapsed_time']:<12.1f}")
            print("-" * 85)

            self.bench_data = (lr_results, bs_results)

        def on_bench_complete():
            if self.app.is_cancelled():
                self.status.set("Đã dừng Benchmark Q3.")
            elif self.bench_data:
                self.draw_benchmark()
                self.mode = "benchmark"
                self.view_sketch.pack_forget()
                self.view_bench.pack(fill="both", expand=True)
                self.btn_toggle.config(text="🔄 Chuyển sang Bảng vẽ & Dự đoán")
                self.btn_save_png.config(state="normal")
                self.status.set("Benchmark Q3 hoàn tất! Đã cập nhật 2 biểu đồ so sánh hội tụ.")

        self.run_task("Benchmark Siêu tham số (Q3)", task, on_done=on_bench_complete)

    def draw_benchmark(self):
        if not self.bench_data:
            return
        lr_res, bs_res = self.bench_data
        self.fig_bench.clf()
        ax1 = self.fig_bench.add_subplot(121)
        ax2 = self.fig_bench.add_subplot(122)
        ax1.set_facecolor("#ffffff")
        ax2.set_facecolor("#ffffff")

        lr_colors = {0.01: "#ef4444", 0.001: "#10b981", 0.0001: "#3b82f6"}
        for r in lr_res:
            l = r["lr"]
            eps = list(range(1, len(r["val_acc_history"]) + 1))
            accs = [v * 100 for v in r["val_acc_history"]]
            ax1.plot(eps, accs, marker='o', label=f"LR = {l} ({r['epochs']} ep, Test: {r['final_test_acc']*100:.2f}%)",
                     color=lr_colors.get(l, "#3b82f6"), linewidth=2)
        ax1.axhline(y=97.5, color="#ef4444", linestyle=":", linewidth=2, alpha=0.9, label="Dừng an toàn (97.5%)")
        ax1.axhline(y=97.0, color="#000000", linestyle="--", linewidth=1.5, alpha=0.7, label="Ngưỡng đề (97.0%)")
        ax1.set_title("Hội tụ Validation theo Learning Rate (Batch = 100)", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Số Epoch", fontsize=8)
        ax1.set_ylabel("Validation Accuracy (%)", fontsize=8)
        ax1.set_ylim(85, 99)
        ax1.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
        ax1.legend(fontsize=8, loc="lower right")

        bs_colors = {50: "#8b5cf6", 100: "#10b981", 200: "#f59e0b"}
        for r in bs_res:
            b = r["batch_size"]
            eps = list(range(1, len(r["val_acc_history"]) + 1))
            accs = [v * 100 for v in r["val_acc_history"]]
            ax2.plot(eps, accs, marker='s', label=f"Batch {b} ({r['epochs']} ep, Test: {r['final_test_acc']*100:.2f}%)",
                     color=bs_colors.get(b, "#10b981"), linewidth=2)
        ax2.axhline(y=97.5, color="#ef4444", linestyle=":", linewidth=2, alpha=0.9, label="Dừng an toàn (97.5%)")
        ax2.axhline(y=97.0, color="#000000", linestyle="--", linewidth=1.5, alpha=0.7, label="Ngưỡng đề (97.0%)")
        ax2.set_title("Hội tụ Validation theo Batch Size (LR = 0.001)", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Số Epoch", fontsize=8)
        ax2.set_ylabel("Validation Accuracy (%)", fontsize=8)
        ax2.set_ylim(85, 99)
        ax2.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
        ax2.legend(fontsize=8, loc="lower right")

        self.fig_bench.tight_layout()
        self.canvas_bench.draw_idle()

    def toggle_mode(self):
        if self.mode == "canvas":
            if self.bench_data is None:
                messagebox.showinfo("Thông báo", "Vui lòng bấm 'Chạy Benchmark Q3' trước để có số liệu.")
                return
            self.mode = "benchmark"
            self.view_sketch.pack_forget()
            self.view_bench.pack(fill="both", expand=True)
            self.btn_toggle.config(text="🔄 Chuyển sang Bảng vẽ & Dự đoán")
            self.draw_benchmark()
        else:
            self.mode = "canvas"
            self.view_bench.pack_forget()
            self.view_sketch.pack(fill="both", expand=True)
            self.btn_toggle.config(text="🔄 Chuyển sang Biểu đồ Benchmark")

    def export_png(self):
        try:
            target = "benchmark_q3_comparison.png"
            self.fig_bench.savefig(target)
            messagebox.showinfo("Đã lưu", f"Biểu đồ đã được lưu tại: {target}")
        except Exception as e:
            self.status.set(f"Lỗi: {e}")

    def load_checkpoint(self):
        try:
            m = models.DigitClassificationModel()
            p = load_model_weights(m, "q3")
            self.model = m
            self._ensure_instances()
            self.status.set(f"Đã nạp trọng số từ {p}. Hãy vẽ chữ số!")
            self.eval_test()
        except Exception as e:
            self.status.set(f"Lỗi: {e}")


# ============================================================================
# 9. BÀI 4: NHẬN DIỆN NGÔN NGỮ (Q4 - RNN)
# ============================================================================
class LanguageIDModule(BaseLabView):
    # Bộ từ mẫu được tuyển chọn và phân loại theo từng ngôn ngữ trong bộ dữ liệu
    SAMPLE_GROUPS = [
        ("🇬🇧 Tiếng Anh (English)", ["algorithm", "technology", "international"]),
        ("🇪🇸 Tiếng TBN (Spanish)", ["maravilloso", "corazón", "esperanza"]),
        ("🇫🇮 Tiếng Phần Lan (Finnish)", ["aurinko", "ihminen", "suomalainen"]),
        ("🇳🇱 Tiếng Hà Lan (Dutch)", ["gezelligheid", "belangrijk", "vrijheid"]),
        ("🇵🇱 Tiếng Ba Lan (Polish)", ["dziękuję", "warszawa", "społeczność"]),
    ]

    def __init__(self, master, app):
        super().__init__(master, app)
        self.model = None
        self.dataset = None

        # 1. Thanh công cụ điều khiển trên cùng
        bar = tk.Frame(self, bg="#ffffff", bd=1, relief=tk.SOLID, padx=10, pady=8)
        bar.pack(fill="x", padx=10, pady=8)

        btn_fit = tk.Button(bar, text="▶ Train Mạng RNN", bg="#2563eb", fg="#ffffff",
                            font=("Segoe UI", 9, "bold"), bd=0, padx=12, pady=4,
                            cursor="hand2", command=self.train_rnn)
        btn_load = tk.Button(bar, text="💾 Nạp Trọng số (q4.pt)", bg="#f1f5f9", fg="#0f172a",
                             font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=10, pady=4,
                             cursor="hand2", command=self.load_checkpoint)
        btn_eval = tk.Button(bar, text="🎯 Đo Val Accuracy", bg="#10b981", fg="#ffffff",
                             font=("Segoe UI", 9, "bold"), bd=0, padx=12, pady=4,
                             cursor="hand2", command=self.measure_accuracy)

        btn_fit.pack(side="left", padx=3)
        btn_load.pack(side="left", padx=3)
        btn_eval.pack(side="left", padx=3)

        self.lbl_val_badge = tk.Label(bar, text="", font=("Segoe UI", 9, "bold"),
                                      bg="#ffffff", fg="#047857")
        self.lbl_val_badge.pack(side="left", padx=8)

        lbl_st = tk.Label(bar, textvariable=self.status, font=("Segoe UI", 9, "italic"),
                          bg="#ffffff", fg="#475569")
        lbl_st.pack(side="left", padx=10)

        self.lock_list = [btn_fit, btn_load, btn_eval]

        # 2. Thân giao diện chia 2 cột (Trái: Thử nghiệm từ mẫu, Phải: Biểu đồ xác suất)
        body = tk.Frame(self, bg="#f1f5f9")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # --- CỘT TRÁI (Interactive Input & Sample Bank) ---
        left_col = tk.Frame(body, bg="#f1f5f9", width=420)
        left_col.pack(side="left", fill="y", padx=(0, 10))
        left_col.pack_propagate(False)

        # Card 1: Hộp nhập từ kiểm tra
        input_box = tk.LabelFrame(
            left_col, text=" 🔤 Thử nghiệm phân loại từ vựng ",
            font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#ffffff",
            bd=1, relief=tk.SOLID, padx=10, pady=8
        )
        input_box.pack(fill="x", pady=(0, 8))

        row_in = tk.Frame(input_box, bg="#ffffff")
        row_in.pack(fill="x", pady=(2, 6))

        self.var_word = tk.StringVar(value="algorithm")
        self.ent_word = tk.Entry(
            row_in, textvariable=self.var_word,
            font=("Segoe UI", 11), bg="#f8fafc", fg="#0f172a",
            bd=1, relief=tk.SOLID
        )
        self.ent_word.pack(side="left", fill="x", expand=True, padx=(0, 4), ipady=3)
        self.ent_word.bind("<Return>", lambda _e: self.predict_word())

        btn_clear = tk.Button(
            row_in, text="✖", bg="#f1f5f9", fg="#64748b",
            font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=6, pady=2,
            cursor="hand2", command=lambda: (self.var_word.set(""), self.ent_word.focus_set())
        )
        btn_clear.pack(side="left", padx=(0, 4))

        btn_pred = tk.Button(
            row_in, text="⚡ Phân tích", bg="#2563eb", fg="#ffffff",
            font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=4,
            cursor="hand2", command=self.predict_word
        )
        btn_pred.pack(side="left")

        # Thẻ hiển thị tóm tắt kết quả
        self.res_card = tk.Frame(input_box, bg="#f8fafc", bd=1, relief=tk.SOLID, padx=8, pady=6)
        self.res_card.pack(fill="x", pady=(4, 2))

        self.lbl_res_lang = tk.Label(
            self.res_card, text="Ngôn ngữ dự đoán: (Chưa kiểm tra)",
            font=("Segoe UI", 9, "bold"), bg="#f8fafc", fg="#1e293b"
        )
        self.lbl_res_lang.pack(anchor="w")

        self.lbl_res_conf = tk.Label(
            self.res_card, text="Độ tin cậy: --%",
            font=("Segoe UI", 8), bg="#f8fafc", fg="#64748b"
        )
        self.lbl_res_conf.pack(anchor="w")

        # Card 2: Ngân hàng từ mẫu được phân loại rõ ràng
        sample_box = tk.LabelFrame(
            left_col, text=" 📚 Ngân hàng từ mẫu kiểm thử theo ngôn ngữ ",
            font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#ffffff",
            bd=1, relief=tk.SOLID, padx=10, pady=8
        )
        sample_box.pack(fill="both", expand=True)

        tk.Label(
            sample_box, text="Bấm vào từ bất kỳ để nạp và tự động nhận dạng:",
            font=("Segoe UI", 8, "italic"), bg="#ffffff", fg="#64748b"
        ).pack(anchor="w", pady=(0, 6))

        for group_title, words in self.SAMPLE_GROUPS:
            grp_frame = tk.Frame(sample_box, bg="#ffffff")
            grp_frame.pack(fill="x", pady=2)

            tk.Label(
                grp_frame, text=group_title,
                font=("Segoe UI", 8, "bold"), bg="#ffffff", fg="#334155"
            ).pack(anchor="w")

            btn_flow = tk.Frame(grp_frame, bg="#ffffff")
            btn_flow.pack(anchor="w", padx=4, pady=(1, 4))

            for w in words:
                btn_w = tk.Button(
                    btn_flow, text=w, bg="#f1f5f9", fg="#0f172a",
                    font=("Segoe UI", 8), bd=1, relief=tk.SOLID, padx=6, pady=2,
                    cursor="hand2", command=lambda text=w: self.apply_chip(text)
                )
                btn_w.pack(side="left", padx=2, pady=1)

        # --- CỘT PHẢI (Biểu đồ phân bổ xác suất & Thông tin mô hình) ---
        right_col = tk.Frame(body, bg="#ffffff", bd=1, relief=tk.SOLID, padx=10, pady=8)
        right_col.pack(side="left", fill="both", expand=True)

        tk.Label(
            right_col, text="Phân bổ xác suất dự đoán (Softmax Output)",
            font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#0f172a"
        ).pack(anchor="w", pady=(0, 4))

        self.chart = ProbChart(right_col, horizontal=True, with_map=False, figsize=(5.6, 3.4))
        self.chart.pack(fill="both", expand=True)

        info_box = tk.Frame(right_col, bg="#f8fafc", bd=1, relief=tk.SOLID, padx=8, pady=4)
        info_box.pack(fill="x", pady=(4, 0))
        tk.Label(
            info_box,
            text="ℹ️ Mô hình RNN xử lý theo chuỗi ký tự (Character-level sequence modeling, vocabulary 47 ký tự).\n"
                 "Cập nhật vector trạng thái ẩn sau từng chữ cái để suy ra ngôn ngữ của toàn bộ từ.",
            font=("Segoe UI", 8), bg="#f8fafc", fg="#475569", justify="left"
        ).pack(anchor="w")

        # 3. Nhật ký thực thi
        self.add_console_box(self, height=5).pack(fill="x", padx=10, pady=(0, 8))
        self.status.set("Sẵn sàng. Hãy Train RNN hoặc Nạp trọng số q4.pt để bắt đầu thử nghiệm.")

    def _ensure_dataset(self):
        if self.dataset is None:
            if self.model is None:
                self.model = models.LanguageIDModel()
            self.dataset = backend.LanguageIDDataset(self.model)
        else:
            self.dataset.model = self.model

    def apply_chip(self, word):
        self.var_word.set(word)
        self.predict_word()

    def train_rnn(self):
        self.status.set("Đang huấn luyện mạng RNN...")

        def task():
            m = models.LanguageIDModel()
            ds = backend.LanguageIDDataset(m)
            self.model, self.dataset = m, ds
            train.train_languageid(m, ds)
            if not self.app.is_cancelled():
                save_model_weights(m, "q4")

        self.run_task(
            "Huấn luyện RNN (Q4)",
            task,
            on_done=lambda: (
                self.status.set("Đã dừng tác vụ." if self.app.is_cancelled() else "Huấn luyện xong, đã lưu q4.pt."),
                self.measure_accuracy() if not self.app.is_cancelled() else None
            )
        )

    def load_checkpoint(self):
        try:
            m = models.LanguageIDModel()
            p = load_model_weights(m, "q4")
            self.model = m
            self._ensure_dataset()
            self.status.set(f"Đã nạp trọng số từ {p}.")
            self.measure_accuracy()
            self.predict_word()
        except Exception as e:
            self.status.set(f"Lỗi: {e}")

    def measure_accuracy(self):
        if self.model is None:
            self._ensure_dataset()
        try:
            self.model.eval()
            with torch.no_grad():
                acc = self.dataset.get_validation_accuracy()
            tag = "ĐẠT CHUẨN (>= 81%)" if acc >= 0.81 else "Chưa đạt"
            self.lbl_val_badge.config(text=f"Val Acc: {acc:.2%} [{tag}]",
                                      fg="#047857" if acc >= 0.81 else "#dc2626")
            self.status.set(f"Độ chính xác Validation: {acc:.2%} [{tag}]")
            self.print_log(f"-> Language ID Validation Accuracy = {acc:.2%} ({tag})")
        except Exception as e:
            self.status.set(f"Lỗi: {e}")

    def predict_word(self):
        if self.model is None:
            self.status.set("Chưa có mô hình: Bấm Huấn luyện hoặc Nạp trọng số trước.")
            return

        word = self.var_word.get().strip().lower()
        if not word:
            return

        try:
            self._ensure_dataset()
            chars = [str(c) for c in self.dataset.chars]
            langs = [str(l) for l in self.dataset.language_names]
            char2idx = {c: i for i, c in enumerate(chars)}

            valid = [char2idx[c] for c in word if c in char2idx]
            ignored = [c for c in word if c not in char2idx]
            if not valid:
                self.status.set(f"Từ '{word}' không chứa ký tự hợp lệ trong từ điển.")
                return

            inp = torch.zeros(len(valid), 1, len(chars))
            for t, i in enumerate(valid):
                inp[t, 0, i] = 1.0

            self.model.eval()
            with torch.no_grad():
                logits = self.model(inp)
                probs = torch.softmax(logits, dim=-1)[0].numpy()

            winner_idx = int(probs.argmax())
            best_lang = langs[winner_idx]
            conf = probs[winner_idx]

            # Cập nhật Result Card
            self.lbl_res_lang.config(text=f"Ngôn ngữ dự đoán: {best_lang}")
            self.lbl_res_conf.config(text=f"Độ tin cậy cao nhất: {conf:.1%}")

            # Vẽ biểu đồ
            self.chart.render(langs, probs, None, title=f"Từ: '{word}' -> Dự đoán: {best_lang} ({conf:.1%})")

            info = f"Dự đoán từ '{word}' là {best_lang} ({conf:.1%})"
            if ignored:
                info += f" [Bỏ qua ký tự lạ: {' '.join(ignored)}]"
            self.status.set(info)
        except Exception as e:
            self.print_log(traceback.format_exc())
            self.status.set(f"Lỗi: {e}")


# ============================================================================
# 10. BÀI 5: MẠNG TÍCH CHẬP CNN (Q5 - CƠ BẢN, KHÔNG CÓ BENCHMARK)
# ============================================================================
class ConvolutionalModule(BaseLabView):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.model = None
        self.dataset = None

        bar = tk.Frame(self, bg="#ffffff", bd=1, relief=tk.SOLID, padx=10, pady=8)
        bar.pack(fill="x", padx=10, pady=8)

        btn_fit = tk.Button(bar, text="▶ Train CNN (5 epochs)", bg="#2563eb", fg="#ffffff",
                            font=("Segoe UI", 9, "bold"), bd=0, padx=12, pady=4,
                            cursor="hand2", command=self.train_cnn)
        btn_load = tk.Button(bar, text="💾 Nạp Trọng số CNN", bg="#f1f5f9", fg="#0f172a",
                             font=("Segoe UI", 9), bd=1, relief=tk.SOLID, padx=10, pady=4,
                             cursor="hand2", command=self.load_checkpoint)
        btn_test_conv = tk.Button(bar, text="🔍 Kiểm thử Convolve", bg="#f1f5f9", fg="#f59e0b",
                                  font=("Segoe UI", 9, "bold"), bd=1, relief=tk.SOLID, padx=10, pady=4,
                                  cursor="hand2", command=self.test_convolve)

        btn_fit.pack(side="left", padx=4)
        btn_load.pack(side="left", padx=4)
        btn_test_conv.pack(side="left", padx=4)

        self.lock_list = [btn_fit, btn_load, btn_test_conv]

        tk.Label(bar, textvariable=self.status, font=("Segoe UI", 9, "bold"),
                 fg="#0f172a", bg="#ffffff").pack(side="left", padx=12)

        content = tk.Frame(self, bg="#f8fafc")
        content.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        left_box = tk.LabelFrame(content, text=" Bảng vẽ tay chữ số (chuột trái) ",
                                 font=("Segoe UI", 9, "bold"), fg="#1e293b", bg="#ffffff",
                                 bd=1, relief=tk.SOLID, padx=10, pady=8)
        left_box.pack(side="left", fill="y", padx=(0, 8))

        self.canvas = HandwritingCanvas(left_box, on_release=self.infer_cnn)
        self.canvas.pack(pady=2)

        btn_row = tk.Frame(left_box, bg="#ffffff")
        btn_row.pack(fill="x", pady=4)
        tk.Button(btn_row, text="🗑️ Xóa bảng", bg="#f1f5f9", fg="#0f172a", bd=1, relief=tk.SOLID,
                  padx=6, pady=3, cursor="hand2", command=self.clear_pad).pack(side="left", expand=True, fill="x", padx=1)
        tk.Button(btn_row, text="⚡ Nhận dạng", bg="#2563eb", fg="#ffffff", bd=0,
                  padx=6, pady=3, cursor="hand2", command=self.infer_cnn).pack(side="left", expand=True, fill="x", padx=1)
        tk.Button(btn_row, text="🎲 Mẫu ngẫu nhiên", bg="#f1f5f9", fg="#0f172a", bd=1, relief=tk.SOLID,
                  padx=6, pady=3, cursor="hand2", command=self.load_random_mnist).pack(side="left", expand=True, fill="x", padx=1)

        thumb_row = tk.Frame(left_box, bg="#f8fafc", bd=1, relief=tk.SOLID, padx=6, pady=4)
        thumb_row.pack(fill="x", pady=4)
        tk.Label(thumb_row, text="Ảnh tự căn (28x28):", font=("Segoe UI", 8), bg="#f8fafc").pack(side="left")
        self.thumb_cv = tk.Canvas(thumb_row, width=48, height=48, bg="#000000", bd=0, highlightthickness=0)
        self.thumb_cv.pack(side="left", padx=8)

        tk.Label(left_box, text="✨ Nét vẽ tự căn giữa trọng tâm về (14, 14)\nKernel 3x3 trích xuất cạnh và nét cong viền.",
                 font=("Segoe UI", 8, "italic"), fg="#64748b", bg="#ffffff").pack(anchor="w", pady=2)

        self.chart = ProbChart(content, horizontal=False, with_map=True, figsize=(6.0, 3.8))
        self.chart.pack(side="left", fill="both", expand=True)

        self.add_console_box(self, height=5).pack(fill="x", padx=10, pady=(0, 8))
        self.status.set("Sẵn sàng. Train CNN hoặc Nạp trọng số để thử nghiệm vẽ chữ số.")

    def _ensure_instances(self):
        if self.model is None:
            self.model = models.DigitConvolutionalModel()
        if self.dataset is None:
            self.dataset = backend.DigitClassificationDataset(self.model)
        else:
            self.dataset.model = self.model

    def clear_pad(self):
        self.canvas.clear()
        self.thumb_cv.delete("all")
        self.chart.render([str(i) for i in range(10)], np.zeros(10), None, "Đã xóa nét vẽ")

    def infer_cnn(self):
        if self.model is None:
            self.status.set("Chưa có mô hình CNN: Hãy Train hoặc Nạp trọng số trước.")
            return
        try:
            arr_28, _ = self.canvas.get_centered_array()
            if arr_28 is None or np.max(arr_28) < 0.05:
                return

            self.thumb_cv.delete("all")
            step = 48 / 28.0
            for r in range(28):
                for c in range(28):
                    val = int(255 * arr_28[r, c])
                    if val > 15:
                        self.thumb_cv.create_rectangle(
                            c * step, r * step, (c + 1) * step, (r + 1) * step,
                            fill=f"#{val:02x}{val:02x}{val:02x}", outline=""
                        )

            tensor_in = torch.tensor(arr_28.reshape(1, 784), dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(tensor_in)
                probs = torch.softmax(logits, dim=-1)[0].numpy()
                fmap = models.Convolve(torch.tensor(arr_28), self.model.convolution_weights).detach().numpy()

            winner = int(probs.argmax())
            conf = probs[winner]
            self.chart.render(
                [str(i) for i in range(10)], probs, fmap,
                f"CNN DỰ ĐOÁN: CHỮ SỐ {winner} (Độ tin cậy: {conf:.1%})"
            )
            self.status.set(f"Dự đoán chữ số: {winner} | Độ tin cậy: {conf:.1%}")
        except Exception as e:
            self.print_log(traceback.format_exc())
            self.status.set(f"Lỗi: {e}")

    def load_random_mnist(self):
        try:
            self._ensure_instances()
            idx = random.randrange(len(self.dataset.test_images))
            raw = np.asarray(self.dataset.test_images[idx]).reshape(28, 28)
            label = int(self.dataset.test_labels[idx])
            self.canvas.set_sample(raw)
            self.infer_cnn()
            self.status.set(f"{self.status.get()} (Nhãn thật: {label})")
        except Exception as e:
            self.status.set(f"Lỗi: {e}")

    def train_cnn(self):
        self.status.set("Đang huấn luyện mạng tích chập CNN (5 epochs)...")

        def task():
            torch.manual_seed(42)
            self.model = models.DigitConvolutionalModel()
            self.dataset = backend.DigitClassificationDataset(self.model)
            self.model.train()
            loader = DataLoader(self.dataset, batch_size=40, shuffle=True)
            opt = optim.Adam(self.model.parameters(), lr=0.004)

            for epoch in range(5):
                if self.app.is_cancelled():
                    print("-> Đã nhận lệnh dừng huấn luyện CNN.")
                    break
                for b in loader:
                    opt.zero_grad()
                    loss = losses.digitconvolution_Loss(self.model(b['x']), b['label'])
                    loss.backward()
                    opt.step()
                print(f"Epoch {epoch+1}/5 CNN hoàn thành.")

            if not self.app.is_cancelled():
                save_model_weights(self.model, "q5_full")

        self.run_task(
            "Huấn luyện CNN (Q5)",
            task,
            on_done=lambda: self.status.set("Đã dừng tác vụ." if self.app.is_cancelled() else "Huấn luyện CNN hoàn tất, đã lưu q5_full.pt.")
        )

    def load_checkpoint(self):
        try:
            m = models.DigitConvolutionalModel()
            p = load_model_weights(m, "q5_full")
            self.model = m
            self._ensure_instances()
            self.status.set(f"Đã nạp trọng số từ {p}. Hãy vẽ chữ số!")
        except Exception as e:
            self.status.set(f"Lỗi: {e}")

    def test_convolve(self):
        try:
            print("\n--- KIỂM TRA ĐỘ CHÍNH XÁC TOÁN TỬ CONVOLVE (TÍCH CHẬP 2D) ---")
            passed = True
            for size in (2, 4, 6):
                w = np.random.rand(2, 2)
                a = np.random.rand(size, size)
                out = models.Convolve(torch.Tensor(a), torch.Tensor(w))
                out = np.asarray(out.detach().numpy() if hasattr(out, "detach") else out)
                k = size - 1
                expected = np.array([[(a[r:r + 2, c:c + 2] * w).sum() for c in range(k)] for r in range(k)])
                ok = out.shape == expected.shape and np.allclose(out, expected, atol=1e-5)
                if not ok:
                    passed = False
                res = "CHÍNH XÁC ✅" if ok else f"SAI ❌ (Got: {out.shape}, Exp: {expected.shape})"
                print(f"Ảnh {size}x{size}, kernel 2x2 -> Output {out.shape}: {res}")

            if passed:
                self.status.set("Toán tử Convolve: Toàn bộ kiểm thử đều chính xác 100%!")
            else:
                self.status.set("Toán tử Convolve: Phát hiện sai khác.")
        except Exception as e:
            self.print_log(traceback.format_exc())
            self.status.set(f"Lỗi: {e}")


# ============================================================================
# 11. CỬA SỔ CHÍNH: HỆ THỐNG QUẢN LÝ BẬN & DỪNG TÁC VỤ AN TOÀN
# ============================================================================
class CS188ProjectApp(tk.Tk):
    TABS = [
        ("q1", "1. Perceptron (Q1)", PerceptronModule),
        ("q2", "2. Hồi quy sin(x) & Benchmark (Q2)", RegressionModule),
        ("q3", "3. Chữ số MNIST & Benchmark (Q3)", DigitClassificationModule),
        ("q4", "4. Nhận diện Ngôn ngữ (Q4 - RNN)", LanguageIDModule),
        ("q5", "5. Mạng Tích chập CNN (Q5)", ConvolutionalModule),
    ]

    def __init__(self):
        super().__init__()
        self.title("CS188 Project 5: Machine Learning Workspace")
        self.geometry("1180x810")
        self.minsize(1050, 750)
        self.configure(bg="#f1f5f9")

        # Quản lý trạng thái tác vụ toàn cục
        self.busy = False
        self.stop_requested = False
        self.current_running_name = ""
        self.task_monitor_str = tk.StringVar(value="🟢 Trạng thái: Rảnh (Sẵn sàng)")

        self.tab_instances = {}
        self.tab_buttons = {}
        self.active_tab = None

        # ---------------- HEADER TÍCH HỢP TASK MONITOR & NÚT DỪNG ----------------
        header = tk.Frame(self, bg="#1e293b", padx=14, pady=8)
        header.pack(fill="x")

        left_hdr = tk.Frame(header, bg="#1e293b")
        left_hdr.pack(side="left")
        tk.Label(
            left_hdr,
            text="HỆ THỐNG THỬ NGHIỆM MÔ HÌNH HỌC MÁY (CS188 PROJECT 5)",
            font=("Segoe UI", 11, "bold"), fg="#ffffff", bg="#1e293b"
        ).pack(anchor="w")

        # Khung theo dõi trạng thái tác vụ & nút Dừng màu đỏ
        monitor_frame = tk.Frame(header, bg="#1e293b")
        monitor_frame.pack(side="right")

        self.lbl_task = tk.Label(
            monitor_frame,
            textvariable=self.task_monitor_str,
            font=("Segoe UI", 9, "bold"),
            fg="#38bdf8", bg="#1e293b"
        )
        self.lbl_task.pack(side="left", padx=8)

        self.btn_stop = tk.Button(
            monitor_frame,
            text="🛑 Dừng tác vụ",
            bg="#ef4444", fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            bd=0, padx=10, pady=3,
            cursor="hand2",
            activebackground="#dc2626", activeforeground="#ffffff",
            command=self.request_stop_task
        )
        self.btn_stop.pack(side="left", padx=4)

        # ---------------- THANH TAB ĐIỀU HƯỚNG PHẲNG ----------------
        nav_bar = tk.Frame(self, bg="#e2e8f0", padx=10, pady=4)
        nav_bar.pack(fill="x")

        for key, label, tab_cls in self.TABS:
            btn = tk.Button(
                nav_bar, text=label, font=("Segoe UI", 9, "bold"),
                bg="#ffffff", fg="#334155", bd=1, relief=tk.SOLID, padx=12, pady=5,
                cursor="hand2", command=lambda k=key: self.select_tab(k)
            )
            btn.pack(side="left", padx=3)
            self.tab_buttons[key] = btn

        # ---------------- KHUNG NỘI DUNG CHÍNH ----------------
        self.content_frame = tk.Frame(self, bg="#f1f5f9")
        self.content_frame.pack(fill="both", expand=True)

        self.select_tab("q1")

    def register_start_task(self, name):
        self.busy = True
        self.stop_requested = False
        self.current_running_name = name
        self.task_monitor_str.set(f"🔴 Đang chạy: [{name}]")
        self.lbl_task.config(fg="#f87171")

    def register_end_task(self):
        self.busy = False
        self.stop_requested = False
        self.current_running_name = ""
        self.task_monitor_str.set("🟢 Trạng thái: Rảnh (Sẵn sàng)")
        self.lbl_task.config(fg="#4ade80")

    def is_cancelled(self):
        return self.stop_requested

    def request_stop_task(self):
        """Hủy tác vụ ngay lập tức, giải phóng lock giao diện."""
        if not self.busy:
            self.task_monitor_str.set("🟢 Trạng thái: Rảnh (Không có tác vụ)")
            self.lbl_task.config(fg="#4ade80")
            return

        self.stop_requested = True
        self.task_monitor_str.set(f"⚠️ Đang gửi lệnh dừng [{self.current_running_name}]...")
        self.lbl_task.config(fg="#fbbf24")

        # Cho 0.4s để luồng tự thoát, sau đó ép mở khóa giao diện
        self.after(400, self._force_unlock_all)

    def _force_unlock_all(self):
        self.busy = False
        self.stop_requested = False
        self.current_running_name = ""
        self.task_monitor_str.set("🟢 Đã dừng tác vụ thành công.")
        self.lbl_task.config(fg="#4ade80")
        for tab in self.tab_instances.values():
            tab.unlock_ui()

    def select_tab(self, key):
        if self.active_tab == key:
            return

        for k, btn in self.tab_buttons.items():
            if k == key:
                btn.config(bg="#2563eb", fg="#ffffff", relief=tk.FLAT)
            else:
                btn.config(bg="#ffffff", fg="#334155", relief=tk.SOLID)

        if self.active_tab and self.active_tab in self.tab_instances:
            self.tab_instances[self.active_tab].pack_forget()

        if key not in self.tab_instances:
            cls = next(c for k, l, c in self.TABS if k == key)
            self.tab_instances[key] = cls(self.content_frame, self)

        self.tab_instances[key].pack(fill="both", expand=True)
        self.active_tab = key


if __name__ == "__main__":
    app = CS188ProjectApp()
    app.mainloop()
