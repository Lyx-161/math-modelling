import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn import datasets
from skimage import io
import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL, Button, Label, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def get_image_as_mat(index):
    ds=datasets.fetch_olivetti_faces()
    return np.mat(ds.images[index])

def get_image_from_file(file_path):
    img=io.imread(file_path)
    return img.astype(np.uint8)

def svd_compress(image, k):
    """改进后的SVD压缩支持彩色图像"""
    if len(image.shape) == 3:  # 彩色图像处理
        channels = []
        for i in range(image.shape[2]):
            # 处理每个颜色通道
            channel = image[:,:,i]
            U, S, VT = la.svd(channel)
            
            # 保留前k个奇异值
            Uk = U[:, :k]
            Sk = np.diag(S[:k])
            VTk = VT[:k, :]
            
            # 重构通道
            reconstructed = Uk @ Sk @ VTk
            channels.append(reconstructed)
        # 合并三个通道并截断到0-255范围
        compressed = np.clip(np.dstack(channels), 0, 255).astype(np.uint8)
        return compressed
    else:  # 灰度图像处理
        U, S, VT = la.svd(image)
        Uk = U[:, :k]
        Sk = np.diag(S[:k])
        VTk = VT[:k, :]
        result=Uk @ Sk @ VTk
        return result
    
def calculate_compression_ratio(image, k):
    m, n = image.shape[:2]
    print(m,n)
    original_size = m * n * (3 if len(image.shape)==3 else 1)
    compressed_size = 3*k*(m + n + 1) if len(image.shape)==3 else k*(m + n + 1)
    return original_size / compressed_size

def calculate_psnr(original, compressed):
    return psnr(original, compressed, data_range=255)

def calculate_ssim(original, compressed):
    # 如果是彩色图像，指定 channel_axis=2
    if original.ndim == 3 and original.shape[2] == 3:
        return ssim(original, compressed, data_range=255, channel_axis=2)
    else:
        return ssim(original, compressed, data_range=255)

class ImageGUI:
    def __init__(self,master):
        self.master=master
        master.title("Image Compression with SVD")

        self.control_frame=Frame(master)
        self.control_frame.pack(side=tk.LEFT,padx=10,pady=10)# Create a frame for the plots
        self.plot_frame = Frame(master)
        self.plot_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create a button to load an image
        self.load_button = Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_button.pack()

        # Create a slider for k value
        self.k_scale = Scale(self.control_frame, from_=1, to=100, orient=HORIZONTAL, label="k value")
        self.k_scale.set(10)
        self.k_scale.pack()

        # Create a button to compress the image
        self.compress_button = Button(self.control_frame, text="Compress Image", command=self.compress_image)
        self.compress_button.pack()

        # Create labels for compression ratio
        self.ratio_label = Label(self.control_frame, text="Compression Ratio: ")
        self.ratio_label.pack()

        # Create a figure for the original image
        self.fig_original = plt.figure(figsize=(4, 4))
        self.canvas_original = FigureCanvasTkAgg(self.fig_original, self.plot_frame)
        self.canvas_original.get_tk_widget().pack(side=tk.LEFT)

        # Create a figure for the compressed image
        self.fig_compressed = plt.figure(figsize=(4, 4))
        self.canvas_compressed = FigureCanvasTkAgg(self.fig_compressed, self.plot_frame)
        self.canvas_compressed.get_tk_widget().pack(side=tk.RIGHT)

        # 创建一个框架用于显示评估指标
        self.metrics_frame = Frame(master)
        self.metrics_frame.pack(side=tk.BOTTOM, padx=10, pady=10)
        
        # 创建标签用于显示PSNR和SSIM
        self.psnr_label = Label(self.metrics_frame, text="PSNR: ")
        self.psnr_label.pack(side=tk.LEFT)
        
        self.ssim_label = Label(self.metrics_frame, text="SSIM: ")
        self.ssim_label.pack(side=tk.LEFT)

        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.compressed_image = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = get_image_from_file(self.image_path)
            self.display_image(self.original_image, self.fig_original, "Original Image")

    def compress_image(self):
        if self.original_image is not None:
            k = self.k_scale.get()
            self.compressed_image = svd_compress(self.original_image, k)
            self.display_image(self.compressed_image, self.fig_compressed, f"Compressed Image (k={k})")
            ratio = calculate_compression_ratio(self.original_image, k)
            self.ratio_label.config(text=f"Compression Ratio: {ratio:.2f}")

    def display_image(self, img, fig, title):
        fig.clear()
        ax = fig.add_subplot(111)
        if len(img.shape) == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=plt.cm.gray)
        ax.set_title(title)
        ax.axis('off')
        self.canvas_original.draw()
        self.canvas_compressed.draw()

    def compress_image(self):
        if self.original_image is not None:
            k = self.k_scale.get()
            self.compressed_image = svd_compress(self.original_image, k)
            self.display_image(self.compressed_image, self.fig_compressed, f"Compressed Image (k={k})")
            ratio = calculate_compression_ratio(self.original_image, k)
            self.ratio_label.config(text=f"Compression Ratio: {ratio:.2f}")
            
            # 计算并更新PSNR和SSIM
            psnr_value = calculate_psnr(self.original_image, self.compressed_image)
            ssim_value = calculate_ssim(self.original_image, self.compressed_image)
            self.psnr_label.config(text=f"PSNR: {psnr_value:.2f} dB")
            self.ssim_label.config(text=f"SSIM: {ssim_value:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageGUI(root)
    root.mainloop()
    