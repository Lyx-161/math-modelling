import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
from fitter import PolynomialFitter, SplineFitter, RBFFitter, NeuralNetworkFitter
from parameterizer import CentripetalParameterizer, ChordalParameterizer, FoleyNielsenParameterizer, UniformParameterizer

class CurveFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("平面点列曲线拟合")
        self.root.geometry("900x800")

        # 初始化变量
        self.points = []
        self.degree = tk.IntVar(value=3)
        self.canvas_size = (800, 600)  # 默认画布大小
        self.image = self.create_blank_canvas()
        self.fitter_type = tk.StringVar(value="polynomial")
        self.parameterizer_type = tk.StringVar(value="uniform")
        self.loss_type = tk.StringVar(value="mse")
        self.function_type = tk.StringVar(value="Relu")
        self.temp_point = None  # 临时点
        self.real_time_fitting = False  # 实时拟合标志

        # 创建 GUI 组件
        self.create_widgets()

    def create_widgets(self):
        # 文件选择部分
        file_frame = ttk.LabelFrame(self.root, text="文件选择")
        file_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(file_frame, text="选择图像文件", command=self.load_image).grid(row=0, column=0, padx=10, pady=10)
        self.file_label = ttk.Label(file_frame, text="空白画布已创建")
        self.file_label.grid(row=0, column=1, padx=10, pady=10)

        # 参数设置部分
        param_frame = ttk.LabelFrame(self.root, text="参数设置")
        param_frame.pack(fill="x", padx=10, pady=10)

        column = 0

        ttk.Label(param_frame, text="参数化类型:").grid(row=0, column=column, padx=10, pady=10)
        parameterizer_types = ["uniform", "chordal", "centripetal", "foley_nielsen"]
        ttk.Combobox(param_frame, textvariable=self.parameterizer_type, values=parameterizer_types, state="readonly", width=15).grid(row=0, column=column+1, padx=10, pady=10)
        column += 2

        ttk.Label(param_frame, text="拟合类型:").grid(row=0, column=column, padx=10, pady=10)
        fitter_types = ["polynomial", "spline", "rbf", "neural"]
        ttk.Combobox(param_frame, textvariable=self.fitter_type, values=fitter_types, state="readonly", width=10).grid(row=0, column=column+1, padx=10, pady=10)
        column += 2

        ttk.Label(param_frame, text="多项式阶数:").grid(row=0, column=column, padx=10, pady=10)
        self.degree_spinbox = ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.degree, width=5)
        self.degree_spinbox.grid(row=0, column=column+1, padx=10, pady=10)
        column = 0

        ttk.Label(param_frame, text="损失函数类型:").grid(row=1, column=column, padx=10, pady=10)
        loss_types = ["mse", "chamfer"]
        ttk.Combobox(param_frame, textvariable=self.loss_type, values=loss_types, state="readonly", width=10).grid(row=1, column=column+1, padx=10, pady=10)
        column += 2

        ttk.Label(param_frame, text="激活函数类型:").grid(row=1, column=column, padx=10, pady=10)
        function_types = ["Relu", "Gelu", "Elu"]
        ttk.Combobox(param_frame, textvariable=self.function_type, values=function_types, state="readonly", width=10).grid(row=1, column=column+1, padx=10, pady=10)
        column += 2

        # 操作按钮部分
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(button_frame, text="拟合曲线", command=self.fit_curve).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(button_frame, text="实时拟合", command=self.toggle_real_time_fitting).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(button_frame, text="清除点", command=self.clear_points).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(button_frame, text="保存结果", command=self.save_result).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=10, pady=10)

        # 结果信息部分
        info_frame = ttk.LabelFrame(self.root, text="结果信息")
        info_frame.pack(fill="x", padx=10, pady=10)

        self.result_text = tk.Text(info_frame, height=5, width=80)
        self.result_text.pack(fill="x", padx=10, pady=10)

        # 创建画布部分
        canvas_frame = ttk.LabelFrame(self.root, text="绘图区域")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size[0], height=self.canvas_size[1], bg="white")
        self.canvas.pack(fill="both", expand=True)

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)  # 绑定右键点击事件
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # 显示初始画布
        self.update_canvas()

    def create_blank_canvas(self):
        # 创建一个空白画布
        return np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * 255

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                self.image = cv2.imread(file_path)
                if self.image is not None:
                    # 调整图像大小以匹配画布
                    self.image = cv2.resize(self.image, (self.canvas_size[0], self.canvas_size[1]))
                    self.file_label.config(text=f"已选择文件: {os.path.basename(file_path)}")
                    self.points = []
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"图像加载成功\n")
                    self.update_canvas()
                else:
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"无法加载图像文件\n")
            except Exception as e:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"图像加载失败: {str(e)}\n")

    def update_canvas(self):
        # 将 OpenCV 图像转换为 Tkinter 可显示的格式
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # 更新画布
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def on_canvas_click(self, event):
        # 获取鼠标点击位置
        x, y = event.x, event.y

        # 添加点
        self.points.append((x, y))
        self.result_text.insert(tk.END, f"点已添加: ({x}, {y})\n")

        # 更新画布显示
        self.update_canvas_with_points()

    def on_canvas_right_click(self, event):
        # 右键点击时添加当前鼠标位置的控制点并结束实时拟合
        x, y = event.x, event.y
        self.points.append((x, y))
        self.result_text.insert(tk.END, f"点已添加: ({x}, {y})\n")
        self.toggle_real_time_fitting()  # 禁用实时拟合
        self.update_canvas_with_points()

    def on_canvas_motion(self, event):
        # 鼠标移动时实时显示拟合曲线
        if len(self.points) >= 3 and self.real_time_fitting:
            self.update_canvas_with_fitting(event.x, event.y)

    def update_canvas_with_points(self):
        # 更新画布显示点
        display_image = self.image.copy()
        for point in self.points:
            cv2.circle(display_image, point, 5, (0, 255, 0), -1)
        
        # 更新画布
        self.image = display_image
        self.update_canvas()

    def update_canvas_with_fitting(self, x, y):
        # 更新画布显示点和拟合曲线
        display_image = self.image.copy()
        for point in self.points:
            cv2.circle(display_image, point, 5, (0, 255, 0), -1)
        
        # 添加临时点
        temp_points = self.points.copy()
        temp_points.append((x, y))

        try:
            # 提取点的坐标
            points = np.array(temp_points)
            x_coords = points[:, 0]
            y_coords = points[:, 1]

            degree = self.degree.get()
            fitter_type = self.fitter_type.get()
            parameterizer_type = self.parameterizer_type.get()
            color = [0, 0, 0]

            if parameterizer_type == "uniform":
                parameterizer = UniformParameterizer(points)
                color[0] = 255
            elif parameterizer_type == "chordal":
                parameterizer = ChordalParameterizer(points)
                color[0] = 128
            elif parameterizer_type == "centripetal":
                parameterizer = CentripetalParameterizer(points)
                color[0] = 64
            elif parameterizer_type == "foley_nielsen":
                parameterizer = FoleyNielsenParameterizer(points)
            else:
                raise ValueError(f"Unsupported parameterizer type: {parameterizer_type}")

            t = parameterizer.parameterize()

            if fitter_type == "polynomial":
                fitter = PolynomialFitter(t, x_coords, y_coords)
                x_fit, y_fit = fitter.fit(degree)
                color[2] = 255
                color[1] = degree * 25
            elif fitter_type == "spline":
                fitter = SplineFitter(t, x_coords, y_coords)
                x_fit, y_fit = fitter.fit()
                color[2] = 128
            elif fitter_type == "rbf":
                fitter = RBFFitter(t, x_coords, y_coords)
                x_fit, y_fit = fitter.fit()
                color[2] = 64
            else:
                # 对于神经网络拟合，不进行实时更新
                pass

            # 绘制拟合曲线
            for i in range(len(x_fit) - 1):
                cv2.line(display_image, (int(x_fit[i]), int(y_fit[i])), (int(x_fit[i+1]), int(y_fit[i+1])), color, 2)

        except Exception as e:
            self.result_text.insert(tk.END, f"拟合失败: {str(e)}\n")

        # 绘制临时点
        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)

        # 更新画布
        self.canvas.delete("all")
        image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image

    def toggle_real_time_fitting(self):
        self.real_time_fitting = not self.real_time_fitting
        if self.real_time_fitting:
            self.result_text.insert(tk.END, "实时拟合已启用\n")
        else:
            self.result_text.insert(tk.END, "实时拟合已禁用\n")

    def fit_curve(self):
        if not self.points:
            self.result_text.insert(tk.END, "请先选择一些点\n")
            return

        # 提取点的坐标
        points = np.array(self.points)
        x = points[:, 0]
        y = points[:, 1]

        degree = self.degree.get()
        fitter_type = self.fitter_type.get()
        parameterizer_type = self.parameterizer_type.get()
        loss_type = self.loss_type.get()
        function_type = self.function_type.get()
        color = [0, 0, 0]

        try:
            if parameterizer_type == "uniform":
                parameterizer = UniformParameterizer(points)
                color[0] = 255
            elif parameterizer_type == "chordal":
                parameterizer = ChordalParameterizer(points)
                color[0] = 128
            elif parameterizer_type == "centripetal":
                parameterizer = CentripetalParameterizer(points)
                color[0] = 64
            elif parameterizer_type == "foley_nielsen":
                parameterizer = FoleyNielsenParameterizer(points)
            else:
                raise ValueError(f"Unsupported parameterizer type: {parameterizer_type}")

            t = parameterizer.parameterize()

            if fitter_type == "polynomial":
                fitter = PolynomialFitter(t, x, y)
                x_fit, y_fit = fitter.fit(degree)
                color[2] = 255
                color[1] = degree * 25
            elif fitter_type == "spline":
                fitter = SplineFitter(t, x, y)
                x_fit, y_fit = fitter.fit()
                color[2] = 128
            elif fitter_type == "rbf":
                fitter = RBFFitter(t, x, y)
                x_fit, y_fit = fitter.fit()
                color[2] = 64
            elif fitter_type == "neural":
                fitter = NeuralNetworkFitter(t, x, y, loss_type, function_type)
                x_fit, y_fit = fitter.fit()
                if loss_type == "mse":
                    color[1] = 255
            else:
                raise ValueError(f"Unsupported fitter type: {fitter_type}")

            # 绘制拟合曲线
            display_image = self.image.copy()
            for point in self.points:
                cv2.circle(display_image, point, 5, (0, 255, 0), -1)

            # 绘制拟合曲线
            for i in range(len(x_fit) - 1):
                cv2.line(display_image, (int(x_fit[i]), int(y_fit[i])), (int(x_fit[i+1]), int(y_fit[i+1])), color, 2)

            # 更新画布
            self.image = display_image
            self.update_canvas()

            self.result_text.insert(tk.END, "拟合成功\n")

        except Exception as e:
            self.result_text.insert(tk.END, f"拟合失败: {str(e)}\n")

    def clear_points(self):
        self.points = []
        self.image = self.create_blank_canvas()
        self.update_canvas()
        self.result_text.insert(tk.END, "所有点已清除\n")

    def save_result(self):
        if not self.points:
            self.result_text.insert(tk.END, "没有数据可保存\n")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                points = np.array(self.points)
                t = np.linspace(0, 1, len(points))
                x = points[:, 0]
                y = points[:, 1]
                degree = self.degree.get()
                fitter_type = self.fitter_type.get()

                if fitter_type == "polynomial":
                    x_coefficients = np.polyfit(t, x, degree)
                    y_coefficients = np.polyfit(t, y, degree)

                    with open(file_path, 'w') as f:
                        f.write(f"拟合类型: {fitter_type}\n")
                        f.write(f"x 拟合多项式系数: {x_coefficients}\n")
                        f.write(f"y 拟合多项式系数: {y_coefficients}\n")
                        f.write("\n点列参数:\n")
                        for point in self.points:
                            f.write(f"({point[0]}, {point[1]})\n")
                elif fitter_type == "spline":
                    with open(file_path, 'w') as f:
                        f.write(f"拟合类型: {fitter_type}\n")
                        f.write("\n点列参数:\n")
                        for point in self.points:
                            f.write(f"({point[0]}, {point[1]})\n")

                self.result_text.insert(tk.END, f"结果已保存到: {file_path}\n")
            except Exception as e:
                self.result_text.insert(tk.END, f"保存失败: {str(e)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = CurveFittingApp(root)
    root.mainloop()