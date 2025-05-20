import numpy as np
import matplotlib.pyplot as plt


class Lagrange:
    def __init__(
        self,
        diffusivity=1.0,
        wind_speed=0.3,
        wind_direction=0.0,
        sources=np.array([50, 50]),
        dt=1,
        turbulence_intensity=1,
        tau=1,
        enable_3d=False,
        vg=0.1,
    ):
        self.dim = 3 if enable_3d else 2
        self.num_particles = 0
        self.diffusivity = diffusivity
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.dt = dt
        self.chimney_position = sources
        self.particles = np.zeros(
            (self.num_particles, self.dim)
        )  # 粒子位置self.particles[:emission_rate] = chimney_position  # 初始时刻从烟囱释放粒子
        self.particles[: self.chimney_position[self.dim]] = self.chimney_position[
            0 : self.dim
        ]  # 初始时刻从烟囱释放粒子
        self.turbulence_intensity = turbulence_intensity
        self.tau = tau
        self.vx_turb = np.zeros((self.num_particles, 1))
        self.vy_turb = np.zeros((self.num_particles, 1))
        if enable_3d:
            self.vz_turb = np.zeros((self.num_particles, 1))
        self.enable_3d = enable_3d
        self.vg = vg

    def time_step(self):
        new_particles = np.tile(
            self.chimney_position[0 : self.dim], (self.chimney_position[self.dim], 1)
        )
        self.vx_turb = np.vstack(
            (self.vx_turb, np.zeros(shape=(self.chimney_position[self.dim], 1)))
        )
        self.vy_turb = np.vstack(
            (self.vy_turb, np.zeros(shape=(self.chimney_position[self.dim], 1)))
        )
        if self.enable_3d:
            self.vz_turb = np.vstack(
                (self.vz_turb, np.zeros(shape=(self.chimney_position[self.dim], 1)))
            )
        self.num_particles += self.chimney_position[self.dim]
        self.particles = np.vstack((self.particles, new_particles))

        # 添加随机扩散
        self.particles[:, 0] += np.random.normal(
            0, np.sqrt(2 * self.diffusivity * self.dt), self.particles.shape[0]
        )
        self.particles[:, 1] += np.random.normal(
            0, np.sqrt(2 * self.diffusivity * self.dt), self.particles.shape[0]
        )
        if self.enable_3d:
            self.particles[:, 2] += np.random.normal(
                0, np.sqrt(2 * self.diffusivity * self.dt), self.particles.shape[0]
            )

        self.vx_turb = self.vx_turb * (
            1 - self.dt / self.tau
        ) + self.turbulence_intensity * np.random.randn(
            self.num_particles, 1
        ) * np.sqrt(2 * self.dt / self.tau)
        self.vy_turb = self.vy_turb * (
            1 - self.dt / self.tau
        ) + self.turbulence_intensity * np.random.randn(
            self.num_particles, 1
        ) * np.sqrt(2 * self.dt / self.tau)
        self.particles[:, 0] += np.squeeze(
            (self.vx_turb + self.wind_speed * np.cos(self.wind_direction)) * self.dt
        )
        self.particles[:, 1] += np.squeeze(
            (self.vy_turb + self.wind_speed * np.sin(self.wind_direction)) * self.dt
        )

        if self.enable_3d:
            self.vz_turb = self.vz_turb * (
                1 - self.dt / self.tau
            ) + self.turbulence_intensity * np.random.randn(
                self.num_particles, 1
            ) * np.sqrt(2 * self.dt / self.tau)
            self.particles[:, 1] += np.squeeze((self.vz_turb + self.vg) * self.dt)
        return self.particles


class gaussian_interpolation:
    def __init__(self, sigma):
        self.sigma = sigma

    def gaussian_kernel(self, distance):
        return np.exp(-0.5 * (distance / self.sigma) ** 2)

    def gaussian_kernel_interpolation(self, x, y, z, xi, yi, zi, source):
        res = np.zeros_like(xi)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                # 计算目标点到所有已知点的距离
                dist = np.sqrt(
                    (x - xi[i, j]) ** 2 + (y - yi[i, j]) ** 2 + (z - zi) ** 2
                )
                # 计算高斯权重
                weights = self.gaussian_kernel(dist)
                # 加权求和得到目标点的值
                res[i, j] = np.sum(weights * source)
        return res

    def gaussian_kernel_interpolation_2d(self, x, y, xi, yi, source):
        res = np.zeros_like(xi)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                # 计算目标点到所有已知点的距离
                dist = np.sqrt((x - xi[i, j]) ** 2 + (y - yi[i, j]) ** 2)
                # 计算高斯权重
                weights = self.gaussian_kernel(dist)
                # 加权求和得到目标点的值
                res[i, j] = np.sum(weights * source)
        return res


def update(
    frame, lagrange_list, gauss_inter, grid_x, grid_y, cax, ax, enable_3d, height
):
    dim = 3 if enable_3d else 2
    particles = np.zeros(
        (0, dim)
    )  # 粒子位置self.particles[:emission_rate] = chimney_position  # 初始时刻从烟囱释放粒子
    for lagrange in lagrange_list:
        # 更新粒子位置
        particles_new = lagrange.time_step()
        particles = np.vstack((particles, particles_new))
    # 插值计算浓度分布
    if enable_3d:
        grid_z = gauss_inter.gaussian_kernel_interpolation(
            particles[:, 0],
            particles[:, 1],
            particles[:, 2],
            grid_x,
            grid_y,
            height,
            np.ones(particles.shape[0]),
        )
    else:
        grid_z = gauss_inter.gaussian_kernel_interpolation_2d(
            particles[:, 0],
            particles[:, 1],
            grid_x,
            grid_y,
            np.ones(particles.shape[0]),
        )
    # 更新图像
    cax.set_data(grid_z.T)
    ax.set_title(f"Time Step {frame+1}")
    return (cax,)


def lagrange_process_model(
    sources,
    time_steps,
    dt,
    wind_speed,
    turbulence_intensity,
    st,
    tau,
    vmax,
    enable_3d,
    vg,
    diffusivity,
    height,
):
    grid_x, grid_y = np.mgrid[-100:100:100j, -100:100:100j]  # 100x100的网格

    lagrange_list = []

    for i in range(len(sources)):
        lagrange = Lagrange(
            diffusivity=diffusivity,
            wind_speed=wind_speed,
            wind_direction=np.radians(90),
            sources=sources[i],
            dt=dt,
            turbulence_intensity=turbulence_intensity,
            tau=tau,
            enable_3d=enable_3d,
            vg=vg,
        )
        lagrange_list.append(lagrange)
    sigma = 5
    gauss_inter = gaussian_interpolation(sigma)

    # 创建绘图
    fig, ax = plt.subplots()
    cax = ax.imshow(
        np.zeros_like(grid_x),
        cmap="jet",
        interpolation="bilinear",
        vmin=0,
        vmax=vmax,
    )
    cbar = fig.colorbar(cax)
    cbar.set_label("Concentration")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # 创建动画帧
    placeholder = st.empty()  # 使用 Streamlit 的占位符
    for frame in range(time_steps):
        update(
            frame,
            lagrange_list,
            gauss_inter,
            grid_x,
            grid_y,
            cax,
            ax,
            enable_3d,
            height,
        )
        placeholder.pyplot(fig, use_container_width=True)  # 更新同一位置的图像
