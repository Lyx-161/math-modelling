import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from lagrange_process import gaussian_interpolation


def generate_randn(
    count: int,
    mean: float = 0.0,
    std_dev: float = 1.0,
) -> np.ndarray:
    """Generates random numbers from a normal distribution."""
    return np.random.normal(mean, std_dev, count)


class LagrangeWCSPH_Cubic:
    def __init__(
        self,
        diffusivity=1.0,
        wind_speed=1.0,
        sources=np.array([50, 50, 0]),  # 污染源位置 (x, y, z)
        dx=0.1,
        dt=1.0,
        turbulence_intensity=0.5,
        tau=10,  # 湍流时间尺度
        viscosity=0.1,  # 粘性系数
        h=5.0,  # 光滑长度
        mass=0.01,  # 粒子质量
    ):
        self.dtype = np.float16  # 使用更高精度的浮点数以减少数值误差
        self.diffusivity = diffusivity
        self.wind = wind_speed * np.array(
            [1.0, 0.0], dtype=self.dtype
        )  # 风方向设为 x 方向
        self.turbulence_intensity = turbulence_intensity
        self.tau = tau
        self.dt = dt
        self.dx = dx
        self.viscosity = viscosity
        self.h = h
        self.mass = mass
        self.sources = sources  # 初始源位置
        self.source_num = len(sources)

        # 粒子初始化
        self.particles = np.zeros((0, 2), dtype=self.dtype)
        self.emission_rate = len(self.particles)
        self.velocities = (
            np.random.randn(self.emission_rate, 2).astype(self.dtype)
            * self.turbulence_intensity
        )
        self.densities = np.zeros(self.emission_rate, dtype=self.dtype)
        self.forces = np.zeros_like(self.particles)

    def _kernel(self, r):
        """3D Cubic Spline 核函数"""
        q = r / self.h
        w = np.zeros_like(r, dtype=self.dtype)
        h3 = self.h**3
        inside = q <= 2.0
        w[inside] = (8 / (np.pi * h3)) * (
            (1 - 1.5 * q[inside] ** 2 + 0.75 * q[inside] ** 3) * (q[inside] <= 1)
            + (0.25 * (2 - q[inside]) ** 3) * ((q[inside] > 1) & (q[inside] <= 2))
        )
        return w

    def _gradient(self, r):
        """Cubic Kernel 的梯度"""
        q = r / self.h
        grad_w = np.zeros_like(r, dtype=self.dtype)
        mask1 = q <= 1.0
        mask2 = (q > 1.0) & (q <= 2.0)
        grad_w[mask1] = (8 / (np.pi * self.h**5)) * (-3 * q[mask1] + 3 * q[mask1] ** 3)
        grad_w[mask2] = 0.75 * (2 - q[mask2]) ** 2 - 1.5 * (2 - q[mask2]) * q[mask2]
        return grad_w

    def generate_random(self, source):
        pos_x = generate_randn(source[3], source[0], source[2])
        pos_y = generate_randn(source[3], source[1], source[2])
        return np.stack((pos_x, pos_y), axis=1)

    def _compute_density(self, positions, idxs, dists):
        densities = np.zeros(len(positions), dtype=self.dtype)
        for i in range(len(positions)):
            r = dists[i][np.isfinite(dists[i])]
            densities[i] = np.sum(self.mass * self._kernel(r))
        return densities

    def _compute_viscosity_force(self, positions, velocities, idxs, dists):
        """计算粘性力"""
        acc_viscosity = np.zeros_like(positions, dtype=self.dtype)
        for i in range(len(positions)):
            neighbor_indices = idxs[i][np.isfinite(dists[i])]
            for j in neighbor_indices:
                if i == j:
                    continue
                r_ij = positions[j] - positions[i]
                d_ij = np.linalg.norm(r_ij) + 1e-8
                v_ij = velocities[j] - velocities[i]
                grad_w = self._gradient(d_ij) * r_ij / d_ij
                acc_viscosity[i] += self.mass * v_ij * grad_w
        return -self.viscosity * acc_viscosity

    def _compute_pressure_gradient(self, positions, velocities, idxs, dists):
        # 使用预查找的 idxs & dists
        pressure = self.densities**2
        acc_pressure = np.zeros_like(positions, dtype=self.dtype)
        for i in range(len(positions)):
            neighbor_indices = idxs[i][np.isfinite(dists[i])]
            for j in neighbor_indices:
                if i == j:
                    continue
                r_ij = positions[j] - positions[i]
                d_ij = np.linalg.norm(r_ij) + 1e-8
                grad_w = self._gradient(d_ij) * r_ij / d_ij
                acc_pressure[i] += (
                    self.mass * (pressure[j] / self.densities[j] ** 2) * grad_w
                )
        return -acc_pressure

    def time_step(self):
        # 释放新粒子
        for source in self.sources:
            new_positions = self.generate_random(source)
            self.emission_rate = len(new_positions)
            new_velocities = (
                np.random.randn(self.emission_rate, 2).astype(self.dtype)
                * self.turbulence_intensity
                + self.wind
            )
            self.particles = np.vstack((self.particles, new_positions))
            self.velocities = np.vstack((self.velocities, new_velocities))

        # 构建 KDTree（统一查找）
        print("KDTree")
        tree = cKDTree(self.particles)
        dists, idxs = tree.query(self.particles, k=30, distance_upper_bound=self.h * 2)

        # 1. 计算密度
        print("Compute Density")
        self.densities = self._compute_density(self.particles, idxs, dists)
        # 2.
        print("Compute Pressure")
        acc_pressure = self._compute_pressure_gradient(
            self.particles, self.velocities, idxs, dists
        )

        # 3. 计算粘性力
        print("Compute Viscosity")
        acc_viscosity = self._compute_viscosity_force(
            self.particles, self.velocities, idxs, dists
        )
        # 4. 外部力（风 + 湍流 + 扩散）
        print("Compute External Forces")
        vis_turbulence = self.velocities * (
            1 - self.dt / self.tau
        ) + self.turbulence_intensity * np.random.randn(len(self.particles), 2).astype(
            self.dtype
        ) * np.sqrt(2 * self.dt / self.tau)
        vis_diffusion = np.random.normal(
            0, np.sqrt(2 * self.diffusivity * self.dt), self.particles.shape
        ).astype(self.dtype)

        # 5. 总加速度
        total_acc = acc_pressure + acc_viscosity

        # 6. 更新速度和位置
        print("update")
        self.velocities += total_acc * self.dt
        self.particles += (
            self.velocities + self.wind + vis_turbulence + vis_diffusion
        ) * self.dt
        # breakpoint()
        return self.particles


def update(frame, SPH, gauss_inter, grid_x, grid_y, cax, ax, height):
    # 更新粒子位置
    particles = SPH.time_step()
    # particles = np.vstack((particles, particles_new))
    # 插值计算浓度分布
    grid_z = gauss_inter.gaussian_kernel_interpolation_2d(
        particles[:, 0],
        particles[:, 1],
        # particles[:, 2],
        grid_x,
        grid_y,
        # height,
        np.ones(particles.shape[0]),
    )
    # 更新图像
    cax.set_data(grid_z.T)
    # cax._offset = (particles[:, 0], particles[:, 1])
    ax.set_title(f"Time Step {frame+1}")
    return (cax,)


def sph_process_model(
    sources,
    time_steps=100,
    dt=1.0,
    dx=1.0,
    h=5.0,
    wind_speed=5.0,
    turbulence_intensity=0.5,
    st=None,
    tau=10,
    vg=0.1,
    diffusivity=1.0,
    mass=0.01,
    height=100,
):
    # 定义网格
    grid_x, grid_y = np.mgrid[-100:100:100j, -100:100:100j]

    SPH = LagrangeWCSPH_Cubic(
        diffusivity=diffusivity,
        wind_speed=wind_speed,
        sources=sources,
        dx=dx,
        dt=dt,
        turbulence_intensity=turbulence_intensity,
        tau=tau,
        h=h,
        mass=mass,
    )

    sigma = 5
    gauss_inter = gaussian_interpolation(sigma)

    fig, ax = plt.subplots()
    cax = ax.imshow(
        np.zeros_like(grid_x),
        cmap="jet",
        interpolation="bilinear",
        vmin=0,
        vmax=1000,
    )
    # cax = ax.scatter(
    #     np.zeros_like(grid_x), np.zeros_like(grid_y), c=np.ones_like(grid_x)
    # )
    cbar = fig.colorbar(cax)
    cbar.set_label("Concentration")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")

    # 创建动画帧
    placeholder = st.empty()  # 使用 Streamlit 的占位符
    for frame in range(time_steps):
        print(frame)
        update(
            frame,
            SPH,
            gauss_inter,
            grid_x,
            grid_y,
            cax,
            ax,
            height,
        )
        placeholder.pyplot(fig, use_container_width=True)  # 更新同一位置的图像
        # plt.show()


# if __name__ == "__main__":
#     num_sources = 1
#     sources = []
#     for i in range(num_sources):
#         source_x = 0
#         source_y = 0
#         source_z = 50
#         source_particles = 10
#         sources.append((source_x, source_y, source_z, source_particles))
#         # total_particles += source_particles

#     time_steps = 10
#     dt = 0.1
#     dx = 1.0
#     h = 2.5
#     tau = 10
#     turbulence_intensity = 0.5
#     wind_speed = 5.0
#     vg = 2.0
#     diff = 1.0
#     mass = 0.01
#     vmax = 1000

#     monitor_height = 10

#     sph_process_model(
#         sources,
#         time_steps,
#         dt,
#         dx,
#         h,
#         wind_speed,
#         turbulence_intensity,
#         tau,
#         vg,
#         diff,
#         mass,
#         monitor_height,
#         vmax)
