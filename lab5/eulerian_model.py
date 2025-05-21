import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

def eulerian_model(
        sources,                     # [(x, y, z, Q), ...]
        u,                           # 原始 x 向平均风速
        sigma_y0, sigma_z0,          # 经验扩散系数
        time_steps, dt,
        monitor_height,
        p_absorb,                    # 被吸收概率 p
        k_conc,                      # 未吸收时浓度倍率   k   (0–1)
        kx, ky, kz,                  # 速度缩放 kx, ky, kz
        st):


    x = np.linspace(0, 1000, 500)
    y = np.linspace(-500, 500, 500)
    X, Y = np.meshgrid(x, y)
    C_total = np.zeros_like(X)

    two_pi = 2.0 * np.pi
    w_reflect = (1.0 - p_absorb) * k_conc     # 反弹项强度权重

    for _ in range(time_steps):
        C_step = np.zeros_like(X)

        for sx, sy, sz, Q in sources:

            dx = np.maximum(X - sx, 0.0)
            dx_ref = dx * (u / (kx * u + 1e-12))

            sig_y  = sigma_y0 * (dx ** 0.91) + 1e-9
            sig_z  = sigma_z0 * (dx ** 0.92) + 1e-9

            sig_y_r = ky * sig_y
            sig_z_r = kz * sig_z

            # ① 直射项
            Cz_direct = np.exp(-(monitor_height - sz)**2 / (2.0 * sig_z**2))
            C_direct  = (Q / (two_pi * sig_y * sig_z * u)) \
                        * np.exp(-(Y - sy)**2 / (2.0 * sig_y**2)) \
                        * Cz_direct

            # ② 镜像源（反弹）项
            Cz_reflect = np.exp(-(monitor_height + sz)**2 / (2.0 * sig_z_r**2))
            C_reflect  = w_reflect * (Q / (two_pi * sig_y_r * sig_z_r * (kx*u))) \
                         * np.exp(-(Y - sy)**2 / (2.0 * sig_y_r**2)) \
                         * Cz_reflect

            C_step += C_direct + C_reflect

        C_total += C_step * dt                    # 时间积分

    C_plot = gaussian_filter(C_total, sigma=1.5)  # 视觉平滑

    fig = go.Figure(
        data=go.Heatmap(
            z=C_plot, x=X[0, :], y=Y[:, 0],
            colorscale="Viridis",
            zmin=np.percentile(C_plot, 5),
            zmax=np.percentile(C_plot, 99.8),
            colorbar=dict(title="浓度 (g m⁻³)")
        )
    )
    fig.update_layout(
        title=f"Eulerian 扩散 (观测高 {monitor_height} m)",
        xaxis_title="下风向 X (m)",
        yaxis_title="侧向 Y (m)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
