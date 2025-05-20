import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

def gaussian_plume_model(sources, u, sigma_y, sigma_z, st, monitor_height, offset):
    """
    sources: 含有污染源信息的列表，每个污染源为 (source_x, source_y, source_z, source_strength)
    u: 风速 (m/s)
    sigma_y: 初始横向扩散系数（用于参数化公式）
    sigma_z: 初始垂直扩散系数（用于参数化公式）
    monitor_height: 监测高度 (m)
    offset: 平移量，用于确保污染源坐标非负
    """

    x = np.linspace(0, 1000, 500)
    y = np.linspace(-500, 500, 500)
    X, Y = np.meshgrid(x, y)

    C_total = np.zeros_like(X)

    for source_x, source_y, source_z, Q in sources:
        dx = X - source_x  # 计算下风向距离

        # 使用Pasquill-Gifford参数化扩散系数
        sigma_y_x = sigma_y * (dx ** 0.91) + 1e-9
        sigma_z_x = sigma_z * (dx ** 0.92) + 1e-9

        C = (Q / (2 * np.pi * sigma_y_x * sigma_z_x * u)) * \
            np.exp(- ((Y - source_y) ** 2) / (2 * sigma_y_x ** 2)) * \
            np.exp(- ((monitor_height - source_z) ** 2) / (2 * sigma_z_x ** 2))

        C_total += C

    C_smoothed = gaussian_filter(C_total, sigma=1.5)

    fig = go.Figure(data=go.Heatmap(
        z=C_smoothed,
        x=X[0, :],
        y=Y[:, 0],
        colorscale='Viridis',
        zsmooth='best',
        zmin=np.percentile(C_smoothed, 5),  # 最小颜色值（5%分位数）
        zmax=np.percentile(C_smoothed, 99.8),  # 最大颜色值（99.8%分位数）
        colorbar=dict(title='浓度 (g/m³)'),
        opacity=0.8,
        showscale=True
    ))

    fig.update_layout(
        title=f'高斯烟羽模型 PM2.5 扩散模拟 (监测高度: {monitor_height}m)',
        xaxis_title='X 位置 (m)',
        yaxis_title='Y 位置 (m)',
        margin=dict(l=0, r=0, b=0, t=40),
        autosize=True
    )

    st.plotly_chart(fig, use_container_width=True)
