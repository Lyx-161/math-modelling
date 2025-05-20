import streamlit as st
from lagrangian_particle_model import lagrangian_particle_model
from gaussian_plume_model import gaussian_plume_model
from eulerian_model import eulerian_model

from SPH_model import sph_process_model
from lagrange_process import lagrange_process_model
from MPM_model import MPM_process_model

# 运行方式：cd至此文件夹然后在终端中运行 streamlit run app.py

st.title("大气污染物扩散的粒子追踪")

model_type = st.selectbox(
    "选择模型",
    [
        "Gaussian Plume Model",
        "Lagrangian Particle Model",
        "Lagrangian Process Model",
        "Eulerian Model",
        "SPH Model",
        "MPM Model",
    ],
)

if model_type == "Lagrangian Particle Model":
    with st.sidebar.expander("污染物源设置", expanded=False):
        num_sources = st.slider("污染物源数量", 1, 5, 1)
        sources = []
        total_particles = 0
        for i in range(num_sources):
            col1, col2, col3 = st.columns(3)
            with col1:
                source_x = st.slider(f"源 {i+1} - X", -100, 100, 0)
            with col2:
                source_y = st.slider(f"源 {i+1} - Y", -100, 100, 0)
            with col3:
                source_z = st.slider(f"源 {i+1} - Z", 0, 200, 50)
            source_particles = st.slider(f"源 {i+1} - 排放粒子数量", 100, 5000, 1000)
            sources.append((source_x, source_y, source_z, source_particles))
            total_particles += source_particles  # 累加所有污染源的粒子数量

    with st.sidebar.expander("时间设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            time_steps = st.slider("时间步数", 10, 200, 100)
        with col2:
            dt = st.slider("时间步长 (秒)", 0.1, 5.0, 1.0, step=0.1)

    with st.sidebar.expander("高级扩散参数", expanded=False):
        wind_speed = st.slider("平均风速 (m/s)", 0.0, 10.0, 5.0, step=0.1)
        col1, col2 = st.columns(2)
        with col1:
            tau = st.slider("湍流时间尺度 (秒)", 10, 1000, 10)
        with col2:
            turbulence_intensity = st.slider("湍流强度 (m/s)", 0.0, 2.0, 0.5, step=0.1)

    with st.sidebar.expander("地面吸收设置", expanded=False):
        p_absorb = st.slider("地面吸收概率 p", 0.0, 1.0, 0.10, step=0.01)

        kx = st.slider("x 轴速度倍率 k_x", 0.0, 1.0, 0.8, step=0.01)
        ky = st.slider("y 轴扩散倍率 k_y", 0.0, 1.0, 0.8, step=0.01)
        kz = st.slider("z 轴扩散倍率 k_z", 0.0, 1.0, 0.9, step=0.01)

    enable_3d = st.sidebar.checkbox("启用三维扩散", value=False)
    lagrangian_particle_model(
        total_particles,
        time_steps,
        wind_speed,
        turbulence_intensity,
        dt,
        enable_3d,
        st,
        tau,
        sources,
        p_absorb,
        kx,
        ky,
        kz,
    )


if model_type == "Gaussian Plume Model":
    with st.sidebar.expander("污染物源设置", expanded=False):
        num_sources = st.slider("污染物源数量", 1, 5, 1)
        sources = []
        offset = -100  # 设置一个平移偏移量，确保所有污染源的坐标非负
        for i in range(num_sources):
            col1, col2, col3 = st.columns(3)
            with col1:
                source_x = st.slider(f"源 {i+1} - X 坐标", 0, 200, 0)
            with col2:
                source_y = st.slider(f"源 {i+1} - Y 坐标", -250, 250, 0)
            with col3:
                source_z = st.slider(f"源 {i+1} - Z 坐标", 0, 200, 50)
            source_strength = st.slider(f"源 {i+1} - 污染源强度 (g/s)", 10, 500, 100)
            sources.append((source_x, source_y, source_z, source_strength))

    with st.sidebar.expander("高级扩散参数", expanded=False):
        u = st.slider("平均风速 (m/s)", 1, 10, 5)
        col1, col2 = st.columns(2)
        with col1:
            sigma_y = st.slider("初始横向扩散系数", 0.0, 1.0, 0.08, step=0.01)
        with col2:
            sigma_z = st.slider("初始垂直扩散系数", 0.0, 1.0, 0.06, step=0.01)

    monitor_height = st.sidebar.slider("监测高度 (m)", 0, 200, 10)

    gaussian_plume_model(sources, u, sigma_y, sigma_z, st, monitor_height, offset)


if model_type == "Eulerian Model":  # 运行比较慢
    with st.sidebar.expander("污染物源设置", expanded=False):
        num_sources = st.slider("污染物源数量", 1, 5, 1)
        sources = []
        total_particles = 0
        for i in range(num_sources):
            col1, col2, col3 = st.columns(3)
            with col1:
                source_x = st.slider(f"源 {i+1} - X", -100, 100, 0)
            with col2:
                source_y = st.slider(f"源 {i+1} - Y", -250, 250, 0)
            with col3:
                source_z = st.slider(f"源 {i+1} - Z", 0, 200, 50)
            source_strength = st.slider(f"源 {i+1} - 污染源强度", 100, 5000, 1000)
            sources.append((source_x, source_y, source_z, source_strength))

    with st.sidebar.expander("时间设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            time_steps = st.slider("时间步数", 10, 200, 100)
        with col2:
            dt = st.slider("时间步长 (秒)", 0.1, 5.0, 1.0, step=0.1)

    with st.sidebar.expander("地面吸收及反弹设置", expanded=False):
        p_absorb = st.slider("地面吸收概率 p", 0.0, 1.0, 0.10, step=0.01)

        k_conc = st.slider("反弹后浓度倍率 k", 0.0, 1.0, 0.8, step=0.01)
        col1, col2, col3 = st.columns(3)
        with col1:
            kx = st.slider("x 轴速度倍率 k_x", 0.0, 1.0, 0.8, step=0.01)
        with col2:
            ky = st.slider("y 轴扩散倍率 k_y", 0.0, 1.0, 0.8, step=0.01)
        with col3:
            kz = st.slider("z 轴扩散倍率 k_z", 0.0, 1.0, 0.9, step=0.01)

    with st.sidebar.expander("高级扩散参数", expanded=False):
        u = st.slider("平均风速 (m/s)", 1, 10, 5)
        col1, col2 = st.columns(2)
        with col1:
            sigma_y = st.slider("初始横向扩散系数", 0.0, 1.0, 0.08, step=0.01)
        with col2:
            sigma_z = st.slider("初始垂直扩散系数", 0.0, 1.0, 0.06, step=0.01)

    monitor_height = st.sidebar.slider("监测高度 (m)", 0, 200, 10)

    offset = 100

    eulerian_model(
        sources,
        u,
        sigma_y,
        sigma_z,
        time_steps,
        dt,
        monitor_height,
        p_absorb,
        k_conc,
        kx,
        ky,
        kz,
        st,
    )

if model_type == "SPH Model":
    with st.sidebar.expander("污染物源设置", expanded=False):
        num_sources = st.slider("污染物源数量", 1, 5, 1)
        sources = []
        for i in range(num_sources):
            col1, col2, col3 = st.columns(3)
            with col1:
                source_x = st.slider(f"源 {i+1} - X", -100, 100, 0)
            with col2:
                source_y = st.slider(f"源 {i+1} - Y", -100, 100, -50)
            with col3:
                source_radius = st.slider(f"源 {i+1} -污染半径", 0.0, 1.0, 0.1)
            source_particles = st.slider(f"源 {i+1} - 排放粒子数量", 100, 5000, 1000)
            sources.append((source_x, source_y, source_radius, source_particles))
        # total_particles += source_particles
    with st.sidebar.expander("时间设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            time_steps = st.slider("时间步数", 10, 200, 100)
        with col2:
            dt = st.slider("时间步长 (秒)", 0.1, 5.0, 1.0, step=0.1)

    with st.sidebar.expander("SPH参数", expanded=False):
        dx = st.slider("粒子采样间距 (dx)", 0.01, 5.0, 0.4, step=0.01)
        h = st.slider("平滑长度 (h)", 0.01, 5.0, 1.0, step=0.01)
        mass = st.slider("粒子质量 (kg)", 0.01, 1.0, 0.01, step=0.01)

    with st.sidebar.expander("高级扩散参数", expanded=False):
        diff = st.slider("扩散系数 (m²/s)", 0.0, 10.0, 1.0, step=0.1)
        wind_speed = st.slider("平均风速 (m/s)", 0.0, 10.0, 5.0, step=0.1)
        vg = st.slider("沉降速度 (m/s)", 0.0, 10.0, 2.0, step=0.1)
        col1, col2 = st.columns(2)
        with col1:
            tau = st.slider("湍流时间尺度 (秒)", 10, 1000, 10)
        with col2:
            turbulence_intensity = st.slider("湍流强度 (m/s)", 0.0, 2.0, 0.5, step=0.1)

    monitor_height = st.sidebar.slider("监测高度 (m)", 0, 200, 10)

    sph_process_model(
        sources,
        time_steps,
        dt,
        dx,
        h,
        wind_speed,
        turbulence_intensity,
        st,
        tau,
        vg,
        diff,
        mass,
        monitor_height,
    )

if model_type == "Lagrangian Process Model":
    with st.sidebar.expander("污染物源设置", expanded=False):
        num_sources = st.slider("污染物源数量", 1, 5, 1)
        sources = []
        for i in range(num_sources):
            col1, col2, col3 = st.columns(3)
            with col1:
                source_x = st.slider(f"源 {i+1} - X", -100, 100, 0)
            with col2:
                source_y = st.slider(f"源 {i+1} - Y", -100, 100, -50)
            with col3:
                source_z = st.slider(f"源 {i+1} - Z", 0, 200, 50)
            source_particles = st.slider(f"源 {i+1} - 排放粒子数量", 100, 5000, 1000)
            sources.append((source_x, source_y, source_z, source_particles))

    with st.sidebar.expander("时间设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            time_steps = st.slider("时间步数", 10, 200, 100)
        with col2:
            dt = st.slider("时间步长 (秒)", 0.1, 5.0, 1.0, step=0.1)

    with st.sidebar.expander("高级扩散参数", expanded=False):
        diff = st.slider("扩散系数 (m²/s)", 0.0, 10.0, 1.0, step=0.1)
        wind_speed = st.slider("平均风速 (m/s)", 0.0, 10.0, 5.0, step=0.1)
        vg = st.slider("沉降速度 (m/s)", 0.0, 10.0, 2.0, step=0.1)
        col1, col2 = st.columns(2)
        with col1:
            tau = st.slider("湍流时间尺度 (秒)", 10, 1000, 10)
        with col2:
            turbulence_intensity = st.slider("湍流强度 (m/s)", 0.0, 2.0, 0.5, step=0.1)

    vmax = st.sidebar.slider("图例最大值", 100.0, 10000.0, 400.0, step=100.0)
    enable_3d = True
    monitor_height = st.sidebar.slider("监测高度 (m)", 0, 200, 40)

    lagrange_process_model(
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
        diff,
        monitor_height,
    )

if model_type == "MPM Model":
    with st.sidebar.expander("污染物源设置", expanded=False):
        num_sources = st.slider("污染物源数量", 1, 5, 1)
        sources = []
        for i in range(num_sources):
            col1, col2, col3 = st.columns(3)
            with col1:
                source_x = st.slider(f"源 {i+1} - X 坐标", 0.0, 1.0, 0.5)
            with col2:
                source_y = st.slider(f"源 {i+1} - Y 坐标", 0.0, 1.0, 0.5)
            with col3:
                source_radius = st.slider(f"源 {i+1} -污染半径", 0.0, 1.0, 0.1)
            source_particles = st.slider(f"源 {i+1} - 粒子数量", 100, 5000, 100)
            sources.append((source_x, source_y, source_radius, source_particles))

    with st.sidebar.expander("时间设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            time_steps = st.slider("时间步数", 10, 200, 100)
        with col2:
            dt = st.slider("时间步长 (秒)", 0.001, 0.1, 0.003, step=0.001)

    grid_dims = st.sidebar.slider("网格尺寸", 10, 100, 16)

    with st.sidebar.expander("高级扩散参数", expanded=False):
        wind_speed = st.slider("平均风速 (m/s)", 0.0, 50.0, 30.0, step=0.1)
        diff = st.sidebar.slider("扩散系数 (m^2/s)", 0.0, 1.0, 0.01, step=0.01)

    MPM_process_model(sources, grid_dims, time_steps, dt, wind_speed, st, diff)
