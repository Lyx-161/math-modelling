"""
It solves the convection-diffusion equality with Material Point Method

The equation is given by:

    ∂u/∂t + (v_x, v_y) · ∇u = D∇²u
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def bilinear_weights(x, y):
    """Compute 4 weights base on the relative position of (x, y) within the unit square."""
    wx = 1 - x
    wy = 1 - y
    # corersponding to (0, 0), (0, 1), (1, 0), (1, 1)
    return [wx * wy, wx * y, x * wy, x * y]


def p2g(position: np.ndarray, velocity: np.ndarray, grid_dims: int):
    """Transfer velocity field to a grid (vectorized implementation)."""
    momentum = np.zeros((grid_dims + 1, grid_dims + 1, 2))
    mass = np.zeros((grid_dims + 1, grid_dims + 1))
    particle_count = position.shape[0]
    
    # Compute grid cell indices for all particles
    gx = np.floor(position[:, 0] * grid_dims).astype(int)
    gy = np.floor(position[:, 1] * grid_dims).astype(int)
    
    # Filter out particles at the grid boundary
    valid_particles = (gx < grid_dims) & (gy < grid_dims)
    
    # Only process valid particles
    gx = gx[valid_particles]
    gy = gy[valid_particles]
    vel = velocity[valid_particles]
    
    # Compute relative positions within grid cells
    lx = position[valid_particles, 0] * grid_dims - gx
    ly = position[valid_particles, 1] * grid_dims - gy
    
    # Precompute weights
    wx0 = 1 - lx
    wx1 = lx
    wy0 = 1 - ly
    wy1 = ly
    
    # Compute the four weights for each particle
    w00 = wx0 * wy0
    w01 = wx0 * wy1
    w10 = wx1 * wy0
    w11 = wx1 * wy1
    
    # Accumulate mass and momentum using numpy indexing
    for i, (x, y) in enumerate(zip(gx, gy)):
        # Add mass contributions
        mass[x, y] += w00[i]
        mass[x, y+1] += w01[i]
        mass[x+1, y] += w10[i]
        mass[x+1, y+1] += w11[i]
        
        # Add momentum contributions
        momentum[x, y] += w00[i] * vel[i]
        momentum[x, y+1] += w01[i] * vel[i]
        momentum[x+1, y] += w10[i] * vel[i]
        momentum[x+1, y+1] += w11[i] * vel[i]
    
    # Compute velocities
    velocity_grid = np.zeros_like(momentum)
    non_zero_mass = mass > 1e-6
    velocity_grid[non_zero_mass, 0] = momentum[non_zero_mass, 0] / mass[non_zero_mass]
    velocity_grid[non_zero_mass, 1] = momentum[non_zero_mass, 1] / mass[non_zero_mass]
    
    return velocity_grid, mass, momentum

def g2p(grid: np.ndarray, position: np.ndarray, grid_dims: int):
    """Transfer velocity field to particles (vectorized implementation)."""
    particle_count = position.shape[0]
    velocity = np.zeros_like(position)
    
    # Compute grid cell indices for all particles
    gx = np.floor(position[:, 0] * grid_dims).astype(int)
    gy = np.floor(position[:, 1] * grid_dims).astype(int)
    
    # Filter out particles at the grid boundary
    valid_particles = (gx < grid_dims) & (gy < grid_dims)
    
    # Compute relative positions within grid cells
    lx = position[valid_particles, 0] * grid_dims - gx[valid_particles]
    ly = position[valid_particles, 1] * grid_dims - gy[valid_particles]
    
    # Precompute all weights
    wx0 = 1 - lx
    wx1 = lx
    wy0 = 1 - ly
    wy1 = ly
    
    # Get indices of valid particles
    valid_indices = np.where(valid_particles)[0]
    
    # Apply bilinear interpolation for all valid particles
    for i, idx in enumerate(valid_indices):
        g_x, g_y = gx[idx], gy[idx]
        
        # Apply weights for the four surrounding grid cells
        velocity[idx] += wx0[i] * wy0[i] * grid[g_x, g_y]
        velocity[idx] += wx0[i] * wy1[i] * grid[g_x, g_y + 1]
        velocity[idx] += wx1[i] * wy0[i] * grid[g_x + 1, g_y]
        velocity[idx] += wx1[i] * wy1[i] * grid[g_x + 1, g_y + 1]
    
    return velocity


def generate_randn(
    count: int,
    mean: float = 0.0,
    std_dev: float = 1.0,
) -> np.ndarray:
    """Generates random numbers from a normal distribution."""
    return np.random.normal(mean, std_dev, count)


class MaterialPointMethod:
    def __init__(
        self,
        grid_dims: int = 16,
        wind_vel_x: float = 30,
        wind_vel_y: float = 30,
        diffusion_coefficient: float = 0.01,
        dt: float = 0.003,
        initial_particle_count: int = 10000,
        source_particle_per_step: int = 100,
        source_pos_x: float = 0.5,
        source_pos_y: float = 0.5,
        source_radius: float = 0.1,
        velocity_dumping: float = 0.9999,
    ):
        self.wind_vel_x = wind_vel_x
        self.wind_vel_y = wind_vel_y
        self.diffusion_coefficient = diffusion_coefficient
        self.dt = dt
        self.grid_dims = grid_dims
        self.initial_particle_count = initial_particle_count
        self.source_particle_per_step = source_particle_per_step
        self.source_pos_x = source_pos_x
        self.source_pos_y = source_pos_y
        self.source_radius = source_radius

        self.particle = np.random.uniform(0, 1, (self.initial_particle_count, 2))
        self.particle_vel = np.zeros_like(self.particle)
        self.velocity_dumping = velocity_dumping
        self.mass_grid = np.zeros((grid_dims + 1, grid_dims + 1))
        self.momentum_grid = np.zeros((grid_dims + 1, grid_dims + 1, 2))
        self.velocity_grid = np.zeros((grid_dims + 1, grid_dims + 1, 2))

    def generate_random(self):
        pos_x = generate_randn(
            self.source_particle_per_step, self.source_pos_x, self.source_radius
        )
        pos_y = generate_randn(
            self.source_particle_per_step, self.source_pos_y, self.source_radius
        )
        return np.stack((pos_x, pos_y), axis=1)

    def step(self):
        self.advect()
        self.apply_wind()
        self.diffuse()
        self.dump_velocity()
        new_particles = self.generate_random()
        self.particle = np.concatenate((self.particle, new_particles), axis=0)
        self.particle_vel = np.concatenate(
            (self.particle_vel, np.zeros_like(new_particles)), axis=0
        )
        print(f"particle count: {self.particle.shape[0]}")

    def diffuse(self):
        # 1. discard all the particles out of [0, 1]^2
        in_range = np.all((self.particle >= 0) & (self.particle <= 1), axis=1)
        self.particle = self.particle[in_range]
        self.particle_vel = self.particle_vel[in_range]

        # # 1. p2g
        velocity, mass, momentum = p2g(self.particle, self.particle_vel, self.grid_dims)
        self.velocity_grid, self.mass_grid, self.momentum_grid = velocity, mass, momentum
        print(f"grid: {velocity.mean(axis=(0, 1))}")

        # # 2. diffuse the momentum on the grid
        # laplacian = np.zeros_like(velocity)
        # u_xx = np.diff(velocity[:, :, 0], axis=0, n=2)[:, 1:-1]
        # u_yy = np.diff(velocity[:, :, 0], axis=1, n=2)[1:-1, :]
        # laplacian[1:-1, 1:-1, 0] = u_xx + u_yy

        # v_xx = np.diff(velocity[:, :, 1], axis=0, n=2)[:, 1:-1]
        # v_yy = np.diff(velocity[:, :, 1], axis=1, n=2)[1:-1, :]
        # laplacian[1:-1, 1:-1, 1] = v_xx + v_yy

        # laplacian *= (
        #     self.diffusion_coefficient * self.dt / (1 / (self.grid_dims + 1)) ** 2
        # )
        # print(f"|laplace_u|: {np.linalg.norm(laplacian[:, :, 0])}")
        # print(f"|laplace_v|: {np.linalg.norm(laplacian[:, :, 1])}")

        # laplacian = np.diff(mass, axis=0, n=2)[:, 1:-1] + np.diff(mass, axis=1, n=2)[1:-1, :]
        # 3. add the laplacian to the grid
        coef = self.diffusion_coefficient * self.dt / (1 / (self.grid_dims + 1)) ** 2
        velocity[1:, :, 0] -= np.diff(mass, axis=0, n=1) * coef
        velocity[:, 1:, 1] -= np.diff(mass, axis=1, n=1) * coef
        # 4. g2p
        self.particle_vel = g2p(velocity, self.particle, self.grid_dims)

    def dump_velocity(self):
        self.particle_vel *= self.velocity_dumping  # Apply velocity dumping

    def advect(self):
        self.particle += self.particle_vel * self.dt

    def apply_wind(self):
        self.particle_vel[:, 0] += self.wind_vel_x * self.dt
        self.particle_vel[:, 1] += self.wind_vel_y * self.dt
        print(f"velocity: {self.particle_vel.mean(axis=0)}")

if __name__ == "__main__":
    mpm = MaterialPointMethod()
    
    # Create a better looking figure with improved styling
    plt.style.use('dark_background')  # Dark theme looks better for fluid simulations
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), dpi=100)
    fig.suptitle("Material Point Method Simulation", fontsize=16)
    
    # Create colorbar for particles only once
    particle_scatter = ax[0].scatter([], [], s=2, cmap="plasma", alpha=0.7)
    particle_cbar = fig.colorbar(particle_scatter, ax=ax[0], label="Particle Velocity")
    
    # Create colorbar for mass grid only once
    mass_img = ax[1].imshow(
        np.zeros((mpm.grid_dims+1, mpm.grid_dims+1)).T,
        origin='lower', extent=[0, 1, 0, 1],
        cmap='inferno', vmin=0, interpolation='bilinear'
    )
    mass_cbar = fig.colorbar(mass_img, ax=ax[1], label="Mass Density")
    
    # Animation loop
    def update(frame):
        mpm.step()
        
        # Update particle plot
        ax[0].clear()
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)
        ax[0].set_title(f"Particles (Frame {frame + 1})", fontsize=12)
        ax[0].set_xlabel("X Position")
        ax[0].set_ylabel("Y Position")
        
        # Calculate velocity magnitude for coloring
        vel_mag = np.linalg.norm(mpm.particle_vel, axis=1)
        
        # Plot particles with better visual appearance
        scatter = ax[0].scatter(
            mpm.particle[:, 0],
            mpm.particle[:, 1],
            s=3,
            c=vel_mag,
            cmap="plasma",
            alpha=0.7,
            edgecolors=None
        )
        
        # Update colorbar limits based on current velocity range
        if len(vel_mag) > 0:
            particle_cbar.mappable.set_clim(0, max(0.01, np.percentile(vel_mag, 95)))
        
        # Update mass grid visualization
        ax[1].clear()
        ax[1].set_title("Mass Density Grid", fontsize=12)
        ax[1].set_xlabel("X Position")
        ax[1].set_ylabel("Y Position")
        
        # Plot mass grid with better interpolation
        im = ax[1].imshow(
            mpm.mass_grid.T,  # Transpose for correct orientation
            origin='lower',
            extent=[0, 1, 0, 1],
            cmap='inferno',
            vmin=0,
            vmax=max(0.01, np.percentile(mpm.mass_grid, 95)),  # Dynamic range
            interpolation='bilinear'
        )
        
        # Add timestamp
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        fig.suptitle(f"Material Point Method Simulation - Time: {frame*mpm.dt:.3f}s", fontsize=16)
        
        return scatter, im

    ani = FuncAnimation(fig, update, frames=1000, interval=20, blit=False)
    plt.tight_layout()
    plt.show()
