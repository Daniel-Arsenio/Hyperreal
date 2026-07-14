from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    num_agents: int
    num_rays: int = 16
    world_size: float = 10.0
    agent_radius: float = 0.2
    dt: float = 0.1
    max_speed: float = 2.0
    max_accel: float = 4.0
    max_angular_vel: float = 4.0
    lidar_range: float = 3.0
