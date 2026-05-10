import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation
from typing import Any, Optional

class MobilityDashboard:
    """
    Visualization dashboard for the LTM Handover environment.
    """
    def __init__(self, bs_positions: np.ndarray):
        """
        bs_positions: (NBS, 2) array of BS/Sector coordinates.
        Expected NBS = 21 (7 BS x 3 sectors).
        """
        self.bs_positions = bs_positions
        self.num_sectors = bs_positions.shape[0]
        self.num_bs = self.num_sectors // 3
        
        # 7 Distinct colors for the 7 BS sites
        self.bs_colors = [
            '#1f77b4', # blue
            '#ff7f0e', # orange
            '#2ca02c', # green
            '#d62728', # red
            '#9467bd', # purple
            '#8c564b', # brown
            '#e377c2'  # pink
        ]

    def render_episode(self, history: dict[str, list[Any]], save_path: Optional[str] = None):
        """
        history: Dictionary containing 'ue_pos', 'serving_bs', 'rsrp'.
        """
        ue_positions = np.array(history['ue_pos'])
        serving_sector_history = np.array(history['serving_bs'])
        rsrp_history = np.array(history['rsrp']) # (T, 21)
        
        total_steps = len(ue_positions)
        
        fig, (ax_map, ax_rsrp) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.2]})
        
        # --- 1. SETUP MAP ---
        ax_map.set_title("LTM-HO Mobility Map")
        
        for i in range(self.num_bs):
            idx_start = i * 3
            ax_map.scatter(
                self.bs_positions[idx_start:idx_start+3, 0], 
                self.bs_positions[idx_start:idx_start+3, 1], 
                c=self.bs_colors[i], marker='^', s=80, label=f"BS {i}" if i==0 else "",
                alpha=0.8
            )
            cx = np.mean(self.bs_positions[idx_start:idx_start+3, 0])
            cy = np.mean(self.bs_positions[idx_start:idx_start+3, 1])
            ax_map.text(cx, cy + 25, f"BS {i}", fontsize=9, fontweight='bold', ha='center', color=self.bs_colors[i])

        ue_dot, = ax_map.plot([], [], 'ko', markersize=7, label="UE", zorder=10)
        conn_line, = ax_map.plot([], [], 'k-', linewidth=1.5, alpha=0.6, label="Serving Link", zorder=5)
        traj_line, = ax_map.plot([], [], 'k-', alpha=0.2, linewidth=1)
        
        ax_map.set_xlim(np.min(self.bs_positions[:, 0]) - 80, np.max(self.bs_positions[:, 0]) + 80)
        ax_map.set_ylim(np.min(self.bs_positions[:, 1]) - 80, np.max(self.bs_positions[:, 1]) + 80)
        ax_map.grid(True, linestyle=':', alpha=0.4)
        ax_map.set_aspect('equal')
        ax_map.legend(loc='upper right')

        # --- 2. SETUP RSRP PLOT ---
        ax_rsrp.set_title("Real-time RSRP (Strongest Sector per BS)")
        ax_rsrp.set_xlabel("Time Step")
        ax_rsrp.set_ylabel("RSRP (dBm)")
        ax_rsrp.set_ylim(-115, -40)
        
        # Data Structures for lines
        bg_lines = []     # Faint background lines
        active_lines = [] # Opaque segments where serving
        change_markers = [] 
        
        # Pre-process max RSRP per BS and sector change events
        bs_max_rsrp = np.zeros((total_steps, self.num_bs))
        # serving_mask[t, i] is True if BS i is serving at time t
        serving_mask = np.zeros((total_steps, self.num_bs), dtype=bool)
        sector_change_events = [[] for _ in range(self.num_bs)]
        
        last_sectors = [-1] * self.num_bs
        
        for t in range(total_steps):
            curr_serving_sect = serving_sector_history[t]
            curr_serving_bs = curr_serving_sect // 3 if curr_serving_sect != -1 else -1
            
            for b in range(self.num_bs):
                sects_rsrp = rsrp_history[t, b*3 : b*3+3]
                max_idx = np.argmax(sects_rsrp)
                max_val = sects_rsrp[max_idx]
                bs_max_rsrp[t, b] = max_val
                
                if curr_serving_bs == b:
                    serving_mask[t, b] = True
                
                global_sect_idx = b * 3 + max_idx
                if last_sectors[b] != -1 and global_sect_idx != last_sectors[b]:
                    sector_change_events[b].append((t, max_val))
                last_sectors[b] = global_sect_idx

        # Initialize plot objects
        for i in range(self.num_bs):
            # 1. Increased background opacity to 0.25
            l_bg, = ax_rsrp.plot([], [], color=self.bs_colors[i], linewidth=1.0, alpha=0.25)
            bg_lines.append(l_bg)
            
            # 2. Bold "Active" line (will use NaNs to show only serving segments)
            l_act, = ax_rsrp.plot([], [], color=self.bs_colors[i], linewidth=2.5, alpha=1.0, label=f"BS {i}")
            active_lines.append(l_act)
            
            # 3. Change markers (only visible on active segments)
            scat = ax_rsrp.scatter([], [], color=self.bs_colors[i], marker='x', s=40, zorder=4, alpha=0.8)
            change_markers.append(scat)

        ax_rsrp.legend(loc='upper right', fontsize='small', ncol=4)
        ax_rsrp.grid(True, linestyle=':', alpha=0.5)

        def animate(t):
            # --- Update Map ---
            ue_dot.set_data([ue_positions[t, 0]], [ue_positions[t, 1]])
            traj_line.set_data(ue_positions[:t+1, 0], ue_positions[:t+1, 1])
            
            curr_sect = serving_sector_history[t]
            active_bs_idx = -1
            if curr_sect != -1 and curr_sect < self.num_sectors:
                active_bs_idx = curr_sect // 3
                conn_line.set_color(self.bs_colors[active_bs_idx])
                conn_line.set_data(
                    [ue_positions[t, 0], self.bs_positions[curr_sect, 0]],
                    [ue_positions[t, 1], self.bs_positions[curr_sect, 1]]
                )
            else:
                conn_line.set_data([], [])
            
            # --- Update RSRP Lines ---
            x_vals = np.arange(t+1)
            for i in range(self.num_bs):
                # Update background
                bg_lines[i].set_data(x_vals, bs_max_rsrp[:t+1, i])
                
                # Update active segments (mask non-serving steps with NaN)
                active_y = bs_max_rsrp[:t+1, i].copy()
                active_y[~serving_mask[:t+1, i]] = np.nan
                active_lines[i].set_data(x_vals, active_y)
                
                # Update Markers (only if the change happened during a serving period)
                # To keep it clean, we only show markers for the BS if it's currently serving
                # or if the marker was on an active segment
                valid_events = []
                for ev in sector_change_events[i]:
                    if ev[0] <= t and serving_mask[ev[0], i]:
                        valid_events.append(ev)
                
                if valid_events:
                    change_markers[i].set_offsets(valid_events)
                else:
                    change_markers[i].set_offsets(np.empty((0, 2)))
            
            # FIXED WINDOW: t-480 to t+20
            ax_rsrp.set_xlim(t - 480, t + 20)
            
            return [ue_dot, conn_line, traj_line] + bg_lines + active_lines + change_markers

        num_frames = 500
        step_skip = max(1, total_steps // num_frames)
        
        ani = animation.FuncAnimation(
            fig, animate, frames=range(0, total_steps, step_skip), 
            interval=100, blit=True
        )
        
        if save_path:
            if save_path.endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=15, bitrate=1800)
                ani.save(save_path, writer=writer)
            else:
                ani.save(save_path, writer='pillow', fps=15)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

def get_bs_positions():
    """Official hexagonal cluster coordinates."""
    centers = np.array([
        [300.0, 300.0], [300.0, 500.0], [126.8, 400.0], [126.8, 200.0],
        [300.0, 100.0], [473.2, 200.0], [473.2, 400.0]
    ])
    bs_pos = []
    for bx, by in centers:
        for s in range(3):
            s_angle = np.deg2rad(90 + s * 120)
            bs_pos.append([bx + 15.0 * np.cos(s_angle), by + 15.0 * np.sin(s_angle)])
    return np.array(bs_pos)

if __name__ == "__main__":
    bs_pos = get_bs_positions()
    dash = MobilityDashboard(bs_pos)
    steps = 1000
    history = {
        'ue_pos': [[300 + t*0.2, 300 + t*0.1] for t in range(steps)],
        'serving_bs': [0 if t < 500 else 3 for t in range(steps)],
        'rsrp': np.random.uniform(-100, -60, (steps, 21)).tolist(),
    }
    os.makedirs("results/animations", exist_ok=True)
    dash.render_episode(history, save_path="results/animations/dashboard_video_test.mp4")
