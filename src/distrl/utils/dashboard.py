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
        """
        self.bs_positions = bs_positions
        self.num_bs = bs_positions.shape[0]
        
    def render_episode(self, history: dict[str, list[Any]], save_path: Optional[str] = None):
        """
        history: Dictionary containing 'ue_pos', 'serving_bs', 'rsrp', 'handovers'.
        """
        ue_positions = np.array(history['ue_pos'])
        serving_bs = history['serving_bs']
        rsrp_history = np.array(history['rsrp'])
        
        fig, (ax_map, ax_rsrp) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1]})
        
        # 1. Setup Map
        ax_map.set_title("LTM-HO Mobility Map")
        ax_map.scatter(self.bs_positions[:, 0], self.bs_positions[:, 1], c='red', marker='^', s=100, label="BS/Sector")
        
        for i in range(0, self.num_bs, 3):
            ax_map.text(self.bs_positions[i, 0], self.bs_positions[i, 1] + 10, f"BS {i//3}", fontsize=10, fontweight='bold', ha='center')
        
        ue_dot, = ax_map.plot([], [], 'bo', markersize=8, label="UE")
        conn_line, = ax_map.plot([], [], 'g-', linewidth=2, alpha=0.8, label="Serving Link")
        traj_line, = ax_map.plot([], [], 'b-', alpha=0.3)
        
        ax_map.set_xlim(np.min(self.bs_positions[:, 0]) - 100, np.max(self.bs_positions[:, 0]) + 100)
        ax_map.set_ylim(np.min(self.bs_positions[:, 1]) - 100, np.max(self.bs_positions[:, 1]) + 100)
        ax_map.legend(loc='upper right')
        ax_map.grid(True, linestyle=':', alpha=0.5)

        # 2. Setup RSRP Plot
        ax_rsrp.set_title("Real-time RSRP Levels (Top 8 Strongest Sectors)")
        ax_rsrp.set_xlabel("Time Step")
        ax_rsrp.set_ylabel("RSRP (dBm)")
        ax_rsrp.set_ylim(-120, -30)
        
        # Identify the 8 strongest sectors on average to plot
        avg_rsrp = np.mean(rsrp_history, axis=0)
        top_indices = np.argsort(avg_rsrp)[-8:] # Plot the 8 strongest sectors
        
        rsrp_lines = []
        for i in top_indices:
            line, = ax_rsrp.plot([], [], label=f"Sect {i}")
            rsrp_lines.append(line)
        ax_rsrp.legend(loc='lower right', fontsize='small', ncol=2)
        ax_rsrp.grid(True, linestyle=':', alpha=0.5)

        def animate(t):
            # Update Map
            ue_dot.set_data([ue_positions[t, 0]], [ue_positions[t, 1]])
            traj_line.set_data(ue_positions[:t+1, 0], ue_positions[:t+1, 1])
            
            curr_bs = serving_bs[t]
            if curr_bs != -1 and curr_bs < self.num_bs:
                conn_line.set_data(
                    [ue_positions[t, 0], self.bs_positions[curr_bs, 0]],
                    [ue_positions[t, 1], self.bs_positions[curr_bs, 1]]
                )
            else:
                conn_line.set_data([], [])
            
            # Update RSRP
            for idx, i in enumerate(top_indices):
                rsrp_lines[idx].set_data(range(t+1), rsrp_history[:t+1, i])
            
            # WIDER X-AXIS: Show 500 steps of history
            ax_rsrp.set_xlim(max(0, t - 500), t + 20)
            
            return [ue_dot, conn_line, traj_line] + rsrp_lines

        total_frames = len(ue_positions)
        step_skip = max(1, total_frames // 250)
        
        ani = animation.FuncAnimation(fig, animate, frames=range(0, total_frames, step_skip), interval=40, blit=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=20)
            else:
                ani.save(save_path, writer='ffmpeg', fps=20)
            print(f"Dashboard animation saved to {save_path}")
        else:
            plt.show()

def get_mock_bs_positions(n_bs=7):
    bs_pos = []
    for i in range(n_bs):
        angle = 2 * np.pi * i / (n_bs-1) if i > 0 else 0
        r = 300 if i > 0 else 0
        bx, by = r * np.cos(angle), r * np.sin(angle)
        for s in range(3):
            s_angle = 2 * np.pi * s / 3
            offset = 15
            bs_pos.append([bx + offset * np.cos(s_angle), by + offset * np.sin(s_angle)])
    return np.array(bs_pos)

if __name__ == "__main__":
    bs_pos = get_mock_bs_positions()
    dash = MobilityDashboard(bs_pos)
    steps = 100
    history = {
        'ue_pos': [[t*2, t*0.5] for t in range(steps)],
        'serving_bs': [0 if t < 50 else 1 for t in range(steps)],
        'rsrp': [[-50 - (t-0)**2*0.01, -100 + (t-50)**2*0.01] + [-110]*19 for t in range(steps)],
    }
    os.makedirs("results/animations", exist_ok=True)
    dash.render_episode(history, save_path="results/animations/dashboard_test.gif")
