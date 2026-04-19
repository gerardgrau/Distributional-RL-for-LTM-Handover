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
        bs_positions: (NBS, 2) array of BS coordinates.
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
        ax_map.scatter(self.bs_positions[:, 0], self.bs_positions[:, 1], c='red', marker='^', s=100, label="BS")
        for i in range(self.num_bs):
            ax_map.text(self.bs_positions[i, 0], self.bs_positions[i, 1] + 5, f"BS {i}", fontsize=9, ha='center')
        
        ue_dot, = ax_map.plot([], [], 'bo', label="UE")
        conn_line, = ax_map.plot([], [], 'g--', alpha=0.6, label="Serving Link")
        traj_line, = ax_map.plot([], [], 'b-', alpha=0.3)
        
        ax_map.set_xlim(np.min(self.bs_positions[:, 0]) - 50, np.max(self.bs_positions[:, 0]) + 50)
        ax_map.set_ylim(np.min(self.bs_positions[:, 1]) - 50, np.max(self.bs_positions[:, 1]) + 50)
        ax_map.legend()

        # 2. Setup RSRP Plot
        ax_rsrp.set_title("Real-time RSRP Levels")
        ax_rsrp.set_xlabel("Time Step")
        ax_rsrp.set_ylabel("RSRP (dBm)")
        ax_rsrp.set_ylim(-120, -30)
        
        rsrp_lines = []
        # Only plot top few BS to avoid clutter
        for i in range(min(5, self.num_bs)):
            line, = ax_rsrp.plot([], [], label=f"BS {i}")
            rsrp_lines.append(line)
        ax_rsrp.legend(loc='lower right')

        def animate(t):
            # Update Map
            ue_dot.set_data([ue_positions[t, 0]], [ue_positions[t, 1]])
            traj_line.set_data(ue_positions[:t+1, 0], ue_positions[:t+1, 1])
            
            curr_bs = serving_bs[t]
            if curr_bs != -1:
                conn_line.set_data(
                    [ue_positions[t, 0], self.bs_positions[curr_bs, 0]],
                    [ue_positions[t, 1], self.bs_positions[curr_bs, 1]]
                )
            
            # Update RSRP
            for i, line in enumerate(rsrp_lines):
                line.set_data(range(t+1), rsrp_history[:t+1, i])
            ax_rsrp.set_xlim(max(0, t - 100), t + 5)
            
            return [ue_dot, conn_line, traj_line] + rsrp_lines

        ani = animation.FuncAnimation(fig, animate, frames=len(ue_positions), interval=50, blit=True)
        
        if save_path:
            # Requires ffmpeg for mp4, or can save as gif
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow')
            else:
                ani.save(save_path, writer='ffmpeg')
            print(f"Dashboard animation saved to {save_path}")
        else:
            plt.show()

# Mock BS positions for testing (can be improved by reading SUMO/Environment data)
def get_mock_bs_positions(n=7):
    # Hexagonal grid approximate
    pos = []
    for i in range(n):
        angle = 2 * np.pi * i / (n-1) if i > 0 else 0
        r = 200 if i > 0 else 0
        pos.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(pos)

if __name__ == "__main__":
    # Test visualization with dummy data
    bs_pos = get_mock_bs_positions()
    dash = MobilityDashboard(bs_pos)
    
    steps = 100
    history = {
        'ue_pos': [[t*2, t*0.5] for t in range(steps)],
        'serving_bs': [0 if t < 50 else 1 for t in range(steps)],
        'rsrp': [[-50 - (t-0)**2*0.01, -100 + (t-50)**2*0.01] + [-110]*5 for t in range(steps)],
    }
    
    # Save a test gif
    os.makedirs("results/tests", exist_ok=True)
    dash.render_episode(history, save_path="results/tests/dashboard_test.gif")
