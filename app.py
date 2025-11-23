import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Optional, Dict
import statistics
st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */
    .algorithm-title {
        font-size: 32px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
        color: #4f46e5;
    }

    .metric-card, .stat-box {
        background: #111827;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        color: white;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.25);
        border: 1px solid #1f2937;
    }

    .metric-card h2, .metric-card h3 {
        margin: 0;
        color: #a5b4fc;
    }

    .stat-box h3 {
        color: #a5b4fc;
        margin: 0;
    }

    .stat-box p {
        color: #9ca3af;
        margin: 6px 0 0 0;
    }

    /* ---------- CLEAN CHESSBOARD ---------- */
    .chessboard-wrapper {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }

    .chessboard {
        border: 3px solid #374151;
        box-shadow: 0 0 10px rgba(0,0,0,0.25);
    }

    .chessboard table {
        border-collapse: collapse;
    }

    .chessboard td {
        width: 48px;
        height: 48px;
        text-align: center;
        vertical-align: middle;
        font-size: 30px;
        font-weight: bold;
    }

    .white-cell {
        background: #f3f4f6;
    }

    .black-cell {
        background: #9ca3af;
    }

    .queen {
        color: #dc2626; /* red queen for visibility */
        text-shadow: 0 0 6px rgba(0,0,0,0.3);
    }

    .board-title {
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #374151;
    }

</style>
""", unsafe_allow_html=True)

# Configure page
st.set_page_config(
    page_title="8-Queens AI Solver Comparison",
    page_icon="♛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for smooth animations
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        margin-bottom: 2rem;
        animation: slideIn 0.8s ease-out;
    }
    
    .chess-square {
        width: 60px;
        height: 60px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .chess-square.light {
        background: linear-gradient(135deg, #f0d9b5 0%, #e8d1a7 100%);
    }
    
    .chess-square.dark {
        background: linear-gradient(135deg, #b58863 0%, #a07855 100%);
    }
    
    .chess-square.has-queen {
        animation: fadeIn 0.4s ease-out;
    }
    
    .chess-square:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        z-index: 10;
    }
    
    .queen-piece {
        filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));
        transition: all 0.3s ease;
    }
    
    .chess-square:hover .queen-piece {
        transform: rotate(10deg) scale(1.1);
    }
    
    .chessboard-container {
        display: inline-block;
        border: 4px solid #764ba2;
        border-radius: 12px;
        padding: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: fadeIn 0.6s ease-out;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        animation: slideIn 0.5s ease-out;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    .algorithm-title {
        font-size: 2rem;
        font-weight: bold;
        color: #764ba2;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #667eea;
        margin-bottom: 1rem;
        animation: slideIn 0.6s ease-out;
    }
    
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: slideIn 0.4s ease-out;
    }
    
    .attack-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    .attack-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    .attack-medium {
        background: linear-gradient(135deg, #ffd93d 0%, #f9ca24 100%);
        color: #333;
    }
    
    .attack-low {
        background: linear-gradient(135deg, #6bcf7f 0%, #51cf66 100%);
        color: white;
    }
    
    .attack-zero {
        background: linear-gradient(135deg, #4dabf7 0%, #339af0 100%);
        color: white;
    }
    
    .progress-container {
        background: #f1f3f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CORE 8-QUEENS CLASSES
# ============================================================================

class EightQueens:
    """Base class for 8-Queens problem."""
    
    def __init__(self, n: int = 8):
        self.n = n
    
    def count_attacks(self, state: List[int]) -> int:
        """Count number of attacking pairs of queens."""
        attacks = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j]:
                    attacks += 1
                elif abs(state[i] - state[j]) == abs(i - j):
                    attacks += 1
        return attacks
    
    def generate_random_state(self) -> List[int]:
        """Generate random initial state."""
        return list(np.random.randint(0, self.n, self.n))
    
    def is_solution(self, state: List[int]) -> bool:
        """Check if state is a solution."""
        return self.count_attacks(state) == 0


class HillClimbingSolver(EightQueens):
    """Hill Climbing solver with random restart."""
    
    def __init__(self, n: int = 8, max_restarts: int = 100, max_iterations: int = 1000):
        super().__init__(n)
        self.max_restarts = max_restarts
        self.max_iterations = max_iterations
        self.history = []
        self.attack_history = []
    
    def get_neighbors(self, state: List[int]) -> List[List[int]]:
        """Generate all neighbors."""
        neighbors = []
        for col in range(self.n):
            for row in range(self.n):
                if state[col] != row:
                    neighbor = state.copy()
                    neighbor[col] = row
                    neighbors.append(neighbor)
        return neighbors
    
    def hill_climb_single(self, initial_state: List[int]) -> Tuple[List[int], int, bool]:
        """Single hill climbing attempt."""
        current = initial_state.copy()
        iterations = 0
        self.history = [current.copy()]
        self.attack_history = [self.count_attacks(current)]
        
        while iterations < self.max_iterations:
            iterations += 1
            current_attacks = self.count_attacks(current)
            
            if current_attacks == 0:
                return current, iterations, True
            
            neighbors = self.get_neighbors(current)
            best_neighbor = None
            best_attacks = current_attacks
            
            for neighbor in neighbors:
                neighbor_attacks = self.count_attacks(neighbor)
                if neighbor_attacks < best_attacks:
                    best_attacks = neighbor_attacks
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                return current, iterations, False
            
            current = best_neighbor
            self.history.append(current.copy())
            self.attack_history.append(self.count_attacks(current))
        
        return current, iterations, False
    
    def solve(self, initial_state: Optional[List[int]] = None) -> dict:
        """Solve with random restart."""
        start_time = time.time()
        total_iterations = 0
        restarts = 0
        all_histories = []
        
        if initial_state is None:
            initial_state = self.generate_random_state()
        
        current_state = initial_state.copy()
        original_state = initial_state.copy()
        
        for restart in range(self.max_restarts):
            restarts = restart + 1
            solution, iterations, success = self.hill_climb_single(current_state)
            total_iterations += iterations
            all_histories.append({
                'restart': restart,
                'history': self.history.copy(),
                'attacks': self.attack_history.copy()
            })
            
            if success:
                return {
                    'method': 'Hill Climbing',
                    'solution': solution,
                    'success': True,
                    'restarts': restarts,
                    'iterations': total_iterations,
                    'runtime': time.time() - start_time,
                    'history': self.history,
                    'attack_history': self.attack_history,
                    'all_histories': all_histories,
                    'initial_state': original_state
                }
            
            current_state = self.generate_random_state()
        
        return {
            'method': 'Hill Climbing',
            'solution': solution,
            'success': False,
            'restarts': restarts,
            'iterations': total_iterations,
            'runtime': time.time() - start_time,
            'history': self.history,
            'attack_history': self.attack_history,
            'all_histories': all_histories,
            'initial_state': original_state
        }


class CSPBacktrackingSolver(EightQueens):
    """CSP with basic backtracking."""
    
    def __init__(self, n: int = 8):
        super().__init__(n)
        self.iterations = 0
        self.history = []
        self.attack_history = []
    
    def is_safe(self, state: List[int], col: int, row: int) -> bool:
        """Check if placement is safe."""
        for c in range(col):
            if state[c] == row or abs(state[c] - row) == abs(c - col):
                return False
        return True
    
    def backtrack(self, state: List[int], col: int) -> bool:
        """Backtracking algorithm."""
        self.iterations += 1
        current_state = state.copy()
        display_state = [s if s != -1 else 0 for s in current_state]
        self.history.append(display_state)
        self.attack_history.append(self.count_attacks(display_state))
        
        if col >= self.n:
            return True
        
        for row in range(self.n):
            if self.is_safe(state, col, row):
                state[col] = row
                if self.backtrack(state, col + 1):
                    return True
                state[col] = -1
        
        return False
    
    def solve(self, initial_state: Optional[List[int]] = None) -> dict:
        """Solve using backtracking."""
        start_time = time.time()
        self.iterations = 0
        self.history = []
        self.attack_history = []
        
        state = [-1] * self.n
        success = self.backtrack(state, 0)
        
        return {
            'method': 'CSP Backtracking',
            'solution': state if success else None,
            'success': success,
            'iterations': self.iterations,
            'runtime': time.time() - start_time,
            'history': self.history,
            'attack_history': self.attack_history,
            'initial_state': initial_state
        }


class CSPEnhancedSolver(EightQueens):
    """Enhanced CSP with Forward Checking and MRV."""
    
    def __init__(self, n: int = 8):
        super().__init__(n)
        self.iterations = 0
        self.history = []
        self.attack_history = []
    
    def forward_check(self, domains: List[set], col: int, row: int) -> List[set]:
        """Forward checking."""
        new_domains = [d.copy() for d in domains]
        
        for c in range(col + 1, self.n):
            new_domains[c].discard(row)
            diff = c - col
            new_domains[c].discard(row + diff)
            new_domains[c].discard(row - diff)
            
            if len(new_domains[c]) == 0:
                return None
        
        return new_domains
    
    def select_mrv_variable(self, state: List[int], domains: List[set]) -> int:
        """MRV heuristic."""
        min_values = float('inf')
        mrv_col = -1
        
        for col in range(self.n):
            if state[col] == -1:
                if len(domains[col]) < min_values:
                    min_values = len(domains[col])
                    mrv_col = col
        
        return mrv_col
    
    def backtrack_fc(self, state: List[int], domains: List[set]) -> bool:
        """Backtracking with FC and MRV."""
        self.iterations += 1
        display_state = [s if s != -1 else 0 for s in state]
        self.history.append(display_state)
        self.attack_history.append(self.count_attacks(display_state))
        
        if -1 not in state:
            return True
        
        col = self.select_mrv_variable(state, domains)
        if col == -1:
            return True
        
        for row in list(domains[col]):
            state[col] = row
            new_domains = self.forward_check(domains, col, row)
            
            if new_domains is not None:
                if self.backtrack_fc(state, new_domains):
                    return True
            
            state[col] = -1
        
        return False
    
    def solve(self, initial_state: Optional[List[int]] = None) -> dict:
        """Solve using enhanced CSP."""
        start_time = time.time()
        self.iterations = 0
        self.history = []
        self.attack_history = []
        
        state = [-1] * self.n
        domains = [set(range(self.n)) for _ in range(self.n)]
        success = self.backtrack_fc(state, domains)
        
        return {
            'method': 'CSP Enhanced',
            'solution': state if success else None,
            'success': success,
            'iterations': self.iterations,
            'runtime': time.time() - start_time,
            'history': self.history,
            'attack_history': self.attack_history,
            'initial_state': initial_state
        }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_html_chessboard(state: List[int], title: str = "Chessboard", attacks: int = None, 
                          highlight_col: int = -1) -> str:
    """Create HTML/CSS animated chessboard."""
    n = len(state)
    
    # Determine attack level for styling
    attack_class = "attack-zero"
    if attacks is not None:
        if attacks == 0:
            attack_class = "attack-zero"
        elif attacks <= 2:
            attack_class = "attack-low"
        elif attacks <= 5:
            attack_class = "attack-medium"
        else:
            attack_class = "attack-high"
    
    # Build board HTML more carefully
    board_rows = []
    for row in range(n):
        row_squares = []
        for col in range(n):
            is_light = (row + col) % 2 == 0
            square_class = "light" if is_light else "dark"
            has_queen = state[col] == (n - 1 - row)
            
            highlight = "border: 3px solid #ffd93d;" if col == highlight_col else ""
            queen_symbol = "♛" if has_queen else ""
            has_queen_class = "has-queen" if has_queen else ""
            
            square_html = f'<div class="chess-square {square_class} {has_queen_class}" style="{highlight}"><span class="queen-piece">{queen_symbol}</span></div>'
            row_squares.append(square_html)
        
        row_html = '<div style="display: flex; line-height: 0;">' + ''.join(row_squares) + '</div>'
        board_rows.append(row_html)
    
    board_content = ''.join(board_rows)
    
    attack_html = ""
    if attacks is not None:
        attack_html = f'<div class="attack-indicator {attack_class}">⚔️ Attacks: {attacks}</div>'
    
    final_html = f"""
<div style="text-align: center; margin: 20px 0;">
    <h3 style="color: #764ba2; margin-bottom: 10px;">{title}</h3>
    {attack_html}
    <div style="margin-top: 15px;">
        <div class="chessboard-container">
            <div style="background: white; border-radius: 8px; padding: 4px;">
                {board_content}
            </div>
        </div>
    </div>
</div>
"""
    
    return final_html


def create_convergence_plot(attack_history: List[int], title: str = "Convergence", 
                           current_idx: int = -1) -> go.Figure:
    """Create animated convergence plot."""
    fig = go.Figure()
    
    # Full history
    fig.add_trace(go.Scatter(
        y=attack_history,
        mode='lines',
        line=dict(color='#e0e0e0', width=1),
        name='Full Path',
        showlegend=False
    ))
          
    # Current progress
    if current_idx > 0:
        fig.add_trace(go.Scatter(
            y=attack_history[:current_idx+1],
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6, color='#764ba2'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)',
            name='Progress'
        ))
        
        # Current point
        fig.add_trace(go.Scatter(
            x=[current_idx],
            y=[attack_history[current_idx]],
            mode='markers',
            marker=dict(size=15, color='#ff6b6b', symbol='star'),
            name='Current',
            showlegend=False
        ))
    else:
        fig.add_trace(go.Scatter(
            y=attack_history,
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6, color='#764ba2'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)',
            name='Progress'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16, color='#764ba2')),
        xaxis_title='Iteration',
        yaxis_title='Attacking Pairs',
        height=250,
        hovermode='x unified',
        plot_bgcolor='#f8f9fa',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def create_comparison_chart(results: Dict) -> go.Figure:
    """Create comparison bar chart."""
    methods = list(results.keys())
    
    fig = go.Figure()
    
    colors = ['#667eea', '#764ba2', '#f093fb']
    
    fig.add_trace(go.Bar(
        name='Iterations',
        x=methods,
        y=[results[m]['iterations'] for m in methods],
        marker_color=colors[0],
        text=[results[m]['iterations'] for m in methods],
        texttemplate='%{text}',
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Iterations: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Algorithm Performance Comparison',
        xaxis_title='Method',
        yaxis_title='Iterations',
        height=350,
        hovermode='x unified',
        plot_bgcolor='#f8f9fa'
    )
    
    return fig


def create_performance_chart(results: Dict) -> go.Figure:
    """Create performance metrics chart."""
    methods = list(results.keys())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Runtime (ms)',
        x=methods,
        y=[results[m]['runtime'] * 1000 for m in methods],
        marker=dict(
            color=[results[m]['runtime'] * 1000 for m in methods],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="ms")
        ),
        text=[f"{results[m]['runtime'] * 1000:.2f}" for m in methods],
        texttemplate='%{text} ms',
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Runtime: %{y:.2f} ms<extra></extra>'
    ))
    
    fig.update_layout(
        title='Runtime Performance',
        xaxis_title='Method',
        yaxis_title='Runtime (milliseconds)',
        height=350,
        hovermode='x unified',
        plot_bgcolor='#f8f9fa'
    )
    
    return fig


# ============================================================================
# PAGE VIEWS
# ============================================================================

def show_hill_climbing_page():
    """Hill Climbing algorithm page with smooth animation, restructured layout, and dynamic custom state input boxes."""
    
    # Heading using standard markdown (no custom CSS injection)
    st.markdown('## ⛰️ Hill Climbing with Random Restart')
    
    # --- 1. Top Configuration & Controls ---
    st.markdown("### ⚙️ Configuration and Controls")
    
    param_tab, init_state_tab = st.tabs(["Algorithm Parameters", "Initial State Setup"])
    
    with param_tab:
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            board_size = st.slider("🎲 Board Size (N)", 4, 12, 8, key="hc_size")
            max_restarts = st.slider("🔄 Max Restarts", 10, 100, 50, key="hc_restarts", 
                                     help="Maximum times to restart from a new random state if stuck.")
            
        with config_col2:
            max_iterations = st.slider("🔢 Max Iterations per Run", 100, 1000, 500, key="hc_iters")
            animation_speed = st.slider("⚡ Animation Speed (s)", 0.05, 1.0, 0.2, 
                                        help="Seconds between visualized moves.", key="hc_speed")
            
    with init_state_tab:
        initial_state_mode = st.radio(
            "Initial State Mode", 
            ["Random Initial State", "Custom Initial State"], 
            index=0,
            key="hc_init_mode", 
            horizontal=True
        )
        
        # Initialize default custom state in session_state if it doesn't exist
        if "hc_custom_state_list" not in st.session_state or len(st.session_state.hc_custom_state_list) != board_size:
            st.session_state.hc_custom_state_list = list(range(board_size)) 

        if initial_state_mode == "Custom Initial State":
            st.markdown(f"**Enter Row Index (0 to {board_size-1}) for each of the {board_size} columns:**")
            
            num_cols_display = min(board_size, 12) 
            cols = st.columns(num_cols_display)
            parsed_state = []
            
            for col_idx in range(board_size):
                with cols[col_idx % num_cols_display]: 
                    default_value = st.session_state.hc_custom_state_list[col_idx] if col_idx < len(st.session_state.hc_custom_state_list) else 0

                    queen_row = st.number_input(
                        f"Col {col_idx}", 
                        min_value=0, max_value=board_size - 1, 
                        value=default_value, 
                        step=1, key=f"hc_custom_q_{col_idx}",
                        label_visibility="visible"
                    )
                    parsed_state.append(queen_row)
            
            st.session_state.hc_custom_state_list = parsed_state
        
    st.markdown("---")

    # --- 2. Run Button (Standalone) ---
    if "hc_running" not in st.session_state:
        st.session_state["hc_running"] = False
        
    # Button uses Streamlit's primary color
    if st.button("🚀 Run Hill Climbing", key="run_hc", use_container_width=True, type="primary"): 
        st.session_state["hc_running"] = True
        st.rerun()

    st.markdown("---")

    # --- 3. Visualization and Results (Bottom Section) ---
    
    if st.session_state.hc_running:
        st.markdown("### 🏃 Algorithm Execution")
        
        # --- Pre-run Setup ---
        initial_state = None
        # Assuming HillClimbingSolver, count_attacks are defined
        solver = HillClimbingSolver(board_size, max_restarts, max_iterations) 

        if initial_state_mode == "Random Initial State":
            initial_state = solver.generate_random_state()
        else: # Custom Initial State
            initial_state = st.session_state.hc_custom_state_list
            if len(initial_state) != board_size:
                st.error("Internal error: Custom state size mismatch.")
                st.session_state.hc_running = False
                st.stop()
        
        st.session_state.hc_initial_state_final = initial_state
        initial_attacks = solver.count_attacks(initial_state)

        # --- Visualization Containers ---
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1: iter_metric = st.empty()
        with stats_col2: attack_metric = st.empty()
        with stats_col3: restart_metric = st.empty()
        
        board_col, progress_col = st.columns([1, 1])
        with board_col: board_placeholder = st.empty() 
        with progress_col: 
            progress_placeholder = st.empty()
            chart_placeholder = st.empty() 
            
        # Show initial state in the animation area before starting
        with board_placeholder.container():
            st.markdown(
                create_html_chessboard(initial_state, "🎬 Initial State", initial_attacks),
                unsafe_allow_html=True
            )
            
        iter_metric.metric("Iteration", "0")
        attack_metric.metric("Attacking Pairs", initial_attacks)
        restart_metric.metric("Restarts", "0")
        
        time.sleep(1)
        
        # Run algorithm
        with st.spinner('🔄 Computing solution...'):
            result = solver.solve(initial_state) 
        
        # --- Animation Loop ---
        history = result['history']
        attack_history = result['attack_history']
        
        progress_bar = progress_placeholder.progress(0)
        step = max(1, len(history) // 50)
        
        for i in range(0, len(history), step):
            state = history[i]
            attacks = attack_history[i]
            
            board_placeholder.markdown(
                create_html_chessboard(state, f"🎯 Iteration {i+1}/{len(history)}", attacks),
                unsafe_allow_html=True
            )
            
            iter_metric.metric("Iteration", f"{i+1}/{len(history)}")
            attack_metric.metric("Attacking Pairs", attacks)
            restart_metric.metric("Restarts", result['restarts'])
            
            chart_placeholder.plotly_chart(
                create_convergence_plot(attack_history, "📈 Convergence Progress", i),
                use_container_width=True,
                key=f"hc_chart_{i}"
            )
            
            progress_bar.progress((i + 1) / len(history))
            time.sleep(animation_speed)
        
        # Clear animation elements
        board_placeholder.empty() 
        progress_placeholder.empty()
        chart_placeholder.empty() 
        
        # --- Final Summary & Comparison ---
        st.markdown("---")
        st.markdown("### 📋 Final Results and Analysis")
        
        # 4. Display Initial and Final States Side-by-Side (ROW 1: Boards)
        initial_final_col1, initial_final_col2 = st.columns(2)
        
        final_solution = result['solution']
        final_attacks = 0 if result['success'] else solver.count_attacks(final_solution)
        
        with initial_final_col1:
            # Display the original initial state 
            st.markdown(
                create_html_chessboard(st.session_state.hc_initial_state_final, 
                                       f"🎬 **Initial State** (Attacks: {initial_attacks})", 
                                       initial_attacks),
                unsafe_allow_html=True
            )

        with initial_final_col2:
            # Display the final state
            if result['success']:
                st.success("🎉 Solution Found!")
                st.balloons()
                st.markdown(
                    create_html_chessboard(final_solution, "✅ **Final Solution** (Attacks: 0)", 0),
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ Local Optimum Reached.")
                st.markdown(
                    create_html_chessboard(final_solution, f"❌ **Final State** (Attacks: {final_attacks})", final_attacks),
                    unsafe_allow_html=True
                )
        
        # 5. Display Convergence Chart (ROW 2 - Full Width)
        st.markdown("---")
        st.markdown("#### 📈 Convergence Path")
        st.plotly_chart(
            create_convergence_plot(result['attack_history'], "Complete Attack Count over Iterations"),
            use_container_width=True
        )
            
        # Summary metrics (below the boards and chart)
        st.markdown("---")
        metric_cols = st.columns(4)
        with metric_cols[0]: st.metric("✅ Success", "Yes" if result['success'] else "No")
        with metric_cols[1]: st.metric("🔄 Restarts", result['restarts'])
        with metric_cols[2]: st.metric("🔢 Total Iterations", result['iterations'])
        with metric_cols[3]: st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
        
        # Detailed Analysis Expander
        with st.expander("📊 Detailed Run Metrics", expanded=False):
            st.json({
                'Initial Attacks': initial_attacks,
                'Final Attacks': final_attacks,
                'Total States Explored': len(history),
                'Average Iterations per Restart': f"{result['iterations'] / max(1, result['restarts']):.1f}"
            })
            
        st.session_state.hc_result = result
        st.session_state.hc_running = False 
        
    elif 'hc_result' in st.session_state:
        st.info("Configuration complete. Press the **Run Hill Climbing** button to start a new simulation.")
def show_csp_backtracking_page():
    """CSP Backtracking page with custom initial state input, two final boards, and dynamic metric display."""
    
    st.markdown('## 🔙 CSP with Backtracking')
    
    # --- 1. Top Configuration & Controls ---
    st.markdown("### ⚙️ Configuration and Controls")
    
    param_tab, init_state_tab = st.tabs(["Algorithm Parameters", "Initial State Setup"])
    
    with param_tab:
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            board_size = st.slider("🎲 Board Size (N)", 4, 12, 8, key="csp_size")
            animation_speed = st.slider("⚡ Animation Speed (s)", 0.05, 1.0, 0.1, 
                                        help="Seconds between frames", key="csp_speed")
            
        with config_col2:
            st.empty() 
            
    with init_state_tab:
        initial_state_mode = st.radio(
            "Initial State Mode", 
            ["Empty Initial State (Standard CSP)", "Custom Initial State"], 
            index=0,
            key="csp_init_mode", 
            horizontal=True
        )
        
        # Initialize default custom state
        if "csp_custom_state_list" not in st.session_state or len(st.session_state.csp_custom_state_list) != board_size:
            st.session_state.csp_custom_state_list = [-1] * board_size 

        initial_state_for_run: Optional[List[int]] = None
        
        if initial_state_mode == "Custom Initial State":
            st.markdown(f"**Enter Row Index (0 to {board_size-1}) for each column, or **-1** to leave empty:**")
            
            num_cols_display = min(board_size, 12) 
            cols = st.columns(num_cols_display)
            parsed_state = []
            
            for col_idx in range(board_size):
                with cols[col_idx % num_cols_display]: 
                    default_value = st.session_state.csp_custom_state_list[col_idx] if col_idx < len(st.session_state.csp_custom_state_list) else -1

                    queen_row = st.number_input(
                        f"Col {col_idx}", 
                        min_value=-1, max_value=board_size - 1, 
                        value=default_value, 
                        step=1, key=f"csp_custom_q_{col_idx}",
                        label_visibility="visible"
                    )
                    parsed_state.append(int(queen_row))
            
            st.session_state.csp_custom_state_list = parsed_state
            initial_state_for_run = st.session_state.csp_custom_state_list
        else:
            initial_state_for_run = [-1] * board_size

    st.markdown("---")

    # --- 2. Run Button (Standalone) ---
    if "csp_running" not in st.session_state:
        st.session_state["csp_running"] = False
        
    # Use col2 as the main execution area for validation feedback
    execution_col = st.columns([1, 2])[1] 

    if st.button("🚀 Run CSP Backtracking", key="run_csp", use_container_width=True, type="primary"):
        # Store initial state *before* running for final comparison
        st.session_state["csp_initial_state_final"] = initial_state_for_run
        
        # --- VALIDATION: Check for conflicts in the custom initial state ---
        solver_temp = CSPBacktrackingSolver(board_size) 
        initial_attacks = solver_temp.count_attacks(initial_state_for_run)
        
        if initial_attacks > 0 and initial_state_mode == "Custom Initial State":
            with execution_col:
                st.error(f"❌ **Invalid Initial State:** The custom state has {initial_attacks} attacking pairs. Backtracking requires a conflict-free starting point.")
                st.warning("Please modify the row placements to ensure no queens attack each other before clicking run.")
                st.markdown(
                    create_html_chessboard(initial_state_for_run, 
                                           f"⚠️ Conflicting Initial State (Attacks: {initial_attacks})", 
                                           initial_attacks),
                    unsafe_allow_html=True
                )
            st.stop()
            
        st.session_state["csp_running"] = True
        st.rerun()

    st.markdown("---")

    # --- 3. Visualization and Results (Bottom Section) ---
    
    with execution_col:
        if 'csp_running' in st.session_state and st.session_state.csp_running:
            
            # --- Pre-run Setup ---
            st.markdown("### 🏃 Algorithm Execution")
            solver = CSPBacktrackingSolver(board_size)
            initial_state = st.session_state.csp_initial_state_final
            initial_attacks = solver.count_attacks(initial_state)

            # --- Visualization Containers ---
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1: iter_metric = st.empty()
            with stats_col2: attack_metric = st.empty()
            with stats_col3: col_metric = st.empty()
            
            board_col, progress_col = st.columns([1, 1])
            with board_col: board_placeholder = st.empty() 
            with progress_col: 
                progress_placeholder = st.empty()
                chart_placeholder = st.empty() 
                
            # Show initial state
            with board_placeholder.container():
                st.markdown(
                    create_html_chessboard(initial_state, "🎬 Initial State", initial_attacks),
                    unsafe_allow_html=True
                )
                
            current_col_proc = sum(1 for x in initial_state if x != -1) 
            iter_metric.metric("Iteration", "0")
            attack_metric.metric("Attacking Pairs", initial_attacks)
            col_metric.metric("Column Processing", f"{current_col_proc}/{board_size}")
            
            time.sleep(1) 
            
            # Run algorithm
            with st.spinner('🔄 Searching for solution...'):
                result = solver.solve(initial_state) 
            
            # --- Animation Loop ---
            history = result['history']
            attack_history = result['attack_history']
            
            progress_bar = progress_placeholder.progress(0)
            step = max(1, len(history) // 50)
            
            for i in range(0, len(history), step):
                state = history[i]
                attacks = attack_history[i]
                
                current_col = sum(1 for x in state if x != -1) 
                
                board_placeholder.markdown(
                    create_html_chessboard(
                        state, 
                        f"🔍 Iteration {i+1}/{len(history)}", 
                        attacks, 
                        highlight_col=current_col-1 if current_col > 0 else -1
                    ),
                    unsafe_allow_html=True
                )
                
                iter_metric.metric("Iteration", f"{i+1}/{len(history)}")
                attack_metric.metric("Attacking Pairs", attacks)
                col_metric.metric("Column Processing", f"{current_col}/{board_size}")
                
                chart_placeholder.plotly_chart(
                    create_convergence_plot(attack_history, "📈 Search Progress", i),
                    use_container_width=True,
                    key=f"csp_chart_{i}"
                )
                
                progress_bar.progress((i + 1) / len(history))
                time.sleep(animation_speed)
            
            # Clear animation elements
            board_placeholder.empty() 
            progress_placeholder.empty()
            chart_placeholder.empty() 
            
            # --- Final Summary & Comparison ---
            st.markdown("---")
            st.markdown("### 📋 Final Results and Analysis")
            
            # 4. Display Initial and Final States Side-by-Side (ROW 1: Boards)
            initial_final_col1, initial_final_col2 = st.columns(2)
            
            final_solution = result['solution']
            final_attacks = 0 if result['success'] else solver.count_attacks(final_solution)
            
            with initial_final_col1:
                st.markdown(
                    create_html_chessboard(initial_state, 
                                           f"🎬 **Initial State** (Queens Placed: {sum(1 for x in initial_state if x != -1)})", 
                                           initial_attacks),
                    unsafe_allow_html=True
                )

            with initial_final_col2:
                if result['success']:
                    st.success("🎉 Solution Found!")
                    st.balloons()
                    st.markdown(
                        create_html_chessboard(final_solution, "✅ **Final Solution**", 0),
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("❌ Search Terminated.")
                    st.markdown(
                        create_html_chessboard(final_solution, "❌ **Final Partial State**", final_attacks),
                        unsafe_allow_html=True
                    )
            
            # 5. Display Convergence Chart (ROW 2 - Full Width)
            st.markdown("---")
            st.markdown("#### 📈 Complete Search Path")
            st.plotly_chart(
                create_convergence_plot(result['attack_history'], "Attack Count over Search Steps"),
                use_container_width=True
            )
                
            # Summary metrics (below the boards and chart)
            st.markdown("---")
            
            # Check if the 'backtracks' data is available
            show_backtracks = 'backtracks' in result
            num_metrics = 4 if show_backtracks else 3
            
            metric_cols = st.columns(num_metrics)
            
            # Column 1: Success
            with metric_cols[0]: 
                st.metric("✅ Success", "Yes" if result['success'] else "No")
                
            # Column 2: Iterations
            with metric_cols[1]: 
                st.metric("🔢 Total Iterations", result['iterations'])
            
            if show_backtracks:
                # Column 3: Backtracks (if present)
                with metric_cols[2]: 
                    st.metric("🔙 Backtracks", result['backtracks'])
                
                # Column 4: Runtime
                with metric_cols[3]: 
                    st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
            else:
                # Column 3: Runtime (if Backtracks is absent)
                with metric_cols[2]: 
                    st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
            
            # Detailed Analysis Expander
            with st.expander("📊 Detailed Run Metrics", expanded=False):
                detail_json = {
                    'Initial Queens Placed': sum(1 for x in initial_state if x != -1),
                    'Final Attacks': 0 if result['success'] else final_attacks,
                    'Total States Explored': len(history),
                    'Method': result['method'],
                }
                if show_backtracks:
                    detail_json['Backtracks'] = result['backtracks']
                
                st.json(detail_json)
                
            st.session_state.csp_result = result
            st.session_state.csp_running = False 
            
        elif 'csp_result' in st.session_state:
            st.info("Configuration complete. Press the **Run CSP Backtracking** button to start a new search.")
def show_csp_backtracking_page():
    """CSP Backtracking page with custom initial state input and two final boards display."""
    
    st.markdown('## 🔙 CSP with Backtracking')
    
    # --- 1. Top Configuration & Controls ---
    st.markdown("### ⚙️ Configuration and Controls")
    
    param_tab, init_state_tab = st.tabs(["Algorithm Parameters", "Initial State Setup"])
    
    with param_tab:
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            board_size = st.slider("🎲 Board Size (N)", 4, 12, 8, key="csp_size")
            # Max iterations is less relevant for backtracking, but we'll keep the speed control
            animation_speed = st.slider("⚡ Animation Speed (s)", 0.05, 1.0, 0.1, 
                                        help="Seconds between frames", key="csp_speed")
            
        with config_col2:
            # Placeholder for future CSP controls (e.g., Variable Ordering, Value Ordering)
            # Keeping the layout consistent with Hill Climbing
            st.empty() 
            
    with init_state_tab:
        # Initial State Input/Selection (Adapted from Hill Climbing)
        initial_state_mode = st.radio(
            "Initial State Mode", 
            ["Empty Initial State (Standard CSP)", "Custom Initial State"], 
            index=0,
            key="csp_init_mode", 
            horizontal=True
        )
        
        # Initialize default custom state in session_state if it doesn't exist
        # Default state for custom CSP: all queens at row 0 (which is an empty board for N-Queens CSP solver)
        if "csp_custom_state_list" not in st.session_state or len(st.session_state.csp_custom_state_list) != board_size:
            # Use -1 to represent an empty position in a column for CSP initialization
            st.session_state.csp_custom_state_list = [-1] * board_size 

        initial_state_for_run = None
        
        if initial_state_mode == "Custom Initial State":
            st.markdown(f"**Enter Row Index (0 to {board_size-1}) for each column, or **-1** to leave empty:**")
            
            num_cols_display = min(board_size, 12) 
            cols = st.columns(num_cols_display)
            parsed_state = []
            
            for col_idx in range(board_size):
                with cols[col_idx % num_cols_display]: 
                    default_value = st.session_state.csp_custom_state_list[col_idx] if col_idx < len(st.session_state.csp_custom_state_list) else -1

                    queen_row = st.number_input(
                        f"Col {col_idx}", 
                        min_value=-1, max_value=board_size - 1, # Allows -1 for empty
                        value=default_value, 
                        step=1, key=f"csp_custom_q_{col_idx}",
                        label_visibility="visible"
                    )
                    parsed_state.append(queen_row)
            
            st.session_state.csp_custom_state_list = parsed_state
            initial_state_for_run = st.session_state.csp_custom_state_list
        else:
             # Standard Empty state: -1 for every column
            initial_state_for_run = [-1] * board_size

    st.markdown("---")

    # --- 2. Run Button (Standalone) ---
    if "csp_running" not in st.session_state:
        st.session_state["csp_running"] = False
        
    if st.button("🚀 Run CSP Backtracking", key="run_csp", use_container_width=True, type="primary"):
        st.session_state["csp_running"] = True
        st.session_state["csp_initial_state_final"] = initial_state_for_run # Store for final comparison
        st.rerun()

    st.markdown("---")

    # --- 3. Visualization and Results (Bottom Section) ---
    
    if 'csp_running' in st.session_state and st.session_state.csp_running:
        st.markdown("### 🏃 Algorithm Execution")
        
        # --- Pre-run Setup ---
        solver = CSPBacktrackingSolver(board_size)
        initial_state = st.session_state.csp_initial_state_final
        # Initial attacks for the visual metric (should be calculated on the initial state)
        initial_attacks = solver.count_attacks(initial_state) 

        # --- Visualization Containers ---
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1: iter_metric = st.empty()
        with stats_col2: attack_metric = st.empty()
        with stats_col3: col_metric = st.empty()
        
        board_col, progress_col = st.columns([1, 1])
        with board_col: board_placeholder = st.empty() 
        with progress_col: 
            progress_placeholder = st.empty()
            chart_placeholder = st.empty() 
            
        # Show initial state in the animation area before starting
        with board_placeholder.container():
            # Initial state display (before search starts)
            st.markdown(
                create_html_chessboard(initial_state, "🎬 Initial State", initial_attacks),
                unsafe_allow_html=True
            )
            
        # Initial metric display
        current_col_proc = sum(1 for x in initial_state if x != -1) # Count placed queens
        iter_metric.metric("Iteration", "0")
        attack_metric.metric("Attacking Pairs", initial_attacks)
        col_metric.metric("Column Processing", f"{current_col_proc}/{board_size}")
        
        time.sleep(1) # Pause for initial view
        
        # Run algorithm from the initial state
        with st.spinner('🔄 Searching for solution...'):
            result = solver.solve(initial_state) # Pass initial state to solver
        
        # --- Animation Loop ---
        history = result['history']
        attack_history = result['attack_history']
        
        progress_bar = progress_placeholder.progress(0)
        step = max(1, len(history) // 50)
        
        for i in range(0, len(history), step):
            state = history[i]
            attacks = attack_history[i]
            
            # Find current column being processed
            current_col = sum(1 for x in state if x != -1) 
            
            board_placeholder.markdown(
                create_html_chessboard(
                    state, 
                    f"🔍 Iteration {i+1}/{len(history)}", 
                    attacks, 
                    highlight_col=current_col-1 if current_col > 0 else -1
                ),
                unsafe_allow_html=True
            )
            
            iter_metric.metric("Iteration", f"{i+1}/{len(history)}")
            attack_metric.metric("Attacking Pairs", attacks)
            col_metric.metric("Column Processing", f"{current_col}/{board_size}")
            
            chart_placeholder.plotly_chart(
                create_convergence_plot(attack_history, "📈 Search Progress", i),
                use_container_width=True,
                key=f"csp_chart_{i}"
            )
            
            progress_bar.progress((i + 1) / len(history))
            time.sleep(animation_speed)
        
        # Clear animation elements
        board_placeholder.empty() 
        progress_placeholder.empty()
        chart_placeholder.empty() 
        
        # --- Final Summary & Comparison ---
        st.markdown("---")
        st.markdown("### 📋 Final Results and Analysis")
        
        # 4. Display Initial and Final States Side-by-Side (ROW 1: Boards)
        initial_final_col1, initial_final_col2 = st.columns(2)
        
        final_solution = result['solution']
        final_attacks = 0 if result['success'] else solver.count_attacks(final_solution) # Recalculate if not successful
        
        # Use the stored initial state for the final comparison
        initial_state_final = st.session_state.csp_initial_state_final
        
        with initial_final_col1:
            # Display the original initial state (Chessboard 1)
            st.markdown(
                create_html_chessboard(initial_state_final, 
                                       f"🎬 **Initial State** (Queens Placed: {sum(1 for x in initial_state_final if x != -1)})", 
                                       initial_attacks),
                unsafe_allow_html=True
            )

        with initial_final_col2:
            # Display the final state (Chessboard 2)
            if result['success']:
                st.success("🎉 Solution Found!")
                st.balloons()
                st.markdown(
                    create_html_chessboard(final_solution, "✅ **Final Solution**", 0),
                    unsafe_allow_html=True
                )
            else:
                st.warning("❌ Search Terminated.")
                st.markdown(
                    create_html_chessboard(final_solution, "❌ **Final Partial State**", final_attacks),
                    unsafe_allow_html=True
                )
        
        # 5. Display Convergence Chart (ROW 2 - Full Width)
        st.markdown("---")
        st.markdown("#### 📈 Complete Search Path")
        st.plotly_chart(
            create_convergence_plot(result['attack_history'], "Attack Count over Search Steps"),
            use_container_width=True
        )
            
        # Summary metrics (below the boards and chart)
        st.markdown("---")
        metric_cols = st.columns(3)
        with metric_cols[0]: st.metric("✅ Success", "Yes" if result['success'] else "No")
        with metric_cols[1]: st.metric("🔢 Total Iterations", result['iterations'])
        # with metric_cols[2]: st.metric("🔙 Backtracks", result['backtracks']) # Assuming solver returns 'backtracks'
        with metric_cols[2]: st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
        
        # Detailed Analysis Expander
        with st.expander("📊 Detailed Run Metrics", expanded=False):
            st.json({
                'Initial Queens Placed': sum(1 for x in initial_state_final if x != -1),
                'Final Attacks': 0 if result['success'] else final_attacks,
                'Total States Explored': len(history),
                'Method': result['method'],
            })
            
        st.session_state.csp_result = result
        st.session_state.csp_running = False 
        
    elif 'csp_result' in st.session_state:
        st.info("Configuration complete. Press the **Run CSP Backtracking** button to start a new search.")
import streamlit as st
import time
from typing import List, Optional # Assuming these imports are available

# NOTE: The implementation of CSPEnhancedSolver (must now handle a non-empty initial state), 
# create_html_chessboard, and create_convergence_plot is assumed to be available elsewhere.
# CSPEnhancedSolver is assumed to return 'backtracks' or 'efficiency_score'.

def show_csp_enhanced_page():
    """Enhanced CSP page with animation, custom initial state, and robust final summary."""
    
    st.markdown('<div class="algorithm-title">⚡ CSP with Forward Checking & MRV</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuration and Controls")
        
        param_tab, init_state_tab = st.tabs(["Algorithm Parameters", "Initial State Setup"])

        with param_tab:
            board_size = st.slider("🎲 Board Size (N)", 4, 12, 8, key="cspe_size")
            animation_speed = st.slider("⚡ Animation Speed (s)", 0.05, 1.0, 0.1, 
                                        help="Seconds between frames", key="cspe_speed")
            
        initial_state_for_run: Optional[List[int]] = None
        
        with init_state_tab:
            initial_state_mode = st.radio(
                "Initial State Mode", 
                ["Empty Initial State (Standard CSP)", "Custom Initial State"], 
                index=0,
                key="cspe_init_mode", 
                horizontal=True
            )
            
            # Initialize default custom state (Defaulting to row 0)
            if "cspe_custom_state_list" not in st.session_state or len(st.session_state.cspe_custom_state_list) != board_size:
                st.session_state.cspe_custom_state_list = [0] * board_size 

            if initial_state_mode == "Custom Initial State":
                st.markdown(f"**Enter Row Index (0 to {board_size-1}), or **-1** to leave empty:**")
                
                num_cols_display = min(board_size, 12) 
                cols = st.columns(num_cols_display)
                parsed_state = []
                
                for col_idx in range(board_size):
                    with cols[col_idx % num_cols_display]: 
                        default_value = st.session_state.cspe_custom_state_list[col_idx] if col_idx < len(st.session_state.cspe_custom_state_list) else 0

                        queen_row = st.number_input(
                            f"Col {col_idx}", 
                            min_value=-1, max_value=board_size - 1, 
                            value=default_value, 
                            step=1, key=f"cspe_custom_q_{col_idx}",
                            label_visibility="visible"
                        )
                        parsed_state.append(int(queen_row))
                
                st.session_state.cspe_custom_state_list = parsed_state
                initial_state_for_run = st.session_state.cspe_custom_state_list
            else:
                initial_state_for_run = [-1] * board_size
        
        st.markdown("---")
        
        st.markdown("""
        **📚 Algorithm Overview:**
        
        Enhanced CSP with **intelligent heuristics**:
        - 🎯 **Forward Checking:** Prunes domains early
        - 🧠 **MRV:** Picks most constrained variable
        - ⚡ More efficient than basic backtracking
        """)
        
        st.markdown("---")
        
        if st.button("🚀 Run Enhanced CSP", key="run_cspe", use_container_width=True):
            st.session_state["cspe_initial_state_final"] = initial_state_for_run
            
            # --- VALIDATION: Check for conflicts in the custom initial state ---
            solver_temp = CSPEnhancedSolver(board_size) 
            initial_attacks = solver_temp.count_attacks(initial_state_for_run)
            
            if initial_attacks > 0 and initial_state_mode == "Custom Initial State":
                with col2:
                    st.error(f"❌ **Invalid Initial State:** The custom state has {initial_attacks} attacking pairs.")
                    st.warning("Please modify the row placements to ensure no queens attack each other.")
                    st.markdown(
                        create_html_chessboard(initial_state_for_run, 
                                               f"⚠️ Conflicting Initial State (Attacks: {initial_attacks})", 
                                               initial_attacks),
                        unsafe_allow_html=True
                    )
                st.stop()
                
            st.session_state["cspe_running"] = True
            st.rerun()
            
    with col2:
        if 'cspe_running' in st.session_state and st.session_state.cspe_running:
            
            st.markdown("### 🏃 Algorithm Execution")
            solver = CSPEnhancedSolver(board_size)
            initial_state = st.session_state.cspe_initial_state_final
            initial_attacks = solver.count_attacks(initial_state)

            # Animation containers
            board_placeholder = st.empty()
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1: iter_metric = st.empty()
            with stats_col2: attack_metric = st.empty()
            with stats_col3: efficiency_metric = st.empty() # Keeping original metric
            
            progress_placeholder = st.empty()
            # chart_placeholder is REMOVED from animation area
            
            # Run algorithm
            with st.spinner('🔄 Searching for solution...'):
                result = solver.solve(initial_state) # Pass initial state
            
            # Animate
            history = result['history']
            attack_history = result['attack_history']
            
            progress_bar = progress_placeholder.progress(0)
            step = max(1, len(history) // 50)
            
            for i in range(0, len(history), step):
                state = history[i]
                attacks = attack_history[i]
                
                # Assuming state represents a partial solution (row placement)
                current_col = sum(1 for x in state if x != -1)
                
                board_placeholder.markdown(
                    create_html_chessboard(state, f"🧠 Iteration {i+1}/{len(history)}", 
                                            attacks, highlight_col=current_col-1 if current_col > 0 else -1),
                    unsafe_allow_html=True
                )
                
                iter_metric.metric("Iteration", f"{i+1}/{len(history)}")
                attack_metric.metric("Attacking Pairs", attacks)
                # Keep the original metric from the provided code
                efficiency_metric.metric("Efficiency", f"{(i+1)/board_size:.1f}x")
                
                # REMOVED: chart_placeholder.plotly_chart(...) to ensure chart displays only at the end
                
                progress_bar.progress((i + 1) / len(history))
                time.sleep(animation_speed)
            
            progress_placeholder.empty()
            board_placeholder.empty() # Clear board for final summary display

            # --- Final Summary & Comparison ---
            st.markdown("---")
            st.markdown("### 📋 Final Results and Analysis")
            
            # 1. Display Initial and Final States Side-by-Side (ROW 1: Boards)
            initial_final_col1, initial_final_col2 = st.columns(2)
            
            final_solution = result['solution']
            final_attacks = 0 if result['success'] else solver.count_attacks(final_solution)
            initial_state_final = st.session_state.cspe_initial_state_final 
            
            with initial_final_col1:
                # Display the original initial state (Chessboard 1)
                st.markdown(
                    create_html_chessboard(initial_state_final, 
                                           f"🎬 **Initial State** (Queens Placed: {sum(1 for x in initial_state_final if x != -1)})", 
                                           initial_attacks),
                    unsafe_allow_html=True
                )

            with initial_final_col2:
                # Display the final state (Chessboard 2)
                if result['success']:
                    st.success("🎉 Solution Found!")
                    st.balloons()
                    st.markdown(
                        create_html_chessboard(final_solution, "✅ **Final Solution**", 0),
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("❌ Search Terminated.")
                    st.markdown(
                        create_html_chessboard(final_solution, "❌ **Final Partial State**", final_attacks),
                        unsafe_allow_html=True
                    )

            # 2. Display Convergence Chart (ROW 2 - Full Width - ONLY AFTER ANIMATION)
            st.markdown("---")
            st.markdown("#### 📈 Complete Search Path")
            st.plotly_chart(
                create_convergence_plot(result['attack_history'], "Attack Count over Search Steps"),
                use_container_width=True
            )
            
            # 3. Dynamic Summary metrics
            st.markdown("---")
            
            # Check for optional metrics like 'backtracks' or 'nodes_explored'
            show_backtracks = 'backtracks' in result
            num_metrics = 3 + (1 if show_backtracks else 0)
            
            metric_cols = st.columns(num_metrics)
            
            # Common Metrics
            with metric_cols[0]: 
                st.metric("✅ Success", "Yes" if result['success'] else "No")
            with metric_cols[1]: 
                st.metric("🔢 Total Iterations", result['iterations'])
            
            if show_backtracks:
                # Backtracks (Conditional)
                with metric_cols[2]: 
                    st.metric("🔙 Backtracks", result['backtracks'])
                # Runtime (Column 4)
                with metric_cols[3]: 
                    st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
            else:
                # Runtime (Column 3)
                with metric_cols[2]: 
                    st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
            
            # Detailed Analysis Expander
            with st.expander("📊 Detailed Analysis", expanded=False):
                detail_json = {
                    'Method': result['method'],
                    'Initial Queens Placed': sum(1 for x in initial_state_final if x != -1),
                    'Total States Explored': len(history),
                    'Final Attacks': 0 if result['success'] else final_attacks,
                }
                if show_backtracks:
                    detail_json['Backtracks'] = result['backtracks']
                if 'efficiency_score' in result:
                    detail_json['Heuristic Efficiency'] = f"{result['efficiency_score']:.2f}x optimal"

                st.json(detail_json)
                
            st.session_state.cspe_result = result 
            st.session_state.cspe_running = False
        
        elif 'cspe_result' in st.session_state:
            st.info("Configuration complete. Press the **Run Enhanced CSP** button to start a new search.")

def show_comparison_page():
    """Comparison page for all algorithms."""
    # Ensure pandas, statistics, and plotly are available for the entire script context
    import statistics 
    import pandas as pd
    import plotly.graph_objects as go
    import time
    
    st.markdown('<div class="algorithm-title">📊 Algorithm Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        with st.container():
            # NOTE: Assuming EightQueens, HillClimbingSolver, CSPBacktrackingSolver, CSPEnhancedSolver are defined elsewhere
            # and time is imported.
            board_size = st.slider("🎲 Board Size", 4, 12, 8, key="comp_board_size_slider")
            max_restarts = st.slider("🔄 Max Restarts (HC)", 10, 100, 50, key="comp_max_restarts_slider")
            num_tests = st.slider("🧪 Number of Test Cases", 1, 5, 3, key="comp_num_tests", 
                                 help="Run multiple test cases with different initial states")
        
        st.markdown("---")
        
        st.markdown("""
        **📚 Comparison Mode:**
        
        Run all three algorithms on the **same initial state** to fairly compare:
        - ⛰️ Hill Climbing (Local Search)
        - 🔙 CSP Backtracking (Basic)
        - ⚡ CSP Enhanced (FC + MRV)
        
        Compare speed, efficiency, and success rates!
        """)
        
        st.markdown("---")
        
        if st.button("🚀 Run All Algorithms", key="run_comp", use_container_width=True):
            st.session_state.comp_config_restarts = max_restarts
            st.session_state.comp_config_board_size = board_size
            st.session_state.comp_config_num_tests = num_tests
            st.session_state.comp_running = True
            st.rerun()
    
    with col2:
        if st.session_state.get('comp_running', False):
            board_size = st.session_state.get('comp_config_board_size', 8)
            max_restarts = st.session_state.get('comp_config_restarts', 50)
            num_tests = st.session_state.get('comp_config_num_tests', 3)
            
            # Store all test results
            all_test_results = []
            
            # Run multiple test cases
            for test_num in range(num_tests):
                st.markdown(f"## 🧪 Test Case {test_num + 1}")
                
                # Generate common initial state for this test
                base = EightQueens(board_size)
                initial_state = base.generate_random_state()
                
                st.markdown("### 🎯 Initial Configuration")
                # Keeping the initial graphical board as it's just one element
                st.markdown(
                    create_html_chessboard(initial_state, f"Test {test_num + 1} - Initial State", 
                                          base.count_attacks(initial_state)),
                    unsafe_allow_html=True
                )
                
                # Progress container
                progress_container = st.container()
                
                with progress_container:
                    st.markdown("### 🔄 Running Algorithms...")
                    
                    algo_status = st.empty()
                    results = {}
                    
                    try:
                        # Hill Climbing
                        algo_status.info("⛰️ Running Hill Climbing...")
                        hc_solver = HillClimbingSolver(board_size, max_restarts, 500)
                        results['Hill Climbing'] = hc_solver.solve(initial_state.copy())
                        algo_status.success("✅ Hill Climbing Complete")
                        time.sleep(0.3)
                        
                        # CSP Backtracking
                        algo_status.info("🔙 Running CSP Backtracking...")
                        csp_solver = CSPBacktrackingSolver(board_size)
                        results['CSP Backtracking'] = csp_solver.solve(initial_state.copy())
                        algo_status.success("✅ CSP Backtracking Complete")
                        time.sleep(0.3)
                        
                        # Enhanced CSP
                        algo_status.info("⚡ Running Enhanced CSP...")
                        cspe_solver = CSPEnhancedSolver(board_size)
                        results['CSP Enhanced'] = cspe_solver.solve(initial_state.copy())
                        algo_status.success("✅ All Algorithms Complete!")
                        time.sleep(0.3)
                        
                    except Exception as e:
                        st.error(f"❌ Error during algorithm execution: {str(e)}")
                        continue
                    finally:
                        algo_status.empty()
                
                # Store results for this test
                test_result = {
                    'test_num': test_num + 1,
                    'initial_state': initial_state,
                    'initial_attacks': base.count_attacks(initial_state),
                    'results': results
                }
                all_test_results.append(test_result)
                
                # Display results for this test
                st.markdown(f"### 🏆 Test {test_num + 1} - Performance Comparison")
                
                # Metrics
                metric_cols = st.columns(3)
                for i, (method, result) in enumerate(results.items()):
                    with metric_cols[i]:
                        success_emoji = "✅" if result['success'] else "❌"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{success_emoji} {method}</h3>
                            <h2>{result['runtime']*1000:.2f} ms</h2>
                            <p>{result['iterations']} iterations</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # --- START FIX: Solutions displayed vertically (stacked) using full width ---
                st.markdown(f"### 🎯 Test {test_num + 1} - Solutions")
                
                # Iterate through results and display each board using the full width of col2
                for method, result in results.items():
                    st.markdown(f"#### {method} Solution")
                    
                    if result['success'] and result['solution']:
                        # The chessboard is displayed using the full width of the parent container (col2)
                        st.markdown(
                            create_html_chessboard(result['solution'], f"{method}", 0),
                            unsafe_allow_html=True
                        )
                        st.markdown("---") # Separator after each board
                    else:
                        st.error(f"❌ {method} - No solution found")
                        st.markdown("---") # Separator after the error
                # --- END FIX ---
                
                st.markdown("---")
            
            # Store all results in session state
            st.session_state.comp_all_test_results = all_test_results
            
            # Aggregate Analysis
            st.markdown("# 📊 Aggregate Analysis Across All Tests")
            
            # Calculate aggregate statistics
            aggregate_stats = {}
            for method in ['Hill Climbing', 'CSP Backtracking', 'CSP Enhanced']:
                method_results = [test['results'][method] for test in all_test_results]
                
                aggregate_stats[method] = {
                    'success_rate': sum(1 for r in method_results if r['success']) / len(method_results) * 100,
                    'avg_iterations': statistics.mean([r['iterations'] for r in method_results]),
                    'avg_runtime': statistics.mean([r['runtime'] for r in method_results]),
                    'min_iterations': min([r['iterations'] for r in method_results]),
                    'max_iterations': max([r['iterations'] for r in method_results]),
                    'min_runtime': min([r['runtime'] for r in method_results]),
                    'max_runtime': max([r['runtime'] for r in method_results]),
                    'total_successes': sum(1 for r in method_results if r['success']),
                    'total_tests': len(method_results)
                }
            
            # Display aggregate metrics
            st.markdown("### 📈 Overall Performance Metrics")
            agg_cols = st.columns(3)
            
            for i, (method, stats) in enumerate(aggregate_stats.items()):
                with agg_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{method}</h3>
                        <h4>Success Rate: {stats['success_rate']:.1f}%</h4>
                        <p>Avg Runtime: {stats['avg_runtime']*1000:.2f} ms</p>
                        <p>Avg Iterations: {stats['avg_iterations']:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed comparison table for all tests
            st.markdown("### 📋 Detailed Comparison Table (All Tests)")
            
            detailed_data = []
            for test in all_test_results:
                test_num = test['test_num']
                for method, result in test['results'].items():
                    row = {
                        'Test Case': test_num,
                        'Algorithm': method,
                        'Success': '✅ Yes' if result['success'] else '❌ No',
                        'Iterations': result['iterations'],
                        'Runtime (ms)': f"{result['runtime']*1000:.4f}",
                        'Restarts': result.get('restarts', 'N/A'),
                        'Initial Attacks': test['initial_attacks']
                    }
                    detailed_data.append(row)
            
            detailed_df = pd.DataFrame(detailed_data)
            st.dataframe(detailed_df, use_container_width=True, hide_index=True)
            
            # Summary statistics table
            st.markdown("### 📊 Summary Statistics")
            
            summary_data = []
            for method, stats in aggregate_stats.items():
                row = {
                    'Algorithm': method,
                    'Success Rate (%)': f"{stats['success_rate']:.1f}%",
                    'Avg Iterations': f"{stats['avg_iterations']:.2f}",
                    'Avg Runtime (ms)': f"{stats['avg_runtime']*1000:.4f}",
                    'Min Iterations': stats['min_iterations'],
                    'Max Iterations': stats['max_iterations'],
                    'Total Success': f"{stats['total_successes']}/{stats['total_tests']}"
                }
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Comparison charts (Already corrected to be vertical)
            st.markdown("### 📊 Performance Visualization")
            
            # Average iterations comparison
            st.markdown("#### Average Iterations")
            fig_iter = go.Figure()
            fig_iter.add_trace(go.Bar(
                x=list(aggregate_stats.keys()),
                y=[stats['avg_iterations'] for stats in aggregate_stats.values()],
                marker_color=['#667eea', '#764ba2', '#f093fb'],
                text=[f"{stats['avg_iterations']:.1f}" for stats in aggregate_stats.values()],
                textposition='auto',
            ))
            fig_iter.update_layout(
                title='Average Iterations',
                xaxis_title='Algorithm',
                yaxis_title='Iterations',
                height=350
            )
            st.plotly_chart(fig_iter, use_container_width=True)
            
            # Average runtime comparison
            st.markdown("#### Average Runtime")
            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(
                x=list(aggregate_stats.keys()),
                y=[stats['avg_runtime']*1000 for stats in aggregate_stats.values()],
                marker_color=['#667eea', '#764ba2', '#f093fb'],
                text=[f"{stats['avg_runtime']*1000:.2f}" for stats in aggregate_stats.values()],
                textposition='auto',
            ))
            fig_time.update_layout(
                title='Average Runtime',
                xaxis_title='Algorithm',
                yaxis_title='Runtime (ms)',
                height=350
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Success rate comparison
            st.markdown("#### Success Rate Comparison")
            fig_success = go.Figure()
            fig_success.add_trace(go.Bar(
                x=list(aggregate_stats.keys()),
                y=[stats['success_rate'] for stats in aggregate_stats.values()],
                marker_color=['#667eea', '#764ba2', '#f093fb'],
                text=[f"{stats['success_rate']:.1f}%" for stats in aggregate_stats.values()],
                textposition='auto',
            ))
            fig_success.update_layout(
                title='Success Rate Comparison',
                xaxis_title='Algorithm',
                yaxis_title='Success Rate (%)',
                height=350,
                yaxis=dict(range=[0, 105])
            )
            st.plotly_chart(fig_success, use_container_width=True)
            
            # Box plot for iterations distribution
            st.markdown("#### Iterations Distribution Across Tests")
            
            fig_box = go.Figure()
            for method in aggregate_stats.keys():
                iterations = [test['results'][method]['iterations'] for test in all_test_results]
                fig_box.add_trace(go.Box(
                    y=iterations,
                    name=method,
                    boxmean='sd'
                ))
            
            fig_box.update_layout(
                title='Distribution of Iterations Across All Test Cases',
                yaxis_title='Iterations',
                height=450
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Key Insights
            st.markdown("### 🔍 Key Insights")
            
            fastest = min(aggregate_stats.items(), key=lambda x: x[1]['avg_runtime'])
            most_efficient = min(aggregate_stats.items(), key=lambda x: x[1]['avg_iterations'])
            most_reliable = max(aggregate_stats.items(), key=lambda x: x[1]['success_rate'])
            
            insight_cols = st.columns(3)
            
            with insight_cols[0]:
                st.markdown(f"""
                <div class="stat-box">
                    <h4>⚡ Fastest Algorithm</h4>
                    <h3>{fastest[0]}</h3>
                    <p>{fastest[1]['avg_runtime']*1000:.2f} ms avg</p>
                </div>
                """, unsafe_allow_html=True)
            
            with insight_cols[1]:
                st.markdown(f"""
                <div class="stat-box">
                    <h4>🎯 Most Efficient</h4>
                    <h3>{most_efficient[0]}</h3>
                    <p>{most_efficient[1]['avg_iterations']:.1f} iterations avg</p>
                </div>
                """, unsafe_allow_html=True)
            
            with insight_cols[2]:
                st.markdown(f"""
                <div class="stat-box">
                    <h4>✅ Most Reliable</h4>
                    <h3>{most_reliable[0]}</h3>
                    <p>{most_reliable[1]['success_rate']:.1f}% success rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Download results
            st.markdown("### 💾 Export Results")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # Export detailed results as CSV
                csv_data = detailed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Detailed Results (CSV)",
                    data=csv_data,
                    file_name=f"8queens_detailed_results_{board_size}x{board_size}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with export_col2:
                # Export summary as CSV
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary (CSV)",
                    data=summary_csv,
                    file_name=f"8queens_summary_{board_size}x{board_size}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with st.expander("📖 Algorithm Analysis", expanded=False):
                st.markdown("""
                #### Comparative Analysis:
                
                **⛰️ Hill Climbing (Local Search):**
                - ⚡ Very fast local search algorithm
                - 🎲 Stochastic approach with random restarts
                - ⚠️ Can get trapped in local optima
                - 📊 Completeness: No (may not find solution)
                - 🎯 Optimality: No (finds local optimum)
                - 💡 Best for: Quick approximate solutions when time is critical
                
                **🔙 CSP Backtracking (Basic):**
                - 🎯 Systematic exhaustive search
                - ✅ Guarantees finding solution if exists
                - 🐢 May explore many states without pruning
                - 📊 Completeness: Yes (finds all solutions)
                - 🎯 Optimality: Yes (systematic search)
                - 💡 Best for: Guaranteed complete solutions, smaller problems
                
                **⚡ CSP Enhanced (Forward Checking + MRV):**
                - 🧠 Intelligent constraint propagation
                - 📉 Prunes search space efficiently using Forward Checking
                - 🎲 Minimum Remaining Values heuristic guides search
                - 🚀 Significantly faster than basic backtracking
                - 📊 Completeness: Yes (finds all solutions)
                - 🎯 Optimality: Yes (optimal search strategy)
                - 💡 Best for: Optimal systematic search with efficiency
                
                ---
                
                **Performance Factors:**
                - **Board Size:** Larger boards exponentially increase difficulty
                - **Initial State:** Random placement affects Hill Climbing convergence
                - **Heuristics:** MRV and FC dramatically improve CSP efficiency
                - **Constraints:** Tighter constraints make search harder
                
                ---
                
                **Recommendations:**
                - Use **Hill Climbing** for quick solutions when completeness isn't required
                - Use **CSP Basic** for learning and understanding backtracking
                - Use **CSP Enhanced** for optimal balance of speed and completeness
                """)
            
            # Add reset button at the bottom
            st.markdown("---")
            if st.button("🔄 Reset and Run New Comparison", key="reset_comp", use_container_width=True):
                st.session_state.comp_running = False
                st.session_state.comp_results = None
                st.session_state.comp_all_test_results = None
                st.rerun()
# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Main heading with authors - Green Chessboard Badge Style
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem; background-color: #7d945d; border-radius: 15px; color: #eeeed5;'>
        <h1 style='margin: 0; font-weight: 700;'>♛ 8-Queens AI Solver ♛</h1>
        <h4 style='margin: 0.3rem 0 0 0; font-weight: 400;'>CEP Project by Jibran Saleem, Sajjad Abidi & Arif Mehdi</h4>
        </div>
        """, unsafe_allow_html=True
    )

    # Intro section - Badge style
    st.markdown(
        """
        <div style='text-align: center; padding: 1.5rem; background-color: #7d945d; border-radius: 15px; margin: 1rem 0; color: #eeeed5;'>
        <h3 style='margin-bottom: 0.5rem;'>Explore AI Algorithms Solving the Classic 8-Queens Problem</h3>
        <p style='margin: 0;'>Compare Hill Climbing, CSP Backtracking, and Enhanced CSP with interactive visualization.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["🏠 Home", "⛰️ Hill Climbing", "🔙 CSP Backtracking", "⚡ CSP Enhanced", "📊 Comparison"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='padding: 1rem; background-color: #7d945d; border-radius: 12px; color: #eeeed5;'>
        <h3 style='margin-top: 0;'>📚 About 8-Queens</h3>
        <p>Place 8 queens on an 8×8 board so no two queens attack each other.</p>
        <p><strong>Constraints:</strong></p>
        <ul style='margin-left: 1rem;'>
        <li>❌ No two queens in same row</li>
        <li>❌ No two queens in same column</li>
        <li>❌ No two queens on same diagonal</li>
        </ul>
        <p style='font-weight: bold;'>Total Solutions: 92 unique solutions</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    if page != "🏠 Home":
        st.sidebar.markdown(
            """
            <div style='padding: 1rem; background-color: #9bb174; border-radius: 10px; color: #eeeed5;'>
            <h3 style='margin-top: 0;'>🎯 Quick Tips</h3>
            <ul>
            <li>Adjust animation speed slider</li>
            <li>Use smaller board sizes (4-8) for faster visualization</li>
            <li>Watch live metrics while the algorithm runs</li>
            </ul>
            </div>
            """, unsafe_allow_html=True
        )

    # Home Page
    if page == "🏠 Home":
        st.markdown(
            """
            <h3 style='color: #7d945d; font-weight: 600;'>🎓 Welcome to the 8-Queens Solver</h3>
            <p style='color: #3d3d3d;'>This application demonstrates AI algorithms solving the classic 8-Queens problem with <strong style='color: #7d945d;'>smooth animations</strong> and <strong style='color: #7d945d;'>detailed visualizations</strong>.</p>
            """, unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("<h3 style='color: #7d945d; font-weight: 600;'>🚀 Choose an Algorithm to Explore</h3>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        # Hill Climbing Badge
        with col1:
            st.markdown(
                """
                <div style='padding: 1rem; background-color: #7d945d; border-radius: 12px; color: #eeeed5;'>
                <h4>⛰️ Hill Climbing</h4>
                <p><strong>Type:</strong> Local Search</p>
                <p><strong>Strategy:</strong> Greedy optimization with random restarts</p>
                <p><strong>Pros:</strong> Fast, simple</p>
                <p><strong>Cons:</strong> May get stuck in local maxima</p>
                <p style='font-style: italic;'>Best for quick approximate solutions</p>
                </div>
                """, unsafe_allow_html=True
            )

        # CSP Backtracking Badge
        with col2:
            st.markdown(
                """
                <div style='padding: 1rem; background-color: #eeeed5; border-radius: 12px; border: 2px solid #7d945d; color: #7d945d;'>
                <h4>🔙 CSP Backtracking</h4>
                <p><strong>Type:</strong> Systematic Search</p>
                <p><strong>Strategy:</strong> Constraint satisfaction with backtracking</p>
                <p><strong>Pros:</strong> Guaranteed solution</p>
                <p><strong>Cons:</strong> Can be slower</p>
                <p style='font-style: italic;'>Best for complete solutions</p>
                </div>
                """, unsafe_allow_html=True
            )

        # CSP Enhanced Badge
        with col3:
            st.markdown(
                """
                <div style='padding: 1rem; background-color: #7d945d; border-radius: 12px; color: #eeeed5;'>
                <h4>⚡ CSP Enhanced</h4>
                <p><strong>Type:</strong> Optimized CSP</p>
                <p><strong>Strategy:</strong> Forward checking + MRV heuristic</p>
                <p><strong>Pros:</strong> Fast & guaranteed</p>
                <p><strong>Cons:</strong> More complex</p>
                <p style='font-style: italic;'>Best for optimal systematic search</p>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("<h3 style='color: #7d945d; font-weight: 600;'>📊 Algorithm Comparison Matrix</h3>", unsafe_allow_html=True)
        comparison_df = pd.DataFrame({
            'Feature': ['Completeness', 'Optimality', 'Time Complexity', 'Space Complexity', 'Predictability'],
            'Hill Climbing': ['❌ No', '❌ No', '🟢 O(k)', '🟢 O(1)', '🟡 Low'],
            'CSP Backtracking': ['✅ Yes', '✅ Yes', '🔴 O(n!)', '🟡 O(n)', '🟢 High'],
            'CSP Enhanced': ['✅ Yes', '✅ Yes', '🟡 O(n²)', '🟡 O(n)', '🟢 High']
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div style='padding: 1rem; background-color: #7d945d; border-radius: 12px; color: #eeeed5; text-align: center;'>
            👈 Select an algorithm from the sidebar to see it in action!
            </div>
            """, unsafe_allow_html=True
        )

    # Algorithm pages
    elif page == "⛰️ Hill Climbing":
        show_hill_climbing_page()
    elif page == "🔙 CSP Backtracking":
        show_csp_backtracking_page()
    elif page == "⚡ CSP Enhanced":
        show_csp_enhanced_page()
    elif page == "📊 Comparison":
        show_comparison_page()

    # Footer Badge
    st.markdown("---")
    st.markdown(
        """
        <div style='padding: 1rem; background-color: #7d945d; border-radius: 12px; color: #eeeed5; text-align: center;'>
        <p style='margin: 0; font-weight: 600;'>♛ 8-Queens AI Solver ♛</p>
        <p style='margin: 0.3rem 0 0 0;'>Built with ❤️ using Streamlit & Python</p>
        <p style='margin: 0.2rem 0 0 0;'>Algorithms: Hill Climbing • CSP Backtracking • Enhanced CSP (FC + MRV)</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
