import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
import random
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
    """CSP with basic backtracking for the N-Queens problem."""
    
    def __init__(self, n: int = 8):
        super().__init__(n) 
        self.iterations = 0
        self.history: List[List[int]] = [] 
        self.attack_history: List[int] = [] 
        
    def is_safe(self, state: List[int], col: int, row: int) -> bool:
        """
        Check if placing a queen in the given (row, col) is safe 
        relative to the already assigned columns (c < col).
        """
        for c in range(col):
            # Check if column c is assigned. state[c] != -1
            if state[c] != -1: 
                # Check for same row or diagonal conflict
                if state[c] == row or abs(state[c] - row) == abs(c - col):
                    return False
        return True
    
    def record_history(self, state: List[int]):
        """Records the current board state and its attack count for animation."""
        self.iterations += 1
        
        # Convert -1 (unassigned) to 0 for visualization compatibility
        display_state = [s if s != -1 else 0 for s in state] 
        
        self.history.append(display_state)
        self.attack_history.append(self.count_attacks(display_state))
        
    def backtrack(self, state: List[int], col: int) -> bool:
        """
        Recursive Backtracking Search.
        """
        
        if col == self.n:
            self.record_history(state)
            return True
        
        for row in range(self.n):
            
            if self.is_safe(state, col, row):
                
                # 1. Tentative Assignment
                state[col] = row
                self.record_history(state)
                
                # 2. Recurse
                if self.backtrack(state, col + 1):
                    return True
                
                # 3. Backtrack / Unassign
                state[col] = -1 
                self.record_history(state) 
                
        return False
    
    def solve(self, initial_state: Optional[List[int]] = None) -> dict:
        """
        Starts the backtracking search. If a partial state is provided, 
        it finds the first unassigned column and starts from there.
        """
        start_time = time.time()
        self.iterations = 0
        self.history = []
        self.attack_history = []
        
        if initial_state and len(initial_state) == self.n:
            state = initial_state.copy()
            # Find the starting column (first unassigned index: -1)
            try:
                start_col = state.index(-1)
            except ValueError:
                # Board is fully assigned (e.g., if a random full state was passed, though unlikely here)
                start_col = self.n
        else:
            # Default: Standard CSP start (empty board)
            state = [-1] * self.n
            start_col = 0
        
        # Record the initial (partial or empty) state
        self.record_history(state)
        
        # Start the recursive search
        success = self.backtrack(state, start_col)
        
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
    """Hill Climbing algorithm page with smooth animation and custom initial inputs."""
    
    st.markdown('<div class="algorithm-title">⛰️ Hill Climbing with Random Restart</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        with st.container():
            board_size = st.slider("🎲 Board Size", 4, 12, 8, key="hc_size")
            max_restarts = st.slider("🔄 Max Restarts", 10, 100, 50, key="hc_restarts")
            max_iterations = st.slider("🔢 Max Iterations", 100, 1000, 500, key="hc_iters")
            animation_speed = st.slider("⚡ Animation Speed", 0.05, 1.0, 0.2, 
                                        help="Seconds between frames", key="hc_speed")
        
        st.markdown("---")
        
        # --- Custom Initial Placement Input ---
        st.markdown("### ✍️ Custom Initial Placement (Optional)")
        
        num_fixed_queens = st.slider(
            "Fix first 'k' Queens (k)", 0, board_size, 0, 
            help="Number of queens (k) to manually fix in columns 0 to k-1. The remaining N-k are randomized.",
            key="hc_fixed_k"
        )
        
        fixed_placement: List[int] = []
        is_valid_input = True
        
        if num_fixed_queens > 0:
            st.markdown(f"Select **Row Index** for Queens in Col 0 to Col {num_fixed_queens - 1} (0 to {board_size - 1}):")
            
            # Use columns to display number inputs neatly
            input_cols = st.columns(min(num_fixed_queens, 6))
            
            for i in range(num_fixed_queens):
                with input_cols[i % len(input_cols)]:
                    row_val = st.number_input(
                        f"Col {i}", 
                        min_value=0, 
                        max_value=board_size - 1, 
                        value=i % board_size, # default value
                        step=1, 
                        key=f"hc_fixed_row_{i}",
                        help=f"Row index for column {i}"
                    )
                    fixed_placement.append(int(row_val))
        
        st.session_state["hc_fixed_placement"] = fixed_placement
        # -------------------------------------
        
        st.markdown("---")
        
        st.markdown("""
        **📚 Algorithm Overview:**
        
        Hill Climbing is a **local search** algorithm:
        - 🎯 Starts from random state
        - 📈 Moves to better neighbors
        - 🔄 Uses random restarts to escape local optima
        - ⚡ Fast but not guaranteed to find solution
        """)
        
        st.markdown("---")
        
        if st.button("🚀 Run Hill Climbing", key="run_hc", use_container_width=True):
            st.session_state["hc_running"] = True
            st.rerun()
    
    with col2:
        if 'hc_running' in st.session_state and st.session_state.hc_running:
            
            solver = HillClimbingSolver(board_size, max_restarts, max_iterations)
            
            # --- Generate Initial State based on User Input ---
            fixed_placement = st.session_state.get("hc_fixed_placement", [])
            num_fixed = len(fixed_placement)
            
            # Start with the fixed part
            initial_state = fixed_placement[:]
            
            # Randomly fill the remaining N-k columns
            remaining_cols = board_size - num_fixed
            random_rows = [random.randint(0, board_size - 1) for _ in range(remaining_cols)]
            initial_state.extend(random_rows)
            
            # Note: Hill Climbing state is a list of N rows, one for each column.
            # We skip the conflict check here because Hill Climbing is designed to 
            # work with *any* initial state, even one full of conflicts.
            
            # --------------------------------------------------

            
            # Animation containers
            board_placeholder = st.empty()
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                iter_metric = st.empty()
            with stats_col2:
                attack_metric = st.empty()
            with stats_col3:
                restart_metric = st.empty()
            
            progress_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Show initial state
            initial_attacks = solver.count_attacks(initial_state)
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
            
            # Animate
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
            
            progress_placeholder.empty()
            
            # Final result
            if result['success']:
                st.balloons()
                st.success("🎉 Solution Found!")
                board_placeholder.markdown(
                    create_html_chessboard(result['solution'], "✅ Final Solution", 0),
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ No solution found within iteration limit")
                board_placeholder.markdown(
                    create_html_chessboard(result['solution'], "❌ Final Stagnant State", solver.count_attacks(result['solution'])),
                    unsafe_allow_html=True
                )
            
            # Summary metrics
            st.markdown("### 📋 Summary")
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("✅ Success", "Yes" if result['success'] else "No")
            with metric_cols[1]:
                st.metric("🔄 Restarts", result['restarts'])
            with metric_cols[2]:
                st.metric("🔢 Total Iterations", result['iterations'])
            with metric_cols[3]:
                st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
            
            with st.expander("📊 Detailed Analysis", expanded=False):
                st.plotly_chart(
                    create_convergence_plot(result['attack_history'], "Complete Convergence Path"),
                    use_container_width=True
                )
                
                st.json({
                    'Initial Attacks': initial_attacks,
                    'Final Attacks': 0 if result['success'] else solver.count_attacks(result['solution']),
                    'Total States Explored': len(history),
                    'Average Iterations per Restart': f"{result['iterations'] / max(1, result['restarts']):.1f}"
                })
            
            st.session_state.hc_result = result
            st.session_state["hc_running"] = False
def on_k_change(board_size):
    """Callback attached to the slider to reset the initial placement list when k changes."""
    # Retrieve the current value of the slider from session state
    num_k = st.session_state.csp_init_k
    
    # If the user selects 0 queens, we MUST ensure the list is empty.
    if num_k == 0:
        st.session_state["csp_initial_placement"] = []
    
    # For k > 0, we ensure the list size matches k to prevent index errors
    # if the board size was just changed and the user is moving the slider.
    # Note: The select boxes will populate this list on the next rerun.
    # The default logic will handle the values, but we ensure the list exists.
    elif "csp_initial_placement" not in st.session_state or len(st.session_state["csp_initial_placement"]) != num_k:
         # Initialize with a basic, conflict-free default (e.g., [0, 1, 2...])
         st.session_state["csp_initial_placement"] = [i % board_size for i in range(num_k)]


def on_k_change(board_size):
    """Callback attached to the slider to reset the initial placement list when k changes."""
    num_k = st.session_state.csp_init_k
    if num_k == 0:
        st.session_state["csp_initial_placement"] = []
    elif "csp_initial_placement" not in st.session_state or len(st.session_state["csp_initial_placement"]) != num_k:
         st.session_state["csp_initial_placement"] = [i % board_size for i in range(num_k)]

def check_initial_conflicts(placement: List[int]) -> bool:
    """
    Checks real queen conflicts for the N-Queens problem.
    Returns True if any two queens attack each other.
    
    Conflicts checked:
    - Same row
    - Same column
    - Same diagonal
    """
    k = len(placement)

    for col in range(k):
        row = placement[col]

        for prev_col in range(col):
            prev_row = placement[prev_col]

            # 1. Same row → conflict
            if row == prev_row:
                return True

            # 2. Same column → conflict
            if col == prev_col:
                return True

            # 3. Diagonal conflict → |Δrow| == |Δcol|
            if abs(row - prev_row) == abs(col - prev_col):
                return True
def on_csp_k_change(board_size):
    """Callback attached to the slider to reset the initial placement list when k changes for CSP."""
    num_k = st.session_state.csp_init_k
    if num_k == 0:
        st.session_state["csp_initial_placement"] = []
    elif "csp_initial_placement" not in st.session_state or len(st.session_state["csp_initial_placement"]) != num_k:
         st.session_state["csp_initial_placement"] = [i % board_size for i in range(num_k)]

# NOTE: The check_initial_conflicts helper function is REMOVED.

# --- Main Streamlit Page Function ---

def show_csp_backtracking_page():
    """CSP Backtracking page with animation and number input for custom initial placement, 
    allowing ANY initial configuration (no conflict check)."""
    
    st.markdown('<div class="algorithm-title">🔙 CSP with Backtracking</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        with st.container():
            board_size = st.slider("🎲 Board Size", 4, 12, 8, key="csp_size")
            animation_speed = st.slider("⚡ Animation Speed", 0.05, 1.0, 0.1, 
                                        help="Seconds between frames", key="csp_speed")
        
        st.markdown("---")
        
        ## ✍️ Custom Initial Placement (No Validation)
        st.markdown("### ✍️ Custom Initial Placement (Optional)")
        
        num_initial_queens = st.slider(
            "Number of Queens to Pre-assign (k)", 0, board_size, 0, 
            key="csp_init_k",
            on_change=on_csp_k_change,
            args=(board_size,),
            help="Select how many queens (k) to manually place in columns 0 to k-1."
        )
        
        initial_placement: List[int] = []
        
        if num_initial_queens > 0:
            st.markdown(f"Select **Row Index** for Queens in Col 0 to Col {num_initial_queens - 1} (0 to {board_size - 1}):")
            
            input_cols = st.columns(min(num_initial_queens, 6)) 
            
            for i in range(num_initial_queens):
                with input_cols[i % len(input_cols)]:
                    row_val = st.number_input(
                        f"Col {i}", 
                        min_value=0, 
                        max_value=board_size - 1, 
                        value=i % board_size, 
                        step=1, 
                        key=f"csp_fixed_row_{i}",
                        help=f"Row index for column {i}"
                    )
                    initial_placement.append(int(row_val))
            
            st.session_state["csp_initial_placement"] = initial_placement
        else:
            st.session_state["csp_initial_placement"] = []
            initial_placement = []
        
        st.markdown("---")
        
        st.markdown("""
        **📚 Algorithm Overview:**
        
        CSP Backtracking is a **systematic search**:
        - 🎯 Builds solution column by column
        - ✅ Checks constraints before placement
        - 🔙 Backtracks on conflicts
        - 💯 Guarantees finding solution if exists
        """)
        
        st.markdown("---")
        
        if st.button("🚀 Run CSP Backtracking", key="run_csp", use_container_width=True):
            st.session_state["csp_running"] = True
            st.rerun()
    
    with col2:
        if 'csp_running' in st.session_state and st.session_state.csp_running:
            
            # 1. Prepare the custom initial state list for the solver
            custom_state = [-1] * board_size
            current_initial_placement = st.session_state.get("csp_initial_placement", [])
            
            for i, row in enumerate(current_initial_placement):
                custom_state[i] = row
            
            # 2. **INITIAL CONFLICT CHECK IS SKIPPED**
            
            # 3. Execute the solver (Relies entirely on CSPBacktrackingSolver to handle initial conflicts)
            solver = CSPBacktrackingSolver(board_size)
            result = solver.solve(initial_state=custom_state) 
            
            # --- Animation and Results display ---
            
            board_placeholder = st.empty()
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1: iter_metric = st.empty()
            with stats_col2: attack_metric = st.empty()
            with stats_col3: col_metric = st.empty()
            
            progress_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            history = result['history']
            attack_history = result['attack_history']
            progress_bar = progress_placeholder.progress(0)
            step = max(1, len(history) // 50)
            
            for i in range(0, len(history), step):
                state = history[i]
                attacks = attack_history[i]
                
                current_col = 0
                try:
                    current_col = state.index(0) 
                except ValueError:
                    current_col = board_size
                
                board_placeholder.markdown(
                    create_html_chessboard(state, f"🔍 Iteration {i+1}/{len(history)}", 
                                            attacks, highlight_col=current_col),
                    unsafe_allow_html=True
                )
                
                iter_metric.metric("Iteration", f"{i+1}/{len(history)}")
                attack_metric.metric("Attacking Pairs", attacks)
                col_metric.metric("Column Processing", f"{current_col}/{board_size}")
                
                chart_placeholder.plotly_chart(
                    create_convergence_plot(attack_history, "📈 Convergence Progress", i),
                    use_container_width=True,
                    key=f"csp_chart_{i}"
                )
                
                progress_bar.progress((i + 1) / len(history))
                time.sleep(animation_speed)
            
            progress_placeholder.empty()
            
            # Final result
            if result['success']:
                st.balloons()
                st.success("🎉 Solution Found!")
                board_placeholder.markdown(
                    create_html_chessboard(result['solution'], "✅ Final Solution", 0),
                    unsafe_allow_html=True
                )
            else:
                # The solver failed, likely due to an impossible starting configuration
                st.error("❌ No solution found (check if your initial setup was impossible)")
            
            # Summary metrics
            st.markdown("### 📋 Summary")
            metric_cols = st.columns(3)
            with metric_cols[0]: st.metric("✅ Success", "Yes" if result['success'] else "No")
            with metric_cols[1]: st.metric("🔢 Total Iterations", result['iterations'])
            with metric_cols[2]: st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
            
            with st.expander("📊 Detailed Analysis", expanded=False):
                st.plotly_chart(
                    create_convergence_plot(result['attack_history'], "Complete Search Path"),
                    use_container_width=True
                )
                st.json({
                    'Method': result['method'],
                    'Total States Explored': len(history),
                    'Final Attacks': 0 if result['success'] else 'N/A',
                    'Search Efficiency': f"{result['iterations'] / board_size:.2f} iterations/column"
                })
            
            st.session_state.csp_result = result
            st.session_state["csp_running"] = False
def show_csp_enhanced_page():
    """Enhanced CSP page with animation."""
    st.markdown('<div class="algorithm-title">⚡ CSP with Forward Checking & MRV</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        with st.container():
            board_size = st.slider("🎲 Board Size", 4, 12, 8, key="cspe_size")
            animation_speed = st.slider("⚡ Animation Speed", 0.05, 1.0, 0.1, 
                                       help="Seconds between frames", key="cspe_speed")
        
        st.markdown("---")
        
        st.markdown("""
        **📚 Algorithm Overview:**
        
        Enhanced CSP with **intelligent heuristics**:
        - 🎯 **Forward Checking:** Prunes domains early
        - 🧠 **MRV:** Picks most constrained variable
        - ⚡ More efficient than basic backtracking
        - 💯 Guarantees optimal search path
        """)
        
        st.markdown("---")
        
        if st.button("🚀 Run Enhanced CSP", key="run_cspe", use_container_width=True):
            # st.session_state.cspe_running = True
            st.session_state["cspe_running"] = True
    
    with col2:
        if 'cspe_running' in st.session_state and st.session_state.cspe_running:
            solver = CSPEnhancedSolver(board_size)
            
            # Animation containers
            board_placeholder = st.empty()
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                iter_metric = st.empty()
            with stats_col2:
                attack_metric = st.empty()
            with stats_col3:
                efficiency_metric = st.empty()
            
            progress_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Run algorithm
            result = solver.solve()
            
            # Animate
            history = result['history']
            attack_history = result['attack_history']
            
            progress_bar = progress_placeholder.progress(0)
            step = max(1, len(history) // 50)
            
            for i in range(0, len(history), step):
                state = history[i]
                attacks = attack_history[i]
                
                current_col = sum(1 for x in state if x != 0)
                
                board_placeholder.markdown(
                    create_html_chessboard(state, f"🧠 Iteration {i+1}/{len(history)}", 
                                         attacks, highlight_col=current_col-1 if current_col > 0 else -1),
                    unsafe_allow_html=True
                )
                
                iter_metric.metric("Iteration", f"{i+1}/{len(history)}")
                attack_metric.metric("Attacking Pairs", attacks)
                efficiency_metric.metric("Efficiency", f"{(i+1)/board_size:.1f}x")
                
                chart_placeholder.plotly_chart(
                    create_convergence_plot(attack_history, "📈 Convergence Progress", i),
                    use_container_width=True,
                    key=f"cspe_chart_{i}"
                )
                
                progress_bar.progress((i + 1) / len(history))
                time.sleep(animation_speed)
            
            progress_placeholder.empty()
            
            # Final result
            if result['success']:
                st.balloons()
                st.success("🎉 Solution Found!")
                board_placeholder.markdown(
                    create_html_chessboard(result['solution'], "✅ Final Solution", 0),
                    unsafe_allow_html=True
                )
            else:
                st.error("❌ No solution found")
            
            # Summary metrics
            st.markdown("### 📋 Summary")
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("✅ Success", "Yes" if result['success'] else "No")
            with metric_cols[1]:
                st.metric("🔢 Total Iterations", result['iterations'])
            with metric_cols[2]:
                st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
            
            with st.expander("📊 Detailed Analysis", expanded=False):
                st.plotly_chart(
                    create_convergence_plot(result['attack_history'], "Complete Search Path"),
                    use_container_width=True
                )
                
                st.json({
                    'Method': result['method'],
                    'Total States Explored': len(history),
                    'Final Attacks': 0 if result['success'] else 'N/A',
                    'Heuristic Efficiency': f"{result['iterations'] / board_size:.2f}x optimal"
                })
            
            st.session_state.cspe_result = result
def on_cspe_k_change(board_size):
    """Callback attached to the slider to reset the initial placement list when k changes for CSPE."""
    # Note the unique key: cspe_init_k
    num_k = st.session_state.cspe_init_k
    
    if num_k == 0:
        # Clear the stored placement list if k is 0
        st.session_state["cspe_initial_placement"] = []
    
    elif "cspe_initial_placement" not in st.session_state or len(st.session_state["cspe_initial_placement"]) != num_k:
         # Initialize with a basic, conflict-free default (e.g., [0, 1, 2...])
         st.session_state["cspe_initial_placement"] = [i % board_size for i in range(num_k)]

def show_csp_enhanced_page():
    """Enhanced CSP page with animation and number input for custom initial placement, 
    with initial conflict check removed to prevent AttributeError."""
    st.markdown('<div class="algorithm-title">⚡ CSP with Forward Checking & MRV</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        with st.container():
            board_size = st.slider("🎲 Board Size", 4, 12, 8, key="cspe_size")
            animation_speed = st.slider("⚡ Animation Speed", 0.05, 1.0, 0.1, 
                                        help="Seconds between frames", key="cspe_speed")
        
        st.markdown("---")
        
        # --- Custom Initial Placement using Number Inputs ---
        st.markdown("### ✍️ Custom Initial Placement (Optional)")
        
        num_initial_queens = st.slider(
            "Number of Queens to Pre-assign (k)", 0, board_size, 0, 
            help="Select how many queens (k) to manually place in columns 0 to k-1.",
            key="cspe_init_k"
        )
        
        initial_placement: List[int] = []
        
        if num_initial_queens > 0:
            st.markdown(f"Select **Row Index** for Queens in Col 0 to Col {num_initial_queens - 1} (0 to {board_size - 1}):")
            
            input_cols = st.columns(min(num_initial_queens, 6))
            
            for i in range(num_initial_queens):
                with input_cols[i % len(input_cols)]:
                    row_val = st.number_input(
                        f"Col {i}", 
                        min_value=0, 
                        max_value=board_size - 1, 
                        value=i % board_size, 
                        step=1, 
                        key=f"cspe_fixed_row_{i}",
                        help=f"Row index for column {i}"
                    )
                    initial_placement.append(int(row_val))
        
        st.session_state["cspe_initial_placement"] = initial_placement
        # ---------------------------------------------
        
        st.markdown("---")
        
        st.markdown("""
        **📚 Algorithm Overview:**
        
        Enhanced CSP with **intelligent heuristics**:
        - 🎯 **Forward Checking:** Prunes domains early
        - 🧠 **MRV (Minimum Remaining Values):** Picks most constrained variable
        - ⚡ More efficient than basic backtracking
        - 💯 Guarantees finding a solution if one exists
        """)
        
        st.markdown("---")
        
        if st.button("🚀 Run Enhanced CSP", key="run_cspe", use_container_width=True):
            st.session_state["cspe_running"] = True
            st.rerun()
    
    with col2:
        if 'cspe_running' in st.session_state and st.session_state.cspe_running:
            
            # 1. Prepare the custom initial state
            custom_state = [-1] * board_size
            current_initial_placement = st.session_state.get("cspe_initial_placement", [])
            
            for i, row in enumerate(current_initial_placement):
                custom_state[i] = row
            
            # 2. **CONFLICT CHECK SECTION REMOVED** to prevent the AttributeError.
            # We assume the CSPEnhancedSolver can handle a potentially invalid starting state.
            
            # 3. Execute the solver with the partial state
            solver = CSPEnhancedSolver(board_size)
            result = solver.solve(initial_state=custom_state) 
            
            # --- Animation and Results display ---
            
            board_placeholder = st.empty()
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1: iter_metric = st.empty()
            with stats_col2: attack_metric = st.empty()
            with stats_col3: efficiency_metric = st.empty()
            
            progress_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            history = result['history']
            attack_history = result['attack_history']
            
            progress_bar = progress_placeholder.progress(0)
            step = max(1, len(history) // 50)
            
            for i in range(0, len(history), step):
                state = history[i]
                attacks = attack_history[i]
                
                # Find current column being processed
                current_col = 0
                try:
                    # Assumes 0 is used for unassigned state in history
                    current_col = state.index(0) 
                except ValueError:
                    current_col = board_size
                
                board_placeholder.markdown(
                    create_html_chessboard(state, f"🧠 Iteration {i+1}/{len(history)}", 
                                            attacks, highlight_col=current_col),
                    unsafe_allow_html=True
                )
                
                iter_metric.metric("Iteration", f"{i+1}/{len(history)}")
                attack_metric.metric("Attacking Pairs", attacks)
                efficiency_metric.metric("Efficiency", f"{(i+1)/board_size:.1f}x")
                
                chart_placeholder.plotly_chart(
                    create_convergence_plot(attack_history, "📈 Convergence Progress", i),
                    use_container_width=True,
                    key=f"cspe_chart_{i}"
                )
                
                progress_bar.progress((i + 1) / len(history))
                time.sleep(animation_speed)
            
            progress_placeholder.empty()
            
            # Final result
            if result['success']:
                st.balloons()
                st.success("🎉 Solution Found!")
                board_placeholder.markdown(
                    create_html_chessboard(result['solution'], "✅ Final Solution", 0),
                    unsafe_allow_html=True
                )
            else:
                st.error("❌ No solution found")
            
            # Summary metrics
            st.markdown("### 📋 Summary")
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("✅ Success", "Yes" if result['success'] else "No")
            with metric_cols[1]:
                st.metric("🔢 Total Iterations", result['iterations'])
            with metric_cols[2]:
                st.metric("⏱️ Runtime", f"{result['runtime']*1000:.2f} ms")
            
            with st.expander("📊 Detailed Analysis", expanded=False):
                st.plotly_chart(
                    create_convergence_plot(result['attack_history'], "Complete Search Path"),
                    use_container_width=True
                )
                
                st.json({
                    'Method': result['method'],
                    'Total States Explored': len(history),
                    'Final Attacks': 0 if result['success'] else 'N/A',
                    'Heuristic Efficiency': f"{result['iterations'] / board_size:.2f}x optimal"
                })
            
            st.session_state.cspe_result = result
            st.session_state["cspe_running"] = False
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
    # Main heading with authors
    st.markdown(
        """
        <h1 style='text-align: center; color: #1b4d3e; font-family: "Georgia, serif";'>
        ♛ 8-Queens AI Solver ♛
        </h1>
        <h4 style='text-align: center; color: #2e7d32; font-family: "Arial, sans-serif";'>
        CEP Project by Jibran Saleem, Sajjad Abidi & Arif Mehdi
        </h4>
        """, unsafe_allow_html=True
    )

    # Intro section
    st.markdown(
        """
        <div style='text-align: center; padding: 1.5rem; background-color: #e8f5e9;
        border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
        <h3 style='color: #1b5e20; margin-bottom: 0.5rem; font-family: "Arial, sans-serif";'>
        Explore AI Algorithms Solving the Classic 8-Queens Problem
        </h3>
        <p style='color: #2e7d32; margin: 0;'>
        Compare Hill Climbing, CSP Backtracking, and Enhanced CSP with interactive visualization.
        </p>
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
        ### 📚 About 8-Queens
        Place 8 queens on an 8×8 board so no two queens attack each other.

        **Constraints:**
        - ❌ No two queens in same row
        - ❌ No two queens in same column
        - ❌ No two queens on same diagonal

        **Total Solutions:** 92 unique solutions
        """
    )
    st.sidebar.markdown("---")

    if page != "🏠 Home":
        st.sidebar.markdown("### 🎯 Quick Tips")
        st.sidebar.info(
            """
            - Adjust animation speed slider
            - Use smaller board sizes (4-8) for faster visualization
            - Watch live metrics while the algorithm runs
            """
        )

    # Home Page
    if page == "🏠 Home":
        st.markdown("### 🎓 Welcome to the 8-Queens Solver")
        st.markdown(
            """
            This application demonstrates AI algorithms solving the classic 8-Queens problem 
            with **smooth animations** and **detailed visualizations**.
            """
        )
        st.markdown("---")
        st.markdown("### 🚀 Choose an Algorithm to Explore")

        col1, col2, col3 = st.columns(3)

        # Hill Climbing
        with col1:
            st.markdown(
                """
                <div style='padding: 1.2rem; background-color: #a5d6a7; border-radius: 10px; 
                border-left: 4px solid #1b5e20; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4>⛰️ Hill Climbing</h4>
                <p><strong>Type:</strong> Local Search</p>
                <p><strong>Strategy:</strong> Greedy optimization with random restarts</p>
                <p><strong>Pros:</strong> Fast, simple</p>
                <p><strong>Cons:</strong> May get stuck in local maxima</p>
                <p><em>Best for quick approximate solutions</em></p>
                </div>
                """, unsafe_allow_html=True
            )

        # CSP Backtracking
        with col2:
            st.markdown(
                """
                <div style='padding: 1.2rem; background-color: #c8e6c9; border-radius: 10px; 
                border-left: 4px solid #2e7d32; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4>🔙 CSP Backtracking</h4>
                <p><strong>Type:</strong> Systematic Search</p>
                <p><strong>Strategy:</strong> Constraint satisfaction with backtracking</p>
                <p><strong>Pros:</strong> Guaranteed solution</p>
                <p><strong>Cons:</strong> Can be slower</p>
                <p><em>Best for complete solutions</em></p>
                </div>
                """, unsafe_allow_html=True
            )

        # CSP Enhanced
        with col3:
            st.markdown(
                """
                <div style='padding: 1.2rem; background-color: #e0f2f1; border-radius: 10px; 
                border-left: 4px solid #388e3c; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4>⚡ CSP Enhanced</h4>
                <p><strong>Type:</strong> Optimized CSP</p>
                <p><strong>Strategy:</strong> Forward checking + MRV heuristic</p>
                <p><strong>Pros:</strong> Fast & guaranteed</p>
                <p><strong>Cons:</strong> More complex</p>
                <p><em>Best for optimal systematic search</em></p>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 📊 Algorithm Comparison Matrix")
        comparison_df = pd.DataFrame({
            'Feature': ['Completeness', 'Optimality', 'Time Complexity', 'Space Complexity', 'Predictability'],
            'Hill Climbing': ['❌ No', '❌ No', '🟢 O(k)', '🟢 O(1)', '🟡 Low'],
            'CSP Backtracking': ['✅ Yes', '✅ Yes', '🔴 O(n!)', '🟡 O(n)', '🟢 High'],
            'CSP Enhanced': ['✅ Yes', '✅ Yes', '🟡 O(n²)', '🟡 O(n)', '🟢 High']
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.markdown("---")
        st.info("👈 Select an algorithm from the sidebar to see it in action!")

    # Algorithm pages
    elif page == "⛰️ Hill Climbing":
        show_hill_climbing_page()
    elif page == "🔙 CSP Backtracking":
        show_csp_backtracking_page()
    elif page == "⚡ CSP Enhanced":
        show_csp_enhanced_page()
    elif page == "📊 Comparison":
        show_comparison_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #2e7d32; padding: 1rem; background-color: #e8f5e9; border-radius: 8px;'>
        <p style='margin: 0; font-size: 0.9rem;'><strong>8-Queens AI Solver</strong> | Built with ❤️ using Streamlit & Python</p>
        <p style='margin: 0.2rem 0 0 0; font-size: 0.8rem;'>Algorithms: Hill Climbing • CSP Backtracking • Enhanced CSP (FC + MRV)</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

