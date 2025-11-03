import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path
import numpy as np
import pandas as pd

# Color configuration
bg_color = "#1e1e2f"
primary_color = "#f4f4f4"
secondary_color = "#ff6f61"
third_color = "#ffc24a"

# Set font
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_rounded_bar_right(x, y, width, height, radius=0.8):
    """Creates a bar with rounded corners only on the right side"""
    left = x
    right = x + width
    bottom = y - height/2
    top = y + height/2
    
    r = min(radius * height, width/2, height/2)
    
    verts = [
        (left, bottom),
        (left, top),
        (right - r, top),
        (right, top),
        (right, top - r),
        (right, bottom + r),
        (right, bottom),
        (right - r, bottom),
        (left, bottom),
    ]
    
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CLOSEPOLY,
    ]
    
    return Path(verts, codes)

def plot_possession_horizontal_bar(df, team_name, title="BALL POSSESSION"):
    """
    Creates a horizontal bar chart for ball possession
    
    Args:
        df: DataFrame with team data
        team_name: Name of the team to highlight
        title: Chart title
    """
    # Assuming df has columns for team and possession
    if ('team', '') in df.columns:
        team_col = ('team', '')
        poss_col = ('Poss', '')
    else:
        team_col = 'team'
        poss_col = 'Poss'
    
    df_sorted = df.sort_values(by=poss_col, ascending=False).set_index(team_col)
    
    fig, ax = plt.subplots(figsize=(6, 10), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')
    
    bar_height = 0.6
    
    for i, (team, row) in enumerate(df_sorted.iterrows()):
        value = row[poss_col]
        color = secondary_color if team == team_name else primary_color
        
        ax.barh(i, value, height=bar_height, color=color, edgecolor='none')
        ax.text(-2, i, team, va='center', ha='right', color=primary_color, fontsize=12)
        ax.text(value + 1, i, f"{value:.1f} %", va='center', color=primary_color, fontsize=12)
    
    plt.title(title, fontsize=24, fontweight='bold', color=primary_color, pad=20)
    ax.set_ylim(-1, len(df_sorted))
    ax.set_xlim(-20, df_sorted[poss_col].max() + 10)
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

def plot_rounded_bar_chart(df, team_name, value_col, title, ascending=True, format_value=None):
    """
    Creates a horizontal bar chart with rounded bars
    
    Args:
        df: DataFrame with team data
        team_name: Name of the team to highlight
        value_col: Column name for values
        title: Chart title
        ascending: Sort order
        format_value: Function to format values for display
    """
    # Handle multi-index columns
    if isinstance(value_col, tuple):
        sort_col = value_col
    else:
        sort_col = value_col
    
    # Add highlight column
    df['_is_highlight'] = (df[('team', '')] == team_name).astype(int)
    df_sorted = df.sort_values(
        by=[sort_col, '_is_highlight'], 
        ascending=[ascending, False]
    ).drop('_is_highlight', axis=1)
    
    df_sorted = df_sorted.set_index(('team', ''))
    
    fig, ax = plt.subplots(figsize=(6, 10), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')
    
    bar_height = 0.6
    n_teams = len(df_sorted)
    
    for i, (team, row) in enumerate(df_sorted.iterrows()):
        y_position = n_teams - 1 - i
        value = row[sort_col]
        color = secondary_color if team == team_name else primary_color
        
        path = create_rounded_bar_right(0, y_position, value, bar_height, radius=0.3)
        patch = PathPatch(path, facecolor=color, edgecolor='none')
        ax.add_patch(patch)
        
        ax.text(-2, y_position, team, va='center', ha='right', color=primary_color, fontsize=12)
        
        if format_value:
            display_value = format_value(value)
        else:
            display_value = f"{value}"
        
        ax.text(value + 1, y_position, display_value, va='center', color=primary_color, fontsize=12)
    
    plt.title(title, fontsize=24, fontweight='bold', color=primary_color, pad=20)
    ax.set_ylim(-1, n_teams)
    ax.set_xlim(-20, df_sorted[sort_col].max() + 10)
    
    plt.tight_layout()
    return fig

def plot_fouls_per_90(df, team_name, title="FOULS COMMITTED /90"):
    """
    Creates a bar chart for fouls committed per 90 minutes
    
    Args:
        df: DataFrame with team data
        team_name: Name of the team to highlight
        title: Chart title
    """
    return plot_rounded_bar_chart(
        df, team_name, ('Performance', 'Fls'), title, 
        ascending=True, format_value=lambda x: f"{round(x/34, 1)}"
    )

def plot_player_minutes_bar(df, player_name, title="PLAYING TIME"):
    """
    Creates a horizontal bar chart for player minutes
    
    Args:
        df: DataFrame with player data
        player_name: Name of the player to highlight
        title: Chart title
    """
    df_sorted = df.sort_values(by=[('Playing Time', 'Min')], ascending=[False])
    df_sorted = df_sorted.set_index(('player', ''))
    
    fig, ax = plt.subplots(figsize=(10, 12), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    bar_height = 0.8
    bar_spacing = 1.2
    n_players = len(df_sorted)
    
    for i, (player, row) in enumerate(df_sorted.iterrows()):
        y_position = (n_players - 1 - i) * bar_spacing
        value = row[('Playing Time', 'Min')]
        color = secondary_color if player == player_name else primary_color
        
        rect = Rectangle((0, y_position - bar_height/2), 
                        value, 
                        bar_height,
                        facecolor=color,
                        edgecolor='none',
                        alpha=0.9)
        ax.add_patch(rect)
        
        ax.text(-50, y_position, player, 
                va='center', ha='right', 
                color=primary_color, 
                fontsize=16,
                fontweight='normal')
        
        if value > 300:
            ax.text(value - 50, y_position, f"{int(value)}", 
                    va='center', ha='right',
                    color=bg_color,
                    fontsize=16,
                    fontweight='bold')
        else:
            ax.text(value + 50, y_position, f"{int(value)}", 
                    va='center', ha='left',
                    color=primary_color,
                    fontsize=16,
                    fontweight='bold')
    
    plt.title(title, fontsize=48, fontweight='bold', 
              color=primary_color, pad=40, loc='center')
    
    ax.set_ylim(-bar_spacing, n_players * bar_spacing)
    max_value = df_sorted[('Playing Time', 'Min')].max()
    ax.set_xlim(-800, max_value + 200)
    
    # Remove axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add grid
    ax.set_xticks(range(0, int(max_value) + 500, 500))
    ax.set_xticklabels([str(x) for x in range(0, int(max_value) + 500, 500)], 
                       color=primary_color, fontsize=14)
    ax.tick_params(axis='x', colors=primary_color, which='both', top=False, bottom=True)
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(True, axis='x', linestyle='-', alpha=0.1, color=primary_color)
    ax.set_yticks([])
    ax.grid(False, axis='y')
    ax.xaxis.set_label_position('bottom')
    
    plt.tight_layout()
    return fig

def plot_minutes_per_match(match_days, minutes, title="Minutes played per match"):
    """
    Creates a bar chart for minutes played per match
    
    Args:
        match_days: List of match days
        minutes: List of minutes played
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    bar_width = 0.7
    bar_color = '#90ee90'
    
    bars = ax.bar(match_days, minutes, width=bar_width, color=bar_color, 
                   edgecolor='none', alpha=0.9)
    
    for i, (day, min_played) in enumerate(zip(match_days, minutes)):
        ax.text(day, min_played - 3, str(min_played), 
                ha='center', va='top', color=bg_color, 
                fontsize=14, fontweight='bold')
    
    ax.text(len(match_days)/2, max(minutes) + 15, title, 
            ha='center', va='center', color=primary_color, 
            fontsize=32, fontweight='bold')
    
    ax.set_xlim(0, len(match_days) + 1)
    ax.set_ylim(0, max(minutes) + 30)
    
    ax.set_xticks(match_days)
    ax.set_xticklabels([str(d) for d in match_days], 
                       color=primary_color, fontsize=12)
    
    ax.set_yticks(range(0, 100, 10))
    ax.set_yticklabels([str(y) for y in range(0, 100, 10)], 
                       color=primary_color, fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(primary_color)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_color(primary_color)
    ax.spines['bottom'].set_alpha(0.3)
    
    ax.grid(True, axis='y', linestyle='-', alpha=0.1, color=primary_color)
    ax.grid(False, axis='x')
    
    ax.tick_params(axis='both', colors=primary_color, labelsize=12)
    ax.tick_params(axis='x', length=0)
    
    plt.tight_layout()
    return fig

def plot_rounded_minutes_per_match(match_days, minutes, title="Minutes played per match"):
    """
    Creates a bar chart with rounded bars for minutes played per match
    
    Args:
        match_days: List of match days
        minutes: List of minutes played
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    bar_width = 0.6
    bar_color = '#90ee90'
    
    for i, (day, min_played) in enumerate(zip(match_days, minutes)):
        rect = FancyBboxPatch(
            (day - bar_width/2, 0),
            bar_width,
            min_played,
            boxstyle="round,pad=0.05",
            facecolor=bar_color,
            edgecolor='none',
            alpha=0.9
        )
        ax.add_patch(rect)
        
        ax.text(day, min_played - 5, str(min_played),
                ha='center', va='top', color=bg_color,
                fontsize=20, fontweight='bold')
    
    ax.text(len(match_days)/2, max(minutes) + 15, title,
            ha='center', va='center', color=primary_color,
            fontsize=28, fontweight='bold')
    
    ax.set_xlim(0.3, len(match_days) + 0.7)
    ax.set_ylim(0, max(minutes) + 30)
    
    ax.set_xticks(match_days)
    ax.set_xticklabels([str(d) for d in match_days],
                       color=primary_color, fontsize=16)
    
    ax.set_yticks(range(0, 100, 10))
    ax.set_yticklabels([str(y) for y in range(0, 100, 10)],
                       color=primary_color, fontsize=14)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.grid(True, axis='y', linestyle='-', alpha=0.2, color=primary_color)
    ax.grid(False, axis='x')
    
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    
    plt.tight_layout()
    return fig