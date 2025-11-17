"""
UI Utilities for MPCIM Dashboard
Provides common styling and UI components
"""

import streamlit as st


def apply_styles():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 600;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 500;
        margin-top: 2rem;
    }
    
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    /* Cards */
    .stAlert {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Dataframes */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .stSuccess {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    
    .stWarning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    
    .stError {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #2196f3;
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title, subtitle=None, icon=None):
    """
    Create a consistent page header with gradient background.
    
    Args:
        title (str): Main title
        subtitle (str, optional): Subtitle text
        icon (str, optional): Emoji icon
    """
    icon_html = f'<span style="font-size: 2.5em; margin-right: 15px;">{icon}</span>' if icon else ''
    subtitle_html = f'<p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 1.1em;">{subtitle}</p>' if subtitle else ''
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center;">
        {icon_html}
        <h1 style="color: white; margin: 0; font-size: 2.2em; display: inline-block;">{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, delta=None, help_text=None, color="blue"):
    """
    Create a styled metric card.
    
    Args:
        label (str): Metric label
        value: Metric value
        delta: Change indicator
        help_text (str, optional): Help tooltip
        color (str): Color theme (blue, green, red, orange)
    """
    color_map = {
        "blue": "#2196f3",
        "green": "#4caf50",
        "red": "#f44336",
        "orange": "#ff9800",
        "purple": "#9c27b0"
    }
    
    bg_color = color_map.get(color, "#2196f3")
    
    delta_html = ""
    if delta is not None:
        delta_color = "#4caf50" if delta >= 0 else "#f44336"
        delta_symbol = "▲" if delta >= 0 else "▼"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.9em; margin-top: 5px;">{delta_symbol} {delta}</div>'
    
    help_html = f'<div style="font-size: 0.8em; color: #666; margin-top: 5px;">{help_text}</div>' if help_text else ''
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {bg_color}15 0%, {bg_color}05 100%);
                border-left: 4px solid {bg_color};
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;">
        <div style="color: #666; font-size: 0.9em; font-weight: 500;">{label}</div>
        <div style="color: #333; font-size: 1.8em; font-weight: 600; margin-top: 5px;">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """, unsafe_allow_html=True)


def info_box(title, content, box_type="info"):
    """
    Create an info box with icon.
    
    Args:
        title (str): Box title
        content (str): Box content
        box_type (str): Type of box (info, success, warning, error)
    """
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    colors = {
        "info": "#2196f3",
        "success": "#4caf50",
        "warning": "#ff9800",
        "error": "#f44336"
    }
    
    icon = icons.get(box_type, "ℹ️")
    color = colors.get(box_type, "#2196f3")
    
    st.markdown(f"""
    <div style="background: {color}15;
                border-left: 4px solid {color};
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;">
        <div style="font-weight: 600; color: {color}; margin-bottom: 8px;">
            {icon} {title}
        </div>
        <div style="color: #333; line-height: 1.6;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def gradient_text(text, gradient="blue-purple"):
    """
    Create gradient colored text.
    
    Args:
        text (str): Text to display
        gradient (str): Gradient type
    """
    gradients = {
        "blue-purple": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "green-blue": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        "pink-orange": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "yellow-red": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
    }
    
    gradient_css = gradients.get(gradient, gradients["blue-purple"])
    
    st.markdown(f"""
    <h2 style="background: {gradient_css};
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               font-weight: 600;">
        {text}
    </h2>
    """, unsafe_allow_html=True)
