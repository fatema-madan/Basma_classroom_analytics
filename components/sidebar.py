from pathlib import Path
import streamlit as st


def render_sidebar():
    """
    Render the BASMA sidebar and return the selected page.
    """

    logo_path = Path("assets/basma_logo.png")
    small_logo_path = Path("assets/basma_logo_small.png")

    with st.sidebar:

        # Display the main BASMA logo
        if logo_path.exists():
            st.image(
                str(logo_path),
                use_container_width=True
            )

        # Use the small logo if the main logo is unavailable
        elif small_logo_path.exists():
            st.image(
                str(small_logo_path),
                width=160
            )

        # Fallback text logo
        else:
            st.markdown("""
            <div style="
                text-align: center;
                color: #527B68;
                margin-bottom: 25px;
            ">
                <h2>BASMA</h2>
                <small>AI Classroom Analytics</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            '<div class="nav-title">Workspace</div>',
            unsafe_allow_html=True
        )

        # Sidebar navigation
        selected_page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard",
                "📹 Live Classroom",
                "👤 Student Profile",
                "📊 Analytics",
                "⚙️ Settings"
            ],
            label_visibility="collapsed"
        )

        st.markdown("""
        <div class="status-box">
            <span class="status-dot">●</span>
            &nbsp; System Online
            <br><br>
            👥 1,248 registered students
        </div>
        """, unsafe_allow_html=True)

    return selected_page
