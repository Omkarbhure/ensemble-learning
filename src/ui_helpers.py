import os
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(CURRENT_DIR, "static")
TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")


def load_css(filename: str = "style.css") -> str:
    """Load CSS file content from static/css directory."""
    css_path = os.path.join(STATIC_DIR, "css", filename)
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def apply_custom_css(filename: str = "style.css"):
    """Inject external CSS into the Streamlit app."""
    css = load_css(filename)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def load_template(template_name: str) -> str:
    """Load an HTML template from templates directory without stale caching."""
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def render_template(template_name: str, **kwargs):
    """Render an HTML template with variable replacements into Streamlit safely."""
    template = load_template(template_name)
    if template:
        safe_kwargs = {
            "category": kwargs.get("category_upper", kwargs.get("category", "")),
            "category_upper": kwargs.get("category_upper", kwargs.get("category", "").upper()),
            "category_lower": kwargs.get("category_lower", kwargs.get("category", "").lower()),
            "category_detail": kwargs.get("category_detail", ""),
            **kwargs,
        }
        try:
            rendered = template.format(**safe_kwargs)
            st.markdown(rendered, unsafe_allow_html=True)
        except Exception:
            rendered = template
            for k, v in safe_kwargs.items():
                rendered = rendered.replace(f"{{{k}}}", str(v))
            st.markdown(rendered, unsafe_allow_html=True)
