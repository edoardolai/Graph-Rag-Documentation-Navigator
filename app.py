import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import query_engine with better error handling
try:
    from query_engine import query_codebase

    import_success = True
except ImportError as e:
    st.error(f"Failed to import query_engine module: {str(e)}")
    st.error(
        "Please make sure query_engine.py is in the same directory as this file and all dependencies are installed."
    )
    import_success = False

# Set page configuration
st.set_page_config(page_title="Code Navigator", layout="wide")

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2c3e50, #34495e);
            padding: 20px;
            border-radius: 8px;
        }
        .sidebar .stRadio > div {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding: 2px 5px;
            margin-bottom: 5px;
        }
        .sidebar .stRadio label {
            color: white !important;
            font-size: 16px;
            font-weight: 500;
        }
    </style>
""",
    unsafe_allow_html=True,
)
page = st.sidebar.radio("", ["Home", "About", "Settings"])

# Custom Styling - Modern Tech Theme
st.markdown(
    """
    <style>
        /* Main styles */
        body {
            background-color: #f8f9fa;
            color: #333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Header styling */
        .main-title {
            text-align: center;
            font-size: 32px;
            font-weight: 600;
            color: #2c3e50;
            padding: 15px 0;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
        }
        
        /* Content container */
        .content-container {
            max-width: 900px;
            margin: auto;
            padding: 10px;
        }
        
        /* Question styling - Code inspired */
        .question-box {
            padding: 18px;
            border-radius: 8px;
            border-left: 5px solid #3498db;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            font-weight: 500;
            color: white;
            margin-bottom: 15px;
            font-family: 'Segoe UI', sans-serif;
        }
        
        /* Answer styling - Documentation inspired */
        .answer-box {
            padding: 20px;
            border-radius: 8px;
            background-color: #ededed;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            color: #000;
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 25px;
        }
        
        .answer-box b {
            color: #2980b9;
        }
        
        /* Query box styling */
        .query-box {
            background-color: #2d3436;
            color: #74b9ff;
            font-family: 'Courier New', monospace;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            line-height: 1.4;
        }
        
        /* Code snippets in context */
        .file-header {
            background-color: #dfe6e9;
            color: #2c3e50;
            padding: 8px 10px;
            border-radius: 5px 5px 0 0;
            font-weight: 600;
            font-size: 14px;
            border-bottom: 1px solid #b2bec3;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f1f2f6 !important;
            border-radius: 5px !important;
        }
        
        /* Submit button */
        .stButton>button {
            background-color: #3498db !important;
            color: white !important;
            border-radius: 5px !important;
            border: none !important;
            padding: 10px 20px !important;
            font-weight: 500 !important;
            transition: all 0.3s !important;
        }
        
        .stButton>button:hover {
            background-color: #2980b9 !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Input field */
        .stTextInput>div>div>input {
            border-radius: 5px !important;
            border: 1px solid #bdc3c7 !important;
            padding: 12px !important;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #3498db !important;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2) !important;
        }
        
        /* About and Settings pages */
        .info-container {
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border-top: 4px solid #3498db;
        }
        
        /* Divider */
        .stDivider {
            margin-top: 30px !important;
            margin-bottom: 30px !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# Check environment variables
def check_environment():
    missing_vars = []
    for var in ["NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD", "OPEN_AI_API_KEY"]:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.info(
            "Please ensure all required environment variables are set in your .env file"
        )
        return False
    return True


# Main Section
if page == "Home":
    st.markdown("<h1 class='main-title'>Code Navigator</h1>", unsafe_allow_html=True)
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)

    if not import_success:
        st.error(
            "Cannot process questions because the query_engine module failed to import."
        )
        st.info(
            """
        To fix this issue:
        1. Make sure you have installed all required packages: `pip install -r requirements.txt`
        2. Verify that query_engine.py is in the same directory as this file
        3. Check that all imports in query_engine.py are working correctly
        """
        )
    else:
        # Session state to handle multiple questions
        if "question_history" not in st.session_state:
            st.session_state.question_history = []

        # Display previous questions and answers
        for index, (q, a, query, context) in enumerate(
            reversed(st.session_state.question_history)
        ):
            st.markdown(
                f"<div class='question-box'>🔍 Question: {q}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='answer-box'><b>Answer:</b><br>{a}</div>",
                unsafe_allow_html=True,
            )

            with st.expander("View Cypher Query", expanded=False):
                st.markdown(
                    f"<div class='query-box'>{query}</div>", unsafe_allow_html=True
                )

            with st.expander("View Context", expanded=False):
                for file, snippets in context.items():
                    st.markdown(
                        f"<p class='file-header'>{file}</p>", unsafe_allow_html=True
                    )
                    for snippet in snippets:
                        st.code(snippet, language="python")

            st.divider()  # Keep structure clean and separate questions

        # User input for a new question
        st.subheader("Ask a new question")
        question = st.text_input(
            "",
            placeholder="Type your question about the CodeCarbon project here...",
            key=f"question_{len(st.session_state.question_history)}",
        )

        if st.button(
            "Submit",
            key=f"submit_{len(st.session_state.question_history)}",
            help="Submit your question",
            use_container_width=True,
        ):

            # Display a spinner while processing
            with st.spinner("Processing your question..."):
                if question and check_environment():
                    try:
                        # Call the query_codebase function from query_engine.py
                        result = query_codebase(question)

                        answer = result["answer"]
                        query = result["query"]
                        context = result["context"]

                        # Add to session state history
                        st.session_state.question_history.append(
                            (question, answer, query, context)
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing your question: {str(e)}")
                        # Log the error for debugging
                        import traceback

                        st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About":
    st.markdown(
        "<h1 class='main-title'>About Code Navigator</h1>", unsafe_allow_html=True
    )

    # Using st.container and columns instead of markdown for better formatting control
    container = st.container()
    with container:
        st.markdown(
            """
        <div style=" padding: 25px; border-radius: 8px; 
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); border-top: 4px solid #3498db;">
            <h3 style="color: #3498db; margin-bottom: 20px; font-size: 22px;">Graph RAG Documentation Navigator</h3>
            <p style="font-size: 16px; line-height: 1.6; margin-bottom: 20px;">
                This application uses a Graph-based Retrieval-Augmented Generation (RAG) system to explore 
                the CodeCarbon codebase.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Features section
        st.markdown(
            """
        <div style="padding: 25px; border-radius: 8px; margin-top: 20px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);">
            <h4 style="color: #3498db; margin-bottom: 15px; font-size: 18px;">Key Features</h4>
            <ul style="list-style-type: circle; margin-left: 20px; line-height: 1.8;">
                <li><span style="font-weight: 600; color: #3498db;">Knowledge Graph:</span> 
                    Uses Neo4j to store code structure and relationships</li>
                <li><span style="font-weight: 600; color: #3498db;">LLM Integration:</span> 
                    Leverages GPT models to analyze code and answer questions</li>
                <li><span style="font-weight: 600; color: #3498db;">Contextual Answers:</span> 
                    Provides relevant code snippets as context</li>
                <li><span style="font-weight: 600; color: #3498db;">Relationship-Aware:</span> 
                    Understands relationships between code components</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # How it works section
        st.markdown(
            """
        <div style=" padding: 25px; border-radius: 8px; margin-top: 20px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);">
            <h4 style="color: #3498db; margin-bottom: 15px; font-size: 18px;">How It Works</h4>
            <ol style="margin-left: 20px; line-height: 1.8;">
                <li>The system extracts entities (Features, Components, Functions, Classes) from code</li>
                <li>It creates relationships between these entities</li>
                <li>When you ask a question, it generates a Cypher query</li>
                <li>It retrieves relevant information from the graph</li>
                <li>It provides a comprehensive answer with code context</li>
            </ol>
        </div>
        """,
            unsafe_allow_html=True,
        )

elif page == "Settings":
    st.markdown("<h1 class='main-title'>Settings</h1>", unsafe_allow_html=True)

    st.markdown(
        """
    <div class='info-container'>
        <h3 style="color: #3498db; margin-bottom: 20px;">Environment Configuration</h3>
        <p>Configure your connection settings for the CodeCarbon Graph RAG system.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Show current file structure
    st.subheader("Project Files")
    try:
        files = [
            f
            for f in os.listdir(current_dir)
            if os.path.isfile(os.path.join(current_dir, f))
        ]
        st.code("\n".join(files))
    except Exception as e:
        st.error(f"Could not list files: {e}")

    # Neo4j settings
    st.subheader("Neo4j Database Settings")
    col1, col2 = st.columns(2)
    with col1:
        neo4j_url = st.text_input(
            "Neo4j URL", value=os.getenv("NEO4J_URL", ""), type="password"
        )
    with col2:
        neo4j_username = st.text_input(
            "Neo4j Username", value=os.getenv("NEO4J_USERNAME", "")
        )

    neo4j_password = st.text_input(
        "Neo4j Password", value=os.getenv("NEO4J_PASSWORD", ""), type="password"
    )

    # OpenAI settings
    st.subheader("OpenAI API Settings")
    openai_key = st.text_input(
        "OpenAI API Key", value=os.getenv("OPEN_AI_API_KEY", ""), type="password"
    )

    # Save button
    if st.button("Save Settings"):
        st.success(
            "Settings saved! (Note: This is a demo - settings aren't permanently saved yet)"
        )

if __name__ == "__main__":
    # This code runs when the script is executed directly
    if not import_success:
        st.error(
            "Application started with errors. Please fix the import issues listed above."
        )
