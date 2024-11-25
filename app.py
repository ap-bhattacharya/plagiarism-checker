import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Helper functions
def vectorize(texts):
    return TfidfVectorizer().fit_transform(texts).toarray()

def similarity(vec1, vec2):
    return cosine_similarity([vec1, vec2])[0][1]

def check_plagiarism(file_contents, filenames):
    vectors = vectorize(file_contents)
    results = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim_score = similarity(vectors[i], vectors[j])
            results.append((filenames[i], filenames[j], sim_score * 100))  # Convert to %
    return results

# Streamlit App Configuration
st.set_page_config(
    page_title="Plagiarism Checker",
    page_icon="📄",
    layout="wide"
)

# Custom CSS Styling
st.markdown(
    """
    <style>
        /* Title and Header Styling */
        .title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subheader {
            text-align: center;
            font-size: 1.2rem;
            margin-bottom: 20px;
        }
        /* Footer Styling */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #22223b;
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown('<div class="title">📄 Plagiarism Checker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader">Compare multiple documents for content similarity effortlessly!</div>',
    unsafe_allow_html=True
)

# File Uploader Section
st.markdown("### 📤 Upload Your Files Below:")
uploaded_files = st.file_uploader(
    "Upload your text files here (only .txt supported):", 
    accept_multiple_files=True, 
    type=["txt"]
)

if uploaded_files:
    file_contents = []
    filenames = []

    # Read uploaded files
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8")
            file_contents.append(content)
            filenames.append(file.name)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

    if file_contents:
        # Display uploaded files
        with st.expander("📂 View Uploaded Files"):
            for name in filenames:
                st.markdown(f"✅ **{name}**")

        # Sidebar - Similarity Slider
        st.sidebar.header("🔧 Adjust Similarity Level")
        similarity_level = st.sidebar.slider(
            "Highlight Similarity Above (%)", 
            min_value=0, 
            max_value=100, 
            value=75, 
            step=5
        )
        st.sidebar.markdown(
            "Set a similarity percentage threshold to highlight potential plagiarism."
        )

        # Perform plagiarism check
        with st.spinner("Analyzing documents for plagiarism..."):
            results = check_plagiarism(file_contents, filenames)

        # Display results
        st.markdown("### 🔍 Plagiarism Results:")
        if results:
            # Convert results to DataFrame
            df = pd.DataFrame(
                results, 
                columns=["File 1", "File 2", "Similarity (%)"]
            )

            # Add a "Similarity Level" column
            df["Similarity Level"] = df["Similarity (%)"].apply(
                lambda x: "High Similarity ⚠️" if x >= similarity_level else "Low Similarity ✔️"
            )

            # Style the table
            def highlight_similarity(row):
                color = "lightcoral" if row["Similarity Level"] == "High Similarity ⚠️" else "lightgreen"
                return [f"background-color: {color}; color: black; font-weight: bold;" for _ in row]

            st.table(
                df.style.apply(highlight_similarity, axis=1).format(
                    {"Similarity (%)": "{:.1f}%"}
                )
            )
            st.success("✅ Analysis Complete!")
        else:
            st.warning("No significant similarity detected.")
    else:
        st.error("⚠️ No readable content detected in the uploaded files.")
else:
    st.info("💡 Upload `.txt` files to begin the plagiarism check!")

# Footer Section
st.markdown(
    '<div class="footer">Made with ❤️ using Streamlit by Jagriti</div>',
    unsafe_allow_html=True
)
