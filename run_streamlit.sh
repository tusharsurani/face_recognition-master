#!/bin/bash
cd "$(dirname "$0")"
eval "$(/opt/homebrew/bin/conda shell.zsh hook)"
streamlit run streamlit_app.py --server.port 8501

