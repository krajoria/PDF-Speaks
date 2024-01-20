# PDF SPEAKS - powered by AI

It is a Streamlit-based application that integrates with Gemini Pro to provide a conversational interface for extracting information from PDF documents.

## Features

- **PDF Chatting:** Engage in natural language conversations to extract information from PDFs.
- **Gemini Pro Integration:** Leverages Gemini Pro for accurate and context-aware responses.
- **Streamlit Interface:** User-friendly web interface for easy interaction.

## Requirements

- Python 3.7+
- [Gemini Pro API Key](https://gemini-pro.com/) (Sign up on Gemini Pro and obtain API key)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/krajoria/PDF-Speaks.git
    cd PDF-Speaks
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up Gemini Pro API Key:

    Obtain your Gemini Pro API Key and set it as an environment variable:

    ```bash
    export GEMINI_PRO_API_KEY="your-api-key"
    ```

4. Run the application:

    ```bash
    streamlit run app.py
    ```

    Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to use the application.
