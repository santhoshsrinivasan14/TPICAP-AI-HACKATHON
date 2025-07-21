# TPICAP AI Hackathon: Personalized RAG Course Recommender

A command-line Retrieval-Augmented Generation (RAG) system that recommends courses based on user questions and remembers each user's feedback to improve recommendations over time.

---

## Features

- Retrieval-Augmented Generation: Finds relevant courses from a CSV using semantic search.
- Personalized Recommendations: Remembers user likes/feedback and boosts their favorite courses in future recommendations.
- Memory System: Stores and adapts to user feedback across sessions.
- Runs Locally: All data and logic run on your machine using local LLMs (Ollama).

---

## How to Run

1. **Clone the repo**
    ```sh
    git clone https://github.com/santhoshsrinivasan14/TPICAP-AI-HACKATHON.git
    cd TPICAP-AI-HACKATHON
    ```

2. **Create virtual environment & install requirements**
    ```sh
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Mac/Linux:
    source venv/bin/activate

    pip install -r requirements.txt
    ```

3. **Pull and start Ollama model**
    - Make sure [Ollama](https://ollama.com/download) is installed and running.
    - Pull the required model (e.g., phi3 or llama3):

    ```sh
    ollama pull phi3
    ```

4. **Run the CLI**
    ```sh
    python main.py
    ```

---

## Example Demo

```sh
🧠 AI Course Recommender with Memory (type 'q' to quit)

Enter your user ID: santhosh

👋 Welcome! Let's find you some courses.

-------------------------------
Ask your question: I want to learn AI

📚 Recommendations:

1. **Introduction to Artificial Intelligence and Machine Learning - University of XYZ (Online)**
   Reason: Provides fundamental knowledge in AI and ML with hands-on Python coding.

2. **Deep Neural Networks for Natural Language Processing - Coursera (Online)**
   Reason: Learn about neural networks for NLP tasks.

3. **Machine Learning with Python – Part 1 | FutureLearn (Online)**
   Reason: Covers essential ML algorithms using Python.

Which course above do you like? (paste course title, or leave blank to skip): Machine Learning with Python – Part 1 | FutureLearn (Online)

-------------------------------
Ask your question: I want hands-on Python projects

📚 Recommendations:

1. **Machine Learning with Python – Part 1 | FutureLearn (Online)** (You liked this before!)
2. **Python for Everybody Specialization – University of Michigan (Coursera)**
3. **Data Engineering with Python – Udemy**

Which course above do you like? (paste course title, or leave blank to skip): Python for Everybody Specialization – University of Michigan (Coursera)

-------------------------------
Ask your question: I want to learn cloud computing

📚 Recommendations:

1. **Cloud Computing Specialization on Coursera**
2. **Data Engineering: Machine Learning at Scale from Johns Hopkins University SAILORS (Online)**
3. **Introduction to Cloud Computing from edX**

Which course above do you like? (paste course title, or leave blank to skip):


User feedback is saved in user_feedback.csv (generated at runtime).

The system improves as users interact and like courses—each user gets personalized, adaptive recommendations.

No API keys required; everything runs locally!


