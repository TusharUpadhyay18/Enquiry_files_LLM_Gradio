import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


def load_file(file):
    file_path = file.name
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(file_path)
        elif ext == ".json":
            return pd.read_json(file_path)
        elif ext in [".txt", ".log"]:
            # Try reading as table with tab/space separation
            return pd.read_table(file_path, delimiter=None, engine='python')
        else:
            raise ValueError("Unsupported file type: " + ext)
    except Exception as e:
        raise ValueError(f"Failed to load file: {str(e)}")


def ask_about_file(file, question):
    try:
        df = pd.read_csv(file.name)

        csv_text = df.to_csv(index=False)

        messages = [
            {"role": "system",
                "content": "You are a helpful data analyst working with CSV data."},
            {"role": "user", "content": f"Here is a CSV preview:\n\n{csv_text}\n\nQuestion: {question}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"


demo = gr.Interface(
    fn=ask_about_file,
    inputs=[
        gr.File(label="Upload CSV / Excel / JSON / TXT file",
                file_types=[".csv", ".xls", ".xlsx", ".json", ".txt", ".log"]),
        gr.Textbox(label="Ask a question about the data")
    ],
    outputs="text",
    title="📁 Multi-File Data Q&A with GPT",
    description="Upload a CSV, Excel, JSON, or TXT file and ask questions in natural language.",
    flagging_mode="never"
)

# Launch app
if __name__ == "__main__":
    demo.launch(share=True)
