from lightrag_api_openai_compatible_demo_simplified import rag
import subprocess


GPT_URL="https://api.openai.com/v1"


changed_files=subprocess.getoutput("git diff --name-only -- './input/*.txt'")


for file in changed_files:
    print("NOME FILE: "+ filename)
    with open(file, "r") as f:
        print("Insert del file nel RAG...")
        rag.insert(f.read())