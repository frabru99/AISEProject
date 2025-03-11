import pandas as pd 
import os
import re
from lightrag_ollama_demo import rag
from ollama import Client

specific_query =["Does the clause describe how and why a service provider collects user information?", "Does the clause describe how user information may be shared with or collected by third parties?", "Does the clause describe the choices and control options available to users?", "Does the clause describe if and how users may access, edit, or delete their information?", "Does the clause describe how long user information is stored?", "Does the clause describe how user information is protected?", "Does the clause describe practices that pertain only to a specific group of users (e.g., children, Europeans, or California residents)?"]
i=0

#solo per il few-shot
for file in os.listdir("./Input_Few_Shot"):
    files_few_shot_= pd.read_csv(os.path.join("./Input_Few_Shot",file), sep=';', header=0)
    for index, row in files_few_shot_.iterrows():
        rag.query(f" 'Question: {row.to_list()[2]}'. Answer: {row.to_list()[1]}")

    files_test = pd.read_csv(os.path.join("./File_Testing",file), sep=';', header=0)
    with open("./output_few_shot" + "/" + file, "w", encoding="utf-8") as f:
        f.write("index;answer;answer (model); re-answer (only if previous answer was 'No')\n")
    with open("./output_few_shot" + "/" + file, "a", encoding="utf-8") as f:
        for index, row in files_test.iterrows():
            print(row.to_list()[2])
            res=rag.query(f"{specific_query[i]} '{row.to_list()[2]}'. Just answer 'yes' or 'no'")
            client = Client(
                host='http://localhost:11434/',
                headers={'Content-Type': 'application/json'}
            )
            response = client.chat(model='deepseek-r1:8b', messages=[
                {
                    'role': 'system',
                    'content': "You are an Evaluator, an agent specialized in assessing the correctness and completeness of the answers provided by a Reasoner. Your task is to: evaluate whether the concept explanations are clear and accurate, check if the problem decomposition into sub-questions is appropriate and whether the answers to those questions are correct, verify if the final reasoning is logical, complete, and leads to the correct answer, perform a counterfactual evaluation, examining alternative scenarios to see if the reasoning and answer remain robust, your assessment must be precise, well-justified, and based on a rigorous analysis.",
                },
                {
                    'role': 'user',
                    'content': f"Adesso tu devi valutare la risposta di un altro LLM: dato che il task era {specific_query[i]} '{row.to_list()[2]}' e dato che la risposta generata è stata '{res}', la risposta fornita è corretta? Rispondi solo con 'corretto' o 'sbagliato', esprimendoti esclusivamente in italiano.",
                },
            ])
             # Testo con la parte di thinking
            deepseek_response = response['message']['content']
            # Rimuove tutto ciò che è tra <think> e </think>
            clean_response = re.sub(r"<think>.*?</think>", "", deepseek_response, flags=re.DOTALL).strip()

            # Definizione delle regex per "corretto" e "sbagliato"
            yes_pattern = re.compile(r"\b(corretto|certamente|ovviamente|assolutamente s[iì]|senz'altro)\b", re.IGNORECASE)
            no_pattern = re.compile(r"\b(sbagliato|negativo|assolutamente no|non credo|non penso)\b", re.IGNORECASE)

            
            if no_pattern.search(clean_response):
                new_res=rag.query(f"La seguente risposta {res} è sbagliata. Rispondi adeguatamente a questa domanda: {specific_query[i]} '{row.to_list()[2]}'. Just answer 'yes' or 'no'")
                f.write(f"{row.to_list()[0]};{row.to_list()[1]};{res};{new_res}\n")
            else:
                f.write(f"{row.to_list()[0]};{row.to_list()[1]};{res};'null'\n")
    i=i+1