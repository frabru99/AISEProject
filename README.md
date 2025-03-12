# LightPrivacy
<p align="center">
  <img src="https://github.com/user-attachments/assets/2ac2887a-f132-41ea-8790-45c95a5c301a" style="width: 500px"/>
</p>


Every day, millions of users accept terms of service and privacy policies without fully reading or understanding their content. Due to their complexity, length and verbosity these documents often obscure crucial details about data processing practices, exposing users to potential privacy risks.

The General Data Protection Regulation (GDPR) establishes strict requirements for transparency, consent, and data processing, aiming to empower individuals with control over their personal data. However, the legal language used in privacy policies and regulatory texts is often challenging for non-experts, making it difficult to determine whether a policy truly complies with GDPR standards.

LightPrivacy is an AI-driven system designed to automatically assess GDPR compliance in privacy policies. It provides:
- Clear and concise **summaries** of privacy policies.
- Automated risk detection to highlight **potential compliance issues.**
- **Explainable assessments,** ensuring transparency through causal reasoning.

The system is built upon **LightRAG**, a retrieval-augmented generation framework structured as a graph, leveraging 2 specialized large language models (LLMs):
- **Reasoning & Response Generation Model** – Analyzes privacy policies and generates compliance assessments.
- **Evaluation Model** – Ensures response consistency, faithfulness, and causality, enhancing trust in the AI’s decisions.

By combining causal reasoning with retrieval-augmented generation, LightPrivacy bridges the gap between legal complexity and practical usability, providing an accessible, interpretable, and reliable tool for GDPR compliance verification.

# How to run
- **Note**: in order to run the demo properly, you must have the 3.12 version for Python.
- **Note**: before starting with this brief guide, you have to install Ollama on your pc and pull the same models described in the repo.

<br>

- Download the .zip file containing the source file from the "Release" section
- Go to "AISE Project" folder:

```
cd LIGHTRAG PULITO/LIGHTRAG/AISE Project/
```
- Install requirements before starting
- Run the script:
```bash
python lightrag_ollama_demo.py
```
# Credits 
- https://github.com/HKUDS/LightRAG
