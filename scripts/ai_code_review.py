import os
import requests
from google import genai

def get_pr_files():
    """Obtém a lista de arquivos alterados no Pull Request via API do GitHub."""
    repo = os.getenv('GITHUB_REPOSITORY')
    pr_number = os.getenv('PR_NUMBER')
    token = os.getenv('GITHUB_TOKEN')
    
    if not pr_number:
        print("Erro: PR_NUMBER não configurado.")
        return []

    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        files = response.json()
        return files if isinstance(files, list) else []
    except Exception as e:
        print(f"Erro ao buscar arquivos do GitHub: {e}")
        return []

def run_ai_review():
    client = genai.Client(api_key=os.getenv('AIzaSyBI13P_FUcC9kFa4mf0IAJShr5aMf0MWeg'))
    files = get_pr_files()
    review_results = []

    for file in files:
        if not isinstance(file, dict): continue
        
        filename = file.get('filename', '')
        status = file.get('status', 'alterado')
        # O 'patch' é onde o GitHub guarda o diff das linhas alteradas
        diff_patch = file.get('patch', '') 

        if not (filename.endswith('.robot') or filename.endswith('.py')) or not diff_patch:
            continue

        # Tentamos ler o arquivo completo para dar contexto, mas focamos no DIFF
        full_content = ""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                full_content = f.read()

        prompt = f"""
        Você é um Engenheiro de QA Sênior. Analise as alterações (DIFF) abaixo.
        Arquivo: {filename}
        Status: {status}

        TAREFA:
        Compare o que foi alterado com o contexto do arquivo completo. 
        Verifique se as novas linhas (+) introduzem inconsistências, quebram keywords existentes 
        ou se as linhas removidas (-) eram essenciais.
        Ignore as variaveis que não são declaradas ou inicializadas no arquivo, mas foque em inconsistências de sintaxe, uso de keywords e estrutura.
        Em cenários que houve a mesma alteraçao para diversas partes do código compacte essas alterações em um único item, evitando repetições desnecessárias. 
        DIFF DAS ALTERAÇÕES:
        ---
        {diff_patch}
        ---

        CONTEXTO COMPLETO DO ARQUIVO:
        ---
        {full_content}
        ---
        
        Responda em Português (Brasil). Seja direto e aponte a linha exata se houver erro.
        """

        try:
            # Chamada correta para o novo SDK google-genai
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            if response.text:
                review_results.append(f"### 📄 Arquivo: `{filename}`\n{response.text}")
        except Exception as e:
            print(f"Erro na análise da IA para {filename}: {e}")

    # Escrevendo o relatório final
    with open("review_report.md", "w", encoding='utf-8') as f:
        f.write("## 🤖 AI Code Review\n\n")
        if review_results:
            f.write("\n\n".join(review_results))
        else:
            f.write("Nenhuma inconsistência relevante encontrada nos arquivos analisados.")

if __name__ == "__main__":
    run_ai_review()