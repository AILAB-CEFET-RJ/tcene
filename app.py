import streamlit as st
from tools.chroma_utils import load_vector_store, load_model_tokenizer, create_embeddings, similarity_search

import yaml
import pandas as pd

# CLI: streamlit run app.py

## ----------------------------------------------------------------------------------------##
print('loading model..')
model, tokenizer = load_model_tokenizer()

print('loading vector_store..')
vector_store, client, collection = load_vector_store(model)
## ---------------------------------------------------------------------------------------- ##


# Set page configuration
st.set_page_config(page_title="Consulta - TCE", layout="centered")

st.title("📑 Consulta de Empenho")
st.markdown("Preencha os campos abaixo com as informações do empenho:")



# Step 1: Initialize form values in session_state (only once)
for key in ["unidade", "credor", "elem_despesa", "historico"]:
    if key not in st.session_state:
        st.session_state[key] = ""

if 'pagina_atual' not in st.session_state:
    st.session_state.pagina_atual = 0
    
if "consulta_realizada" not in st.session_state:
    st.session_state.consulta_realizada = False
    
    

# Create a form for better UX
with st.form("empenho_form"):
    # Layout with columns
    col1, col2 = st.columns(2)
    
    with col1:
        unidade = st.text_input("Unidade", max_chars=40, placeholder="Máx. 40 caracteres", value=st.session_state.unidade)
        credor = st.text_input("Credor", max_chars=40, placeholder="Máx. 40 caracteres", value=st.session_state.credor)
    
    with col2:
        elem_despesa = st.text_input("ElemDespesaTCE", max_chars=40, placeholder="Máx. 40 caracteres", value=st.session_state.elem_despesa)

    historico = st.text_area("Histórico", max_chars=400, placeholder="Máx. 400 caracteres", height=150, value=st.session_state.historico) # Histório: larger text area
    
    

    
    col_submit, col_clear = st.columns([1, 1])
    with col_submit:
        submitted = st.form_submit_button("🔍 Consultar Empenho")
    with col_clear:
        clear = st.form_submit_button("🧹 Limpar Campos")

    # If clear is pressed, reset all fields
    if clear:
        st.session_state.unidade = ""
        st.session_state.credor = ""
        st.session_state.elem_despesa = ""
        st.session_state.historico = ""
        st.session_state.consulta_realizada = False
        st.session_state.pagina_atual = 0
        st.rerun()
    
    # Submit button
    if unidade or credor or elem_despesa or historico:
        st.session_state.consulta_realizada = True
        
        
    # else:
    #     # Gostaria que a mensagem aparecesse em vermelho somente quando a pessoa tentasse clicar sem nada
    #     st.error("⚠️ Preencha pelo menos um dos campos para realizar a consulta.")
        
    ## ---------------------------------------------------------------------------------------- ##

    # Monta a query concatenando os campos preenchidos
    # query_lista = [str(x) for x in [historico, unidade, elem_despesa, credor] if x]
    # query = " ".join(query_lista)
    if historico != "":
        embed_query = create_embeddings(pd.Series(historico), model, tokenizer)[0]
    else:
        embed_query = None
    ## ---------------------------------------------------------------------------------------- ##


# Fora do form para evitar múltiplos submit
if st.session_state.consulta_realizada:
    
    st.session_state.unidade = unidade
    st.session_state.credor = credor
    st.session_state.elem_despesa = elem_despesa
    st.session_state.historico = historico

    
    try:

        documents = similarity_search(collection, embed_query, unidade, credor, elem_despesa, threshold=10)

        total_results = len(documents)
        ## ---------------------------------------------------------------------------------------- ##
        
        
        st.success("✅ Consulta realizada com sucesso!")
        st.write(f"**Itens de Empenho relacionados:**")
        
        
        # Quantidade total de documentos
        total_empenhos = len(documents)
        print(f"Total de documentos encontrados: {total_empenhos}")
        itens_por_pagina = 10
        total_paginas = (total_empenhos + itens_por_pagina - 1) // itens_por_pagina


        # Botões de navegação
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Página anterior") and st.session_state.pagina_atual > 0:
                st.session_state.pagina_atual -= 1
        with col2:
            if st.button("Próxima página ➡️") and st.session_state.pagina_atual < total_paginas - 1:
                st.session_state.pagina_atual += 1
                

        # Exibe página atual
        st.write(f"📄 Página {st.session_state.pagina_atual + 1} de {total_paginas}")
        
        
        # Seleciona os itens dessa página
        inicio = st.session_state.pagina_atual * itens_por_pagina
        fim = min(inicio + itens_por_pagina, total_empenhos)
        
        # Exibe apenas os documentos da página atual
        for count_items, doc in enumerate(documents[0:3], start=0):
            string = doc['document']
            metadata = doc['metadata']
            vlr_empenhado = metadata['Vlr_Empenhado']
            cluster = metadata['Clusters']
            unidade = metadata['Unidade']
            elemDespesa = metadata['ElemDespesaTCE']
            credor = metadata['Credor']
    

            st.subheader(f"Item {count_items + 1}")
            with st.container():
                st.write(f"**Unidade:** {unidade}")
                st.write(f"**Credor:** {credor}")
                st.write(f"**ElemDespesaTCE:** {elemDespesa}")
                st.write(f"**Histórico:** {string}")
                st.write(f"**Valor Empenhado:** {vlr_empenhado}")
                st.write(f"**Cluster:** {cluster}")
            st.markdown("---")
        
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao consultar os empenhos. Coloque mais informação na consulta. {e}")
        documents = []
    ## ---------------------------------------------------------------------------------------- ##




# Footer
st.markdown("---")
st.caption("Sistema de Cadastro de Empenho - Desenvolvido com ❤️ e Streamlit")
