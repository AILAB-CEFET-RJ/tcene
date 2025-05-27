import streamlit as st
#from processing_utils_2 import codificar_campos

# CLI: streamlit run app.py

# Set page configuration
st.set_page_config(page_title="Consulta - TCE", layout="centered")

# Apply a nicer title and header
st.title("📑 Consulta de Empenho")
st.markdown("Preencha os campos abaixo com as informações do empenho:")

# Create a form for better UX
with st.form("empenho_form"):
    # Layout with columns
    col1, col2 = st.columns(2)
    
    with col1:
        unidade = st.text_input("Unidade", max_chars=40, placeholder="Máx. 40 caracteres")
        credor = st.text_input("Credor", max_chars=40, placeholder="Máx. 40 caracteres")
    
    with col2:
        elem_despesa = st.text_input("ElemDespesaTCE", max_chars=40, placeholder="Máx. 40 caracteres")

    # Histório: larger text area
    historico = st.text_area("Histórico", max_chars=400, placeholder="Máx. 400 caracteres", height=150)


    submitted = st.form_submit_button("🔍 Consultar Empenho")
    
    # Submit button
    if unidade or credor or elem_despesa or historico:
        enviado = True
    else:
        enviado = False
        # Gostaria que a mensagem aparecesse em vermelho somente quando a pessoa tentasse clicar sem nada
        st.error("⚠️ Preencha pelo menos um dos campos para realizar a consulta.")
        
    #X  = codificar_campos(historico)
    
    
    # CHAMAMENTO PUBLICO 01 2020 SFI   PROC  2019021268   SERVCOS FINANCEIROS PARA ARRECADACAO DE GUIAS DE TRIBUTOS E DEMAIS RECEITAS DIVERSAS DE ACORDO COM O PADRAO DA FEDERACAO BRASILEIRA DE BANCOS   FEBRABAN COM PRESTACAO DE CONTAS POR MEIO MAGNETICO D
    
    # BANCO DO BRASIL SA

    # PREFEITURA ANGRA DOS REIS

    # OUTROS SERVICOS DE TERCEIROS   PESSOA JURIDICA



    # Fora do form para evitar múltiplos submit
    if submitted and enviado:
        st.success("✅ Consulta realizada com sucesso!")
        st.write(f"**Itens de Empenho relacionados:**")

        # Exemplo de paginação fictícia
        pagina = st.selectbox("Selecione a página:", ["Página 1", "Página 2", "Página 3", "Página 4"])

        if pagina == "Página 1":
            st.subheader("Item 1")
            with st.container():
                st.write("**Unidade:** PREFEITURA ANGRA DOS REIS")
                st.write("**Credor:** Banco do Brasil SA")
                st.write("**ElemDespesaTCE:** OUTROS SERVIÇOS DE TERCEIROS - PESSOA JURÍDICA")
                st.write("**Histórico:** CHAMAMENTO PUBLICO 01 2020 SFI PROC 2019021268 SERVIÇOS FINANCEIROS PARA ARRECADAÇÃO DE GUIAS DE TRIBUTOS E DEMAIS RECEITAS DIVERSAS DE ACORDO COM O PADRÃO DA FEDERAÇÃO BRASILEIRA DE BANCOS - FEBRABAN COM PRESTAÇÃO DE CONTAS POR MEIO MAGNÉTICO D")
                st.write("**Valor Empenhado:** R$ 100.000,00")
            st.markdown("---")

            st.subheader("Item 2")
            with st.container():
                st.write("**Unidade:** PREFEITURA ANGRA DOS REIS")
                st.write("**Credor:** Caixa Econômica Federal")
                st.write("**ElemDespesaTCE:** SERVIÇOS BANCÁRIOS")
                st.write("**Histórico:** CONTRATO DE PRESTAÇÃO DE SERVIÇOS BANCÁRIOS PARA RECEBIMENTO DE TRIBUTOS MUNICIPAIS")
                st.write("**Valor Empenhado:** R$ 150.000,00")
            st.markdown("---")

            st.subheader("Item 3")
            with st.container():
                st.write("**Unidade:** PREFEITURA ANGRA DOS REIS")
                st.write("**Credor:** Banco Santander")
                st.write("**ElemDespesaTCE:** SERVIÇOS FINANCEIROS")
                st.write("**Histórico:** PRESTAÇÃO DE SERVIÇOS DE ARRECADAÇÃO DE TRIBUTOS VIA SISTEMA BANCÁRIO")
                st.write("**Valor Empenhado:** R$ 80.000,00")
                st.markdown("---")
        
        elif pagina == "Página 2":
            st.write("**Unidade:** SECRETARIA MUNICIPAL DE SAÚDE")
            st.write("**Credor:** EMPRESA DE SERVIÇOS LTDA")
            st.write("**ElemDespesaTCE:** MATERIAL DE CONSUMO")
            st.write("**Histórico:** Aquisição de materiais hospitalares")
            st.write("**Valor Empenhado:** R$ 50.000,00")

        elif pagina == "Página 3":
            st.write("**Unidade:** SECRETARIA DE EDUCAÇÃO")
            st.write("**Credor:** LIVRARIA E PAPELARIA CENTRAL")
            st.write("**ElemDespesaTCE:** MATERIAL DIDÁTICO")
            st.write("**Histórico:** Compra de livros para rede municipal")
            st.write("**Valor Empenhado:** R$ 30.000,00")

        elif pagina == "Página 4":
            st.write("**Unidade:** DEPARTAMENTO DE OBRAS")
            st.write("**Credor:** CONSTRUTORA ABC")
            st.write("**ElemDespesaTCE:** OBRAS E INSTALAÇÕES")
            st.write("**Histórico:** Reforma de praça pública")
            st.write("**Valor Empenhado:** R$ 200.000,00")
            

# Optional: add a footer
st.markdown("---")
st.caption("Sistema de Cadastro de Empenho - Desenvolvido com ❤️ e Streamlit")
