import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Função para calcular RSI manualmente
def calculate_rsi_manual(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Funções de análise
def gordon_model(dividend, growth_rate, discount_rate):
    if discount_rate <= growth_rate:
        return None  # Evita divisão por zero ou valores negativos
    return dividend / (discount_rate - growth_rate)

def dcf_model(cash_flows, discount_rate, terminal_growth_rate):
    if not cash_flows:
        return None  # Retorna None se não houver fluxos de caixa
    present_value = 0
    for i, cf in enumerate(cash_flows):
        present_value += cf / ((1 + discount_rate) ** (i + 1))
    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    present_value += terminal_value / ((1 + discount_rate) ** len(cash_flows))
    return present_value

def predict_future_prices(data, days=30):
    model = LinearRegression()
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data['Close'].values
    model.fit(X, y)
    future_X = np.array(range(len(data), len(data) + days)).reshape(-1, 1)
    future_prices = model.predict(future_X)
    return future_prices

# Função para gerar o PDF
def create_pdf_report(filename, ticker, fair_price_gordon, fair_price_dcf, current_price, future_prices, strength, recommendation):
    # Cria o documento PDF
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Título do relatório
    title = Paragraph("Relatório de Análise de Ações", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Introdução
    intro = Paragraph(f"Este relatório contém a análise detalhada do ativo {ticker}.", styles['BodyText'])
    story.append(intro)
    story.append(Spacer(1, 12))

    # Valor Justo
    fair_price_gordon_text = f"Preço justo estimado (Modelo de Gordon): {fair_price_gordon:.2f}" if fair_price_gordon else "Preço justo estimado (Modelo de Gordon): N/A"
    fair_price_dcf_text = f"Preço justo estimado (DCF): {fair_price_dcf:.2f}" if fair_price_dcf else "Preço justo estimado (DCF): N/A"
    story.append(Paragraph(fair_price_gordon_text, styles['BodyText']))
    story.append(Paragraph(fair_price_dcf_text, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Valor Atual
    current_price_text = f"Valor atual: {current_price:.2f}"
    story.append(Paragraph(current_price_text, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Previsão de Lucro
    if future_prices is not None:
        profit_forecast = ((future_prices[-1] - current_price) / current_price) * 100
        profit_forecast_text = f"Previsão de lucro: {profit_forecast:.2f}%"
    else:
        profit_forecast_text = "Previsão de lucro: N/A"
    story.append(Paragraph(profit_forecast_text, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Força da Oportunidade e Recomendação
    story.append(Paragraph(f"Força da Oportunidade: {strength}", styles['BodyText']))
    story.append(Paragraph(f"Recomendação: {recommendation}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Gráfico de Preços Futuros (opcional)
    if future_prices is not None:
        try:
            plt.figure(figsize=(6, 4))
            plt.plot(future_prices, label="Preços Futuros")
            plt.title("Previsão de Preços Futuros")
            plt.xlabel("Dias")
            plt.ylabel("Preço")
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image(buf, width=6*inch, height=4*inch)
            story.append(img)
            plt.close()
        except Exception as e:
            story.append(Paragraph("Gráfico de preços futuros indisponível.", styles['BodyText']))
    else:
        story.append(Paragraph("Gráfico de preços futuros indisponível.", styles['BodyText']))

    # Conclusão
    conclusion = Paragraph("Este relatório foi gerado automaticamente pelo Analisador de Ações.", styles['BodyText'])
    story.append(conclusion)

    # Gera o PDF
    doc.build(story)

# Função para gerar recomendação
def generate_recommendation(fair_price, current_price, future_prices):
    if fair_price is None or current_price is None or future_prices is None:
        return "Dados insuficientes", "Não é possível gerar uma recomendação."

    margin = fair_price - current_price
    future_growth = (future_prices[-1] - current_price) / current_price * 100

    if margin > 0 and future_growth > 0:
        strength = "Forte Oportunidade"
        recommendation = "Considerar investimento. O ativo está subvalorizado e tem potencial de crescimento."
    elif margin > 0 and future_growth <= 0:
        strength = "Oportunidade Moderada"
        recommendation = "Considerar investimento. O ativo está subvalorizado, mas o crescimento futuro é incerto."
    elif margin <= 0 and future_growth > 0:
        strength = "Oportunidade Moderada"
        recommendation = "Cautela. O ativo pode estar sobrevalorizado, mas há potencial de crescimento."
    else:
        strength = "Fraca Oportunidade"
        recommendation = "Evitar investimento. O ativo está sobrevalorizado e com baixo potencial de crescimento."

    return strength, recommendation

# Interface do Streamlit
st.title("📈 Analisador Avançado de Ações")

# Seleção de múltiplos ativos
tickers = st.text_input("Digite os tickers das ações separados por vírgula (ex: AAPL, MSFT, GOOGL):")
ticker_list = [ticker.strip().upper() for ticker in tickers.split(",")] if tickers else []

if ticker_list:
    # Abas para organização
    tab1, tab2, tab3, tab4 = st.tabs(["Análise Individual", "Comparação de Ativos", "Análise Técnica", "Relatório"])

    with tab1:
        st.subheader("Análise Individual")
        selected_ticker = st.selectbox("Selecione um ativo para análise detalhada:", ticker_list)
        stock = yf.Ticker(selected_ticker)
        data = stock.history(period="1y")

        # Dados históricos
        st.write(f"Dados históricos de {selected_ticker}:")
        st.line_chart(data['Close'])

        # Métodos de avaliação
        st.subheader("Avaliação de Preço Justo")
        dividend = stock.dividends.mean() if not stock.dividends.empty else 0
        growth_rate = st.number_input("Taxa de crescimento esperada (%):", value=5.0) / 100
        discount_rate = st.number_input("Taxa de desconto (%):", value=10.0) / 100

        if dividend > 0:
            fair_price_gordon = gordon_model(dividend, growth_rate, discount_rate)
            st.write(f"Preço justo estimado (Modelo de Gordon): {fair_price_gordon:.2f}")

        st.subheader("Fluxo de Caixa Descontado (DCF)")
        try:
            cash_flows = [stock.cashflow.iloc[-1][item] for item in stock.cashflow.columns]
            terminal_growth_rate = st.number_input("Taxa de crescimento terminal (%):", value=3.0) / 100
            fair_price_dcf = dcf_model(cash_flows, discount_rate, terminal_growth_rate)
            if fair_price_dcf is not None:
                st.write(f"Preço justo estimado (DCF): {fair_price_dcf:.2f}")
            else:
                st.write("Dados de fluxo de caixa indisponíveis para cálculo do DCF.")
        except Exception as e:
            st.write(f"Erro ao calcular DCF: {e}")

        # Previsão de preços futuros
        st.subheader("Previsão de Preços Futuros")
        try:
            future_prices = predict_future_prices(data)
            st.write(f"Preços previstos para os próximos 30 dias: {future_prices}")
        except Exception as e:
            st.write(f"Erro ao prever preços futuros: {e}")
            future_prices = None

        # Recomendação
        st.subheader("Recomendação")
        current_price = data['Close'].iloc[-1]
        fair_price = fair_price_gordon if fair_price_gordon else fair_price_dcf

        # Formatação condicional do valor justo
        fair_price_display = f"{fair_price:.2f}" if fair_price is not None else "N/A"
        st.write(f"**Valor Justo:** {fair_price_display}")
        st.write(f"**Valor Atual:** {current_price:.2f}")

        # Formatação condicional da previsão de lucro
        if future_prices is not None:
            profit_forecast = ((future_prices[-1] - current_price) / current_price) * 100
            st.write(f"**Previsão de Lucro:** {profit_forecast:.2f}%")
        else:
            st.write("**Previsão de Lucro:** N/A")

        strength, recommendation = generate_recommendation(fair_price, current_price, future_prices)
        st.write(f"**Força da Oportunidade:** {strength}")
        st.write(f"**Recomendação:** {recommendation}")

    with tab2:
        st.subheader("Comparação de Ativos")
        comparison_data = pd.DataFrame()
        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            comparison_data[ticker] = data['Close']
        st.line_chart(comparison_data)

    with tab3:
        st.subheader("Análise Técnica")
        selected_ticker = st.selectbox("Selecione um ativo para análise técnica:", ticker_list)
        stock = yf.Ticker(selected_ticker)
        data = stock.history(period="1y")
        data['RSI'] = calculate_rsi_manual(data)
        st.write(f"Índice de Força Relativa (RSI) de {selected_ticker}:")
        st.line_chart(data['RSI'])

    with tab4:
        st.subheader("Gerar Relatório")
        if st.button("Gerar Relatório em PDF"):
            report_content = {
                "ticker": selected_ticker,
                "fair_price_gordon": fair_price_gordon,
                "fair_price_dcf": fair_price_dcf,
                "current_price": current_price,
                "future_prices": future_prices,
                "strength": strength,
                "recommendation": recommendation
            }
            create_pdf_report("relatorio.pdf", **report_content)
            st.success("Relatório gerado com sucesso! Verifique o arquivo 'relatorio.pdf'.")

# Alertas personalizados (simulação no terminal)
st.sidebar.subheader("Configuração de Alertas")
alert_ticker = st.sidebar.selectbox("Selecione um ativo para alerta:", ticker_list)
alert_price = st.sidebar.number_input("Defina o preço de alerta:", value=100.0)
if st.sidebar.button("Ativar Alerta"):
    stock = yf.Ticker(alert_ticker)
    current_price = stock.history(period="1d")['Close'].iloc[-1]
    if current_price >= alert_price:
        st.sidebar.success(f"Alerta! {alert_ticker} atingiu {current_price:.2f}.")
    else:
        st.sidebar.info(f"Aguardando {alert_ticker} atingir {alert_price:.2f}.")