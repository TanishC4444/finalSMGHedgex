# import yfinance as yf
# import pandas as pd

# # Function to get full stock data
# def get_full_stock_data(ticker, period="1mo", interval="1d"):
#     stock = yf.Ticker(ticker)
#     data = stock.history(period=period, interval=interval)
#     fundamentals = stock.info
    
#     # Convert fundamentals to DataFrame and remove unwanted columns
#     fundamentals_df = pd.DataFrame([fundamentals])
#     columns_to_remove = [
#         "address1", "city", "state", "zip", "country", "phone", "website", "industry", "industryKey", "industryDisp", 
#         "sector", "sectorKey", "sectorDisp", "longBusinessSummary", "fullTimeEmployees", "companyOfficers", "auditRisk", 
#         "boardRisk", "compensationRisk", "shareHolderRightsRisk", "overallRisk", "governanceEpochDate", 
#         "compensationAsOfEpochDate", "irWebsite", "maxAge", "priceHint", "previousClose", "open", "dayLow", "dayHigh", 
#         "regularMarketPreviousClose", "regularMarketOpen", "regularMarketDayLow", "regularMarketDayHigh", "dividendRate", 
#         "dividendYield", "exDividendDate", "payoutRatio", "fiveYearAvgDividendYield"
#     ]
#     fundamentals_df = fundamentals_df.drop(columns=[col for col in columns_to_remove if col in fundamentals_df], errors='ignore')
    
#     data = data.reset_index()
#     full_data = data.merge(fundamentals_df, how="cross")
    
#     return full_data

# # Function to evaluate the equation
# def evaluate_equation(data, equation):
#     column_replacements = {col: f"row['{col}']" for col in data.columns}

#     # Replace column names in the equation with row-based references
#     for col_name, col_ref in column_replacements.items():
#         equation = equation.replace(col_name, col_ref)

#     points = []
#     for _, row in data.iterrows():
#         try:
#             result = eval(equation)
#             points.append(result)
#         except Exception as e:
#             points.append(None)
#             print(f"Error evaluating equation: {e}")

#     return points

# # Function to calculate movement direction
# def calculate_movement(data):
#     movements = []
#     for i in range(len(data) - 1):  # Exclude the last row
#         next_open = data.loc[i + 1, "Open"]
#         next_close = data.loc[i + 1, "Close"]
#         movement = "+" if (next_close - next_open) > 0 else "-"
#         movements.append(movement)
    
#     movements.append(None)  # Last row has no movement data
#     return movements

# # Function to calculate accuracy
# def calculate_accuracy(data):
#     correct_predictions = 0
#     total_predictions = len(data) - 1  # Exclude last row as it has no movement data
    
#     for i in range(total_predictions):
#         if data.loc[i, 'points'] is not None:
#             predicted_sign = "+" if data.loc[i, 'points'] > 0 else "-"
#             if predicted_sign == data.loc[i, 'movement']:
#                 correct_predictions += 1
    
#     accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
#     return accuracy

# # Main function
# def main():
#     print("Allowed time periods:")
#     print(" - '1d' (1 day)")
#     print(" - '5d' (5 days)")
#     print(" - '1wk' (1 week)")
#     print(" - '1mo' (1 month)")
#     print(" - '3mo' (3 months)")
#     print(" - '6mo' (6 months)")
#     print(" - '1y' (1 year)")
#     print(" - '2y' (2 years)")
#     print(" - '5y' (5 years)")
#     print(" - '10y' (10 years)")
#     print(" - 'ytd' (Year-to-date)")
#     print(" - 'max' (Maximum available data)")
    
#     print("Allowed intervals:")
#     print(" - '1m' (1 minute, only for last 7 days of data)")
#     print(" - '2m' (2 minutes, only for last 60 days of data)")
#     print(" - '5m' (5 minutes, only for last 60 days of data)")
#     print(" - '15m' (15 minutes, only for last 60 days of data)")
#     print(" - '30m' (30 minutes, only for last 60 days of data)")
#     print(" - '1h' (1 hour, only for last 730 days of data)")
#     print(" - '1d' (1 day)")
#     print(" - '5d' (5 days)")
#     print(" - '1wk' (1 week)")
#     print(" - '1mo' (1 month)")
#     print(" - '3mo' (3 months)")

#     ticker = input("Enter stock ticker: ")
#     period = input("Enter time period from the list above: ")
#     interval = input("Enter interval from the list above: ")
    
#     data = get_full_stock_data(ticker, period, interval)
#     columns = sorted(data.columns.tolist(), key=len)

#     print("Available columns for equations:\n", "\n".join(columns))

#     # Ask for the equation to evaluate
#     equation = input("Enter equation to evaluate (use column names, e.g., 'Close * 2 - beta / Open * Volume'): ")
    
#     # Evaluate the equation and calculate points
#     data['points'] = evaluate_equation(data, equation)
    
#     # Calculate movement direction
#     data['movement'] = calculate_movement(data)
    
#     # Calculate accuracy
#     accuracy = calculate_accuracy(data)
    
#     # Save Date, Points, and Movement columns
#     with open("analysis.txt", "w") as f:
#         f.write(data[['Date', 'points', 'movement']].to_string())
#         f.write(f"\n\nPrediction Accuracy: {accuracy:.2f}%")
    
#     print(f"Data and equation results saved to analysis.txt with accuracy: {accuracy:.2f}%")

# if __name__ == "__main__":
#     main()
# import yfinance as yf

# # Choose a stock ticker (e.g., Apple)
# ticker = "AAPL"

# # Download stock data
# stock = yf.Ticker(ticker)

# # Get basic info about the stock
# info = stock.info

# # Display specific info (labeled)
# print(f"Stock Ticker: {ticker}")
# print(f"Name: {info.get('longName', 'N/A')}")
# print(f"Sector: {info.get('sector', 'N/A')}")
# print(f"Industry: {info.get('industry', 'N/A')}")
# print(f"Market Cap: {info.get('marketCap', 'N/A')}")
# print(f"Trailing P/E: {info.get('trailingPE', 'N/A')}")
# print(f"Price-to-Book (P/B) Ratio: {info.get('priceToBook', 'N/A')}")
# print(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
# print(f"Current Price: {info.get('currentPrice', 'N/A')}")
# print(f"Previous Close: {info.get('previousClose', 'N/A')}")
# print(f"Beta (Risk Measure): {info.get('beta', 'N/A')}")
# print(f"52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}")
# print(f"52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}")
# print(f"Dividend Rate: {info.get('dividendRate', 'N/A')}")
# print(f"PE Ratio (TTM): {info.get('peRatio', 'N/A')}")
# print(f"Forward P/E: {info.get('forwardPE', 'N/A')}")
# print(f"Trailing PEG Ratio: {info.get('trailingPEG1Y', 'N/A')}")

import yfinance as yf
import pandas as pd

def get_stock_metrics(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)

    # === Financials for growth calculations ===
    financials = ticker.financials.T.sort_index()

    try:
        rev_start = financials["Total Revenue"].iloc[0]
        rev_end = financials["Total Revenue"].iloc[-1]
        earn_start = financials["Net Income"].iloc[0]
        earn_end = financials["Net Income"].iloc[-1]
        years = len(financials) - 1

        revenue_growth_5y = ((rev_end / rev_start) ** (1 / years)) - 1
        earnings_growth_5y = ((earn_end / earn_start) ** (1 / years)) - 1
    except:
        revenue_growth_5y = None
        earnings_growth_5y = None

    # === Analyst Estimates ===
    try:
        est = ticker.analysis
        revenue_next_q = est.loc["Revenue Estimate"].iloc[0]
        eps_next_q = est.loc["Earnings Estimate"].iloc[0]
        revenue_next_y = est.loc["Revenue Estimate"].iloc[1]
    except:
        revenue_next_q = eps_next_q = revenue_next_y = None

    # === Historical Performance ===
    hist = ticker.history(period="5y")
    latest = hist["Close"].iloc[-1]

    def calc_return(offset_days, annualized=False):
        try:
            past_price = hist["Close"].iloc[-offset_days]
            raw_return = (latest - past_price) / past_price
            if annualized:
                years = offset_days / 252
                return (latest / past_price) ** (1 / years) - 1
            return raw_return
        except:
            return None

    returns = {
        "YTD": calc_return(len(hist.loc[hist.index.year == pd.Timestamp.now().year])),
        "1M": calc_return(21),
        "3M": calc_return(63),
        "1Y": calc_return(252),
        "3Y": calc_return(3 * 252, annualized=True),
        "5Y": calc_return(5 * 252, annualized=True),
        "52W": calc_return(252),
    }

    # === Format into JS-style structure ===
    metrics = [
        {
            "label": "5Y Revenue Growth (Annual)",
            "value": f"{revenue_growth_5y * 100:.2f}%" if revenue_growth_5y else "N/A",
            "tooltip": "Average annual revenue growth over the past 5 years"
        },
        {
            "label": "5Y Earnings Growth (Annual)",
            "value": f"{earnings_growth_5y * 100:.2f}%" if earnings_growth_5y else "N/A",
            "tooltip": "Average annual earnings growth over the past 5 years"
        },
        {
            "label": "Revenue Estimate Next Q",
            "value": f"${revenue_next_q / 1e9:.2f}B" if revenue_next_q else "N/A",
            "tooltip": "Average analyst estimate for next quarter revenue"
        },
        {
            "label": "EPS Estimate Next Q",
            "value": f"${eps_next_q:.2f}" if eps_next_q else "N/A",
            "tooltip": "Average analyst estimate for next quarter earnings per share"
        },
        {
            "label": "Revenue Estimate Next Y",
            "value": f"${revenue_next_y / 1e9:.2f}B" if revenue_next_y else "N/A",
            "tooltip": "Average analyst estimate for next year revenue"
        },
        {
            "label": "YTD Performance",
            "value": f"{returns['YTD']*100:.2f}%" if returns["YTD"] else "N/A",
            "tooltip": "Year-to-date performance"
        },
        {
            "label": "1-Month Return",
            "value": f"{returns['1M']*100:.2f}%" if returns["1M"] else "N/A",
            "tooltip": "Return over the past month"
        },
        {
            "label": "3-Month Return",
            "value": f"{returns['3M']*100:.2f}%" if returns["3M"] else "N/A",
            "tooltip": "Return over the past three months"
        },
        {
            "label": "1-Year Return",
            "value": f"{returns['1Y']*100:.2f}%" if returns["1Y"] else "N/A",
            "tooltip": "Return over the past year"
        },
        {
            "label": "3-Year Return (Annual)",
            "value": f"{returns['3Y']*100:.2f}%" if returns["3Y"] else "N/A",
            "tooltip": "Annualized return over the past three years"
        },
        {
            "label": "5-Year Return (Annual)",
            "value": f"{returns['5Y']*100:.2f}%" if returns["5Y"] else "N/A",
            "tooltip": "Annualized return over the past five years"
        },
        {
            "label": "52-Week Change",
            "value": f"{returns['52W']*100:.2f}%" if returns["52W"] else "N/A",
            "tooltip": "Price change over the past 52 weeks"
        },
    ]

    return metrics

# === Example usage ===
if __name__ == "__main__":
    ticker_symbol = "AAPL"
    stats = get_stock_metrics(ticker_symbol)
    for s in stats:
        print(f"{s['label']}: {s['value']} — {s['tooltip']}")


