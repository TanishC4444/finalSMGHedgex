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

import pandas as pd

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table['Symbol'].tolist()

tickers = get_sp500_tickers()
print(tickers)