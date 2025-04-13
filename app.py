import os
import threading
import webbrowser
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import requests
import pytz
from datetime import datetime, time


app = Flask(__name__)
app.secret_key = os.urandom(24)

USER_DATA_FILE = "users.csv"
SOLD_DATA_FILE = "sold.csv"
BALANCE_FILE = "balance.txt"
TRANSACTION_FILE = "transactions.csv"  # New transaction log file
STARTING_BALANCE = 100000.00  # Initial balance

def is_market_hours():
    """Check if current time is during market hours (8:30 AM - 3:00 PM CST)"""
    # Get current time in CST
    cst = pytz.timezone('US/Central')
    current_time = datetime.now(cst)
    
    # Check if it's a weekday (0 = Monday, 4 = Friday)
    is_weekday = current_time.weekday() <= 4
    
    # Define market hours (8:30 AM - 3:00 PM CST)
    market_start = time(8, 30)
    market_end = time(15, 0)
    
    # Check if current time is within market hours
    current_time_only = current_time.time()
    during_market_hours = market_start <= current_time_only <= market_end
    
    return is_weekday and during_market_hours

# Initialize necessary files
def init_files():
    # Ensure user data file exists
    if not os.path.exists(USER_DATA_FILE) or os.stat(USER_DATA_FILE).st_size == 0:
        df = pd.DataFrame(columns=["Ticker", "Date Bought", "Quantity", "Price Bought", "Current Value", "Earnings", "Change %"])
        df.to_csv(USER_DATA_FILE, index=False)
    
    # Ensure sold data file exists
    if not os.path.exists(SOLD_DATA_FILE):
        pd.DataFrame(columns=["Ticker", "Date Bought", "Date Sold", "Quantity", "Buy Price", "Sell Price", "Profit"]).to_csv(SOLD_DATA_FILE, index=False)
    
    # Ensure balance file exists
    if not os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE, "w") as f:
            f.write(str(STARTING_BALANCE))
            
    # Ensure transaction log file exists
    if not os.path.exists(TRANSACTION_FILE):
        pd.DataFrame(columns=["Date", "Type", "Ticker", "Quantity", "Price", "Total", "Balance After"]).to_csv(TRANSACTION_FILE, index=False)

# Initialize files at startup
init_files()

# Function to log transaction
def log_transaction(transaction_type, ticker, quantity, price, total, balance_after):
    # Read existing transactions
    if os.path.exists(TRANSACTION_FILE):
        transactions_df = pd.read_csv(TRANSACTION_FILE)
    else:
        transactions_df = pd.DataFrame(columns=["Date", "Type", "Ticker", "Quantity", "Price", "Total", "Balance After"])
    
    # Create new transaction entry
    new_transaction = pd.DataFrame({
        "Date": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        "Type": [transaction_type],  # "BUY" or "SELL" or "RESET"
        "Ticker": [ticker],
        "Quantity": [quantity],
        "Price": [price],
        "Total": [total],
        "Balance After": [balance_after]
    })
    
    # Add to transaction log
    transactions_df = pd.concat([transactions_df, new_transaction], ignore_index=True)
    transactions_df.to_csv(TRANSACTION_FILE, index=False)

# Function to get balance
def get_balance():
    try:
        with open(BALANCE_FILE, "r") as f:
            return float(f.read())
    except (FileNotFoundError, ValueError):
        # If file doesn't exist or contains invalid data, recreate it
        with open(BALANCE_FILE, "w") as f:
            f.write(str(STARTING_BALANCE))
        return STARTING_BALANCE

# Function to update balance
def update_balance(amount):
    balance = get_balance() + amount
    with open(BALANCE_FILE, "w") as f:
        f.write(str(balance))
    return balance

# Also update the original sell_stock endpoint to work with the positionId parameter if provided
@app.route("/sell_stock", methods=["POST"])
def sell_stock():
    if not is_market_hours():
        return jsonify({
            "error": "Trading is only available during market hours (8:30 AM - 3:00 PM CST)"
        }), 400
    try:
        ticker = request.args.get("ticker")
        position_id = request.args.get("position_id")
        
        if not ticker:
            return jsonify({"error": "Missing ticker"}), 400
        
        if not position_id:
            return jsonify({"error": "Missing position ID"}), 400
            
        try:
            position_id = int(position_id)
        except ValueError:
            return jsonify({"error": "Invalid position ID"}), 400

        df = pd.read_csv(USER_DATA_FILE)

        # Check if position exists
        if position_id < 0 or position_id >= len(df):
            return jsonify({"error": "Position not found"}), 404
            
        # Check if ticker matches the position
        if df.iloc[position_id]["Ticker"] != ticker:
            return jsonify({"error": "Ticker does not match position ID"}), 400

        # Get the stock data for the specific position
        stock_row = df.iloc[position_id]
        quantity = float(stock_row["Quantity"])
        purchase_price = float(stock_row["Price Bought"])
        
        # Get current stock price
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d", interval="1m").iloc[-1]['Close']
        current_value = current_price * quantity
        earnings = (current_price - purchase_price) * quantity
        
        # Add to sold.csv
        sold_df = pd.read_csv(SOLD_DATA_FILE)
        new_sale = pd.DataFrame({
            "Ticker": [ticker],
            "Date Bought": [stock_row["Date Bought"]],
            "Date Sold": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            "Quantity": [quantity],
            "Buy Price": [purchase_price],
            "Sell Price": [current_price],
            "Profit": [earnings]
        })
        sold_df = pd.concat([sold_df, new_sale], ignore_index=True)
        sold_df.to_csv(SOLD_DATA_FILE, index=False)
        
        # Remove the specific position from the user's holdings
        df = df.drop(position_id)
        df = df.reset_index(drop=True)  # Reset index after dropping row
        df.to_csv(USER_DATA_FILE, index=False)
        
        # Update the balance
        new_balance = update_balance(current_value)
        
        # Log the transaction
        log_transaction("SELL", ticker, quantity, current_price, current_value, new_balance)
        
        return jsonify({
            "message": f"Sold {quantity} shares of {ticker}",
            "new_balance": new_balance
        })

    except Exception as e:
        print(f"Error in sell_stock: {str(e)}")  # Add logging
        return jsonify({"error": str(e)}), 500

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/transactions', methods=["GET"])
def get_transactions():
    # Return all transactions
    if not os.path.exists(TRANSACTION_FILE):
        return jsonify([])
    
    transactions_df = pd.read_csv(TRANSACTION_FILE)
    return jsonify(transactions_df.to_dict(orient="records"))

@app.route('/update_holdings', methods=["GET"])
def get_updated_holdings():
    try:
        if not os.path.exists(USER_DATA_FILE) or os.stat(USER_DATA_FILE).st_size == 0:
            return jsonify([])
            
        df = pd.read_csv(USER_DATA_FILE)

        if df.empty:
            return jsonify([])  # Return empty list if no data

        # Convert all string columns to string to ensure consistent types
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)

        # Ensure numeric columns are properly converted
        numeric_cols = ['Quantity', 'Price Bought', 'Current Value', 'Earnings', 'Change %']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for index, row in df.iterrows():
            ticker = row["Ticker"]
            quantity = float(row["Quantity"])
            price_bought = float(row["Price Bought"])
            
            # Fetch current stock price
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d", interval="1m").iloc[-1]['Close']

            # Update values
            current_value = round(current_price * quantity, 2)
            earnings = round((current_price - price_bought) * quantity, 2)
            change_percent = round(((current_price - price_bought) / price_bought) * 100, 2)

            df.at[index, "Current Value"] = current_value
            df.at[index, "Earnings"] = earnings
            df.at[index, "Change %"] = change_percent

        # Save the updated data
        df.to_csv(USER_DATA_FILE, index=False)
        
        # Convert to records and ensure all values are properly formatted for JSON
        records = []
        for _, row in df.iterrows():
            record = row.to_dict()
            for key, value in record.items():
                if isinstance(value, float):
                    record[key] = float(value)
            records.append(record)
            
        return jsonify(records)
    
    except Exception as e:
        print(f"Error in update_holdings: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/practice')
def practice():
    return render_template('practice.html')

@app.route('/learn')  # Added route for learn.html
def learn():
    return render_template('learn.html')

@app.route('/balance', methods=['GET'])
def balance():
    return jsonify({"balance": get_balance()})

import re  # Add this import at the top with the other imports

@app.route('/simulator')
def simulator():
    return render_template("simulator.html")

def calculate_movement(data):
    movements = []
    # Explicitly use the entire data range
    for i in range(len(data) - 1):  # Exclude the last row
        next_open = data.iloc[i + 1]["Open"]
        next_close = data.iloc[i + 1]["Close"]
        movement = "+" if (next_close - next_open) > 0 else "-"
        movements.append(movement)
    
    movements.append(None)  # Last row has no movement data
    return movements

@app.route('/evaluate_equation', methods=['POST'])
def evaluate_equation():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        period = data.get('period', '1mo')
        interval = data.get('interval', '1d')
        equation = data.get('equation')
        
        if not ticker or not equation:
            return jsonify({"error": "Missing required parameters"}), 400
            
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        # Print debug info
        print(f"Retrieved {len(data)} data points for {ticker} with period={period}, interval={interval}")
        
        # Add fundamental data to each row
        fundamentals = stock.info
        
        # Function to evaluate the equation with added fundamentals
        def evaluate_equation_points(data, equation, fundamentals):
            # Create a DataFrame with reset index for easier access
            df = data.reset_index()
            
            # Add fundamental metrics to each row
            fundamentals_to_add = {
                # Basic info
                "symbol": fundamentals.get("symbol"),
                "shortName": fundamentals.get("shortName"),
                "sector": fundamentals.get("sector"),
                "industry": fundamentals.get("industry"),
                
                # Valuation metrics
                "marketCap": fundamentals.get("marketCap"),
                "enterpriseValue": fundamentals.get("enterpriseValue"),
                "PE": fundamentals.get("trailingPE"),
                "forwardPE": fundamentals.get("forwardPE"),
                "PEG": fundamentals.get("pegRatio"),
                "priceToSales": fundamentals.get("priceToSalesTrailing12Months"),
                "priceToBook": fundamentals.get("priceToBook"),
                "enterpriseToRevenue": fundamentals.get("enterpriseToRevenue"),
                "enterpriseToEbitda": fundamentals.get("enterpriseToEbitda"),
                
                # Dividends
                "dividendRate": fundamentals.get("dividendRate"),
                "dividendYield": fundamentals.get("dividendYield"),
                "payoutRatio": fundamentals.get("payoutRatio"),
                
                # Financial metrics
                "bookValue": fundamentals.get("bookValue"),
                "beta": fundamentals.get("beta"),
                "ROA": fundamentals.get("returnOnAssets"),
                "ROE": fundamentals.get("returnOnEquity"),
                "revenueGrowth": fundamentals.get("revenueGrowth"),
                "grossMargins": fundamentals.get("grossMargins"),
                "operatingMargins": fundamentals.get("operatingMargins"),
                "profitMargins": fundamentals.get("profitMargins"),
                "earningsGrowth": fundamentals.get("earningsGrowth"),
                "currentRatio": fundamentals.get("currentRatio"),
                "quickRatio": fundamentals.get("quickRatio"),
                "debtToEquity": fundamentals.get("debtToEquity"),
                
                # Trading info
                "avgVolume": fundamentals.get("averageVolume"),
                "avgVolume10days": fundamentals.get("averageVolume10days"),
                "sharesOutstanding": fundamentals.get("sharesOutstanding"),
                "floatShares": fundamentals.get("floatShares"),
                "heldPercentInsiders": fundamentals.get("heldPercentInsiders"),
                "heldPercentInstitutions": fundamentals.get("heldPercentInstitutions"),
                "shortRatio": fundamentals.get("shortRatio"),
                "shortPercentOfFloat": fundamentals.get("shortPercentOfFloat"),
                
                # Price metrics
                "fiftyTwoWeekHigh": fundamentals.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": fundamentals.get("fiftyTwoWeekLow"),
                "previousClose": fundamentals.get("previousClose"),
                "fiftyDayAverage": fundamentals.get("fiftyDayAverage"),
                "twoHundredDayAverage": fundamentals.get("twoHundredDayAverage"),
            }
            
            # Dictionary of column references with fundamentals added
            column_replacements = {col: f"row['{col}']" for col in df.columns}
            
            # Add fundamentals to column replacements
            for key, value in fundamentals_to_add.items():
                if value is not None:  # Only add non-None values
                    column_replacements[key] = str(value)
            
            # Sort by length to replace longer names first (avoiding partial replacements)
            sorted_cols = sorted(column_replacements.keys(), key=len, reverse=True)
            
            # Replace column names in the equation
            eq_eval = equation
            for col in sorted_cols:
                # Use word boundary regex to avoid partial replacements
                pattern = r'\b' + re.escape(str(col)) + r'\b'
                eq_eval = re.sub(pattern, column_replacements[col], eq_eval)
            
            points = []
            for idx, row in df.iterrows():
                try:
                    # Convert row to a dictionary and add fundamentals
                    row_dict = row.to_dict()
                    # Merge in fundamentals to each row
                    for key, value in fundamentals_to_add.items():
                        row_dict[key] = value
                        
                    result = eval(eq_eval, {"__builtins__": {}}, {"row": row_dict, "math": __import__('math')})
                    points.append(float(result))
                except Exception as e:
                    points.append(None)
                    print(f"Error evaluating equation at row {idx}: {e}")
            
            return points
        
        # Function to calculate next day movement direction
        def calculate_movement(data):
            movements = []
            
            for i in range(len(data)):
                try:
                    # Current day movement
                    if i < len(data) - 1:
                        # Predict if TOMORROW will close higher than its open
                        next_open = data.iloc[i+1]["Open"]
                        next_close = data.iloc[i+1]["Close"]
                        movement = "+" if (next_close > next_open) else "-"
                    else:
                        # Last data point has no next day to predict
                        movement = None
                        
                    movements.append(movement)
                except Exception as e:
                    print(f"Error calculating movement at index {i}: {e}")
                    movements.append(None)
            
            return movements
            
        # Calculate prediction accuracy
        def calculate_accuracy(points, movements):
            correct_predictions = 0
            total_valid = 0
            
            # Count valid predictions and correct ones
            for i in range(len(points)):
                if i < len(movements) and points[i] is not None and movements[i] is not None:
                    predicted_sign = "+" if points[i] > 0 else "-"
                    if predicted_sign == movements[i]:
                        correct_predictions += 1
                    total_valid += 1
            
            accuracy = (correct_predictions / total_valid) * 100 if total_valid > 0 else 0
            return accuracy, correct_predictions, total_valid
        
        # Evaluate equation with fundamentals included
        points = evaluate_equation_points(data, equation, fundamentals)
        
        # Calculate movements
        movements = calculate_movement(data)
        
        # Calculate accuracy
        accuracy, correct_predictions, total_valid = calculate_accuracy(points, movements)
        
        # Prepare result
        result_data = []
        dates = data.index.strftime('%Y-%m-%d').tolist()
        
        for i in range(len(dates)):
            point_value = None if i >= len(points) or points[i] is None else float(points[i])
            movement_value = None if i >= len(movements) or movements[i] is None else movements[i]
            
            result_data.append({
                "date": dates[i],
                "point": point_value,
                "movement": movement_value
            })

        # Add fundamental columns to the response
        fundamental_columns = []
        for key in fundamentals:
            if fundamentals[key] is not None and not isinstance(fundamentals[key], dict) and not isinstance(fundamentals[key], list):
                fundamental_columns.append(key)
                
        return jsonify({
            "results": result_data,
            "accuracy": round(accuracy, 2),
            "correct_predictions": correct_predictions,
            "total_predictions": total_valid,
            "columns": sorted(data.reset_index().columns.tolist()),
            "fundamental_columns": sorted(fundamental_columns)
        })
        
    except Exception as e:
        print(f"Error in evaluate_equation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/available_columns', methods=['GET'])
def get_available_columns():
    try:
        ticker = request.args.get('ticker')
        period = request.args.get('period', '1mo')
        interval = request.args.get('interval', '1d')
        
        if not ticker:
            return jsonify({"error": "Missing ticker parameter"}), 400
            
        # Get stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        # Get fundamental data
        fundamentals = stock.info
        
        # Important fundamental metrics to prioritize
        priority_fundamentals = [
            "marketCap", "beta", "trailingPE", "forwardPE", "priceToBook",
            "dividendYield", "returnOnEquity", "returnOnAssets", "profitMargins",
            "operatingMargins", "grossMargins", "debtToEquity",
            "currentRatio", "quickRatio", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "fiftyDayAverage", "twoHundredDayAverage", "shortRatio"
        ]
        
        # Filter and process fundamental data for display
        fundamental_columns = []
        important_fundamentals = []
        
        for key in fundamentals:
            # Check if value exists and is not complex
            value = fundamentals.get(key)
            if value is not None and not isinstance(value, (dict, list)):
                if key in priority_fundamentals:
                    important_fundamentals.append({
                        "name": key,
                        "value": value if not isinstance(value, (float, int)) else round(value, 4)
                    })
                else:
                    fundamental_columns.append(key)
        
        # Sort important fundamentals by priority
        important_fundamentals_sorted = sorted(
            important_fundamentals, 
            key=lambda x: priority_fundamentals.index(x["name"]) if x["name"] in priority_fundamentals else 999
        )
        
        # Return both price data columns and fundamental columns
        price_columns = sorted(data.reset_index().columns.tolist())
        
        return jsonify({
            "columns": price_columns,
            "important_fundamentals": [item["name"] for item in important_fundamentals_sorted],
            "fundamental_columns": sorted(fundamental_columns),
            "time_periods": [
                "1d", "5d", "1wk", "1mo", "3mo", "6mo", "1y", 
                "2y", "5y", "10y", "ytd", "max"
            ],
            "intervals": [
                "1m", "2m", "5m", "15m", "30m", "1h", "1d", 
                "5d", "1wk", "1mo", "3mo"
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return jsonify({"error": "Missing ticker"}), 400

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="5m")

        if hist.empty:
            return jsonify({"error": "Invalid ticker or no data available"}), 400

        timestamps = hist.index.strftime('%H:%M').tolist()
        prices = hist['Close'].round(2).tolist()
        stock_info = stock.info

        extra_data = {
            "Previous Close": stock_info.get("previousClose"),
            "Open": stock_info.get("open"),
            "Day High": stock_info.get("dayHigh"),
            "Day Low": stock_info.get("dayLow"),
            "Beta": stock_info.get("beta"),
            "52-Week High": stock_info.get("fiftyTwoWeekHigh"),
            "52-Week Low": stock_info.get("fiftyTwoWeekLow"),
            "Market Cap": stock_info.get("marketCap"),
            "Volume": stock_info.get("volume"),
            "Avg Volume": stock_info.get("averageVolume"),
        }

        return jsonify({"timestamps": timestamps, "prices": prices, "extra_data": extra_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_trades', methods=['GET'])
def get_trades():
    if not os.path.exists(USER_DATA_FILE) or os.stat(USER_DATA_FILE).st_size == 0:
        return jsonify([])

    df = pd.read_csv(USER_DATA_FILE)

    # Round all numerical values
    df["Price Bought"] = df["Price Bought"].round(2)
    df["Current Value"] = df["Current Value"].round(2)
    df["Earnings"] = df["Earnings"].round(2)
    df["Change %"] = df["Change %"].round(2)

    trades = df.to_dict(orient="records")  
    return jsonify(trades)

@app.route('/trade', methods=['GET'])
def trade_stock():
    if not is_market_hours():
        return jsonify({
            "error": "Trading is only available during market hours (8:30 AM - 3:00 PM CST)"
        }), 400
    ticker = request.args.get('ticker', '').upper()
    quantity = int(request.args.get('quantity', 0))
    trade_type = request.args.get('type', '').lower()
    if not ticker or quantity <= 0 or trade_type not in ['buy', 'sell']:
        return jsonify({"error": "Invalid trade parameters"}), 400
    try:
        stock = yf.Ticker(ticker)
        current_price = round(stock.history(period="1d", interval="1m").iloc[-1]['Close'], 2)
        total_cost = round(quantity * current_price, 2)
        # Read existing data
        df = pd.read_csv(USER_DATA_FILE)
        if trade_type == "buy":
            balance = get_balance()
            if total_cost > balance:
                return jsonify({"error": "Insufficient funds"}), 400
            # Always add as a new entry for better transaction tracking
            new_trade = pd.DataFrame({
                "Ticker": [ticker], 
                "Date Bought": [datetime.now().strftime('%Y-%m-%d %H:%M')],
                "Quantity": [quantity], 
                "Price Bought": [current_price], 
                "Current Value": [total_cost],
                "Earnings": [0], 
                "Change %": [0]
            })
            df = pd.concat([df, new_trade], ignore_index=True)
            
            # Update balance and log transaction
            new_balance = update_balance(-total_cost)
            log_transaction("BUY", ticker, quantity, current_price, total_cost, new_balance)
        elif trade_type == "sell":
            # Get all rows with the ticker we want to sell
            stock_rows = df[df["Ticker"] == ticker]
            if stock_rows.empty:
                return jsonify({"error": "No shares available to sell"}), 400
                
            # Calculate total shares owned for this ticker
            total_owned = stock_rows["Quantity"].sum()
            
            # Check if selling more than owned
            if quantity > total_owned:
                return jsonify({"error": f"You only own {total_owned} shares"}), 400
                
            sell_value = quantity * current_price
            remaining_to_sell = quantity
            earnings = 0
            sold_records = []
            
            # Process sale using FIFO (First In, First Out) method
            # Sort by date bought to ensure we sell oldest shares first
            stock_rows = stock_rows.sort_values("Date Bought")
            
            # Prepare sold_df for appending records
            sold_df = pd.read_csv(SOLD_DATA_FILE)
            
            # Track rows to update or remove
            rows_to_remove = []
            
            # Process each lot until we've sold all requested shares
            for index, row in stock_rows.iterrows():
                if remaining_to_sell <= 0:
                    break
                    
                buy_price = float(row["Price Bought"])
                lot_quantity = float(row["Quantity"])
                
                # Determine how many shares to sell from this lot
                selling_from_lot = min(remaining_to_sell, lot_quantity)
                lot_earnings = (current_price - buy_price) * selling_from_lot
                
                # Add to overall earnings
                earnings += lot_earnings
                
                # Create record for sold.csv
                new_sale = {
                    "Ticker": ticker, 
                    "Date Bought": row["Date Bought"],
                    "Date Sold": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "Quantity": selling_from_lot, 
                    "Buy Price": buy_price, 
                    "Sell Price": current_price,
                    "Profit": lot_earnings
                }
                sold_records.append(new_sale)
                
                # Update remaining quantity
                if selling_from_lot == lot_quantity:
                    # Remove this row completely
                    rows_to_remove.append(index)
                else:
                    # Update quantity in this row
                    new_quantity = lot_quantity - selling_from_lot
                    df.at[index, "Quantity"] = new_quantity
                    df.at[index, "Current Value"] = round(new_quantity * current_price, 2)
                
                # Update remaining to sell
                remaining_to_sell -= selling_from_lot
            
            # Remove any rows that have been completely sold
            if rows_to_remove:
                df = df.drop(rows_to_remove)
            
            # Add sold records to sold.csv
            sold_df = pd.concat([sold_df, pd.DataFrame(sold_records)], ignore_index=True)
            sold_df.to_csv(SOLD_DATA_FILE, index=False)
            
            # Update balance and log transaction
            new_balance = update_balance(sell_value)
            log_transaction("SELL", ticker, quantity, current_price, sell_value, new_balance)
            
        # Save updated holdings
        df.to_csv(USER_DATA_FILE, index=False)
        
        return jsonify({
            "success": f"{trade_type.capitalize()} {quantity} shares of {ticker} at ${current_price:.2f}", 
            "balance": get_balance()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stock_analysis', methods=['GET'])
def stock_analysis():
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return jsonify({"error": "Missing ticker"}), 400
    
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Add current price and change calculations
    hist = stock.history(period="1d")
    if len(hist) >= 2:
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100
    else:
        current_price = info.get('currentPrice', info.get('previousClose', 0))
        change = 0
        change_percent = 0
    
    # Add the current price and change to the response
    info['currentPrice'] = current_price
    info['change'] = change
    info['changePercent'] = change_percent
    
    return jsonify(info)

@app.route('/stock_chart', methods=['GET'])
def stock_chart():
    ticker = request.args.get('ticker', '').upper()
    period = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    
    if not ticker:
        return jsonify({"error": "Missing ticker"}), 400
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    
    timestamps = hist.index.strftime('%Y-%m-%d').tolist()
    prices = hist['Close'].round(2).tolist()
    
    return jsonify({"timestamps": timestamps, "prices": prices})

@app.route('/quarterly_data', methods=['GET'])
def quarterly_data():
    ticker = request.args.get('ticker', '').upper()
    if not ticker:
        return jsonify({"error": "Missing ticker"}), 400
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get quarterly financials
        quarterly_financials = stock.quarterly_financials
        quarterly_balance_sheet = stock.quarterly_balance_sheet
        quarterly_cashflow = stock.quarterly_cashflow
        
        # Check if we got data
        if quarterly_financials is None or quarterly_financials.empty:
            return jsonify({"quarters": [], "error": "No quarterly data available"})
        
        # Transpose for easier manipulation
        financials_df = quarterly_financials.T
        
        # Get dates from financials as they should be consistent across statements
        quarters = []
        
        for date, row in financials_df.iterrows():
            quarter_data = {
                "quarter": f"{date.strftime('%Y Q%q').replace('Q%q', f'Q{(date.month-1)//3+1}')} ({date.strftime('%b %d, %Y')})",
                "date": date.strftime('%Y-%m-%d'),
            }
            
            # Add key financial metrics from income statement
            if "Total Revenue" in quarterly_financials.index:
                quarter_data["revenue"] = float(row.get("Total Revenue", 0)) if not pd.isna(row.get("Total Revenue", 0)) else 0
            
            if "Net Income" in quarterly_financials.index:
                quarter_data["netIncome"] = float(row.get("Net Income", 0)) if not pd.isna(row.get("Net Income", 0)) else 0
            
            if "Operating Income" in quarterly_financials.index:
                quarter_data["operatingIncome"] = float(row.get("Operating Income", 0)) if not pd.isna(row.get("Operating Income", 0)) else 0
            
            if "Gross Profit" in quarterly_financials.index:
                quarter_data["grossProfit"] = float(row.get("Gross Profit", 0)) if not pd.isna(row.get("Gross Profit", 0)) else 0
            
            # Add balance sheet data if available for the same date
            if quarterly_balance_sheet is not None and not quarterly_balance_sheet.empty and date in quarterly_balance_sheet.columns:
                balance_data = quarterly_balance_sheet[date]
                
                if "Total Assets" in balance_data:
                    quarter_data["totalAssets"] = float(balance_data["Total Assets"]) if not pd.isna(balance_data["Total Assets"]) else 0
                
                if "Total Liabilities Net Minority Interest" in balance_data:
                    quarter_data["totalLiabilities"] = float(balance_data["Total Liabilities Net Minority Interest"]) if not pd.isna(balance_data["Total Liabilities Net Minority Interest"]) else 0
                
                if "Total Equity Gross Minority Interest" in balance_data:
                    quarter_data["totalEquity"] = float(balance_data["Total Equity Gross Minority Interest"]) if not pd.isna(balance_data["Total Equity Gross Minority Interest"]) else 0
            
            # Add cash flow data if available
            if quarterly_cashflow is not None and not quarterly_cashflow.empty and date in quarterly_cashflow.columns:
                cashflow_data = quarterly_cashflow[date]
                
                if "Operating Cash Flow" in cashflow_data:
                    quarter_data["operatingCashFlow"] = float(cashflow_data["Operating Cash Flow"]) if not pd.isna(cashflow_data["Operating Cash Flow"]) else 0
                
                if "Free Cash Flow" in cashflow_data:
                    quarter_data["freeCashFlow"] = float(cashflow_data["Free Cash Flow"]) if not pd.isna(cashflow_data["Free Cash Flow"]) else 0
            
            # Calculate key financial ratios
            if quarter_data.get("revenue", 0) > 0:
                if "grossProfit" in quarter_data:
                    quarter_data["grossMargin"] = round(quarter_data["grossProfit"] / quarter_data["revenue"] * 100, 2)
                    
                if "operatingIncome" in quarter_data:
                    quarter_data["operatingMargin"] = round(quarter_data["operatingIncome"] / quarter_data["revenue"] * 100, 2)
                    
                if "netIncome" in quarter_data:
                    quarter_data["netMargin"] = round(quarter_data["netIncome"] / quarter_data["revenue"] * 100, 2)
            
            quarters.append(quarter_data)
        
        # Sort by date (most recent first)
        quarters = sorted(quarters, key=lambda x: x["date"], reverse=True)
        
        # Remove the last (oldest) quarter to avoid incomplete data
        if len(quarters) > 0:
            quarters = quarters[:-1]
        
        return jsonify({"quarters": quarters})
    
    except Exception as e:
        print(f"Error in quarterly_data: {str(e)}")
        return jsonify({"error": str(e), "quarters": []}), 500

def evaluate_equation_for_ticker(ticker, period, interval, equation):
    """
    Evaluates a prediction equation for a specific ticker and returns accuracy metrics.
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return {"error": f"No data available for {ticker}"}
        
        # Print debug info
        print(f"Retrieved {len(df)} data points for {ticker}")
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Add commonly used indicators/metrics
        if len(df) > 5:  # Only calculate if we have enough data
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
        if len(df) > 20:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
        if len(df) > 50:
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
        # Price changes
        df['PriceChange'] = df['Close'] - df['Open']
        df['PriceChangePct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Drop rows with NaN values that might have been introduced by indicators
        df = df.dropna()
        
        if len(df) < 2:
            return {"error": f"Insufficient data for {ticker} after processing"}
        
        # Calculate the equation result for each row
        results = []
        valid_points = 0
        correct_predictions = 0
        signal_sum = 0
        
        safe_globals = {
            "__builtins__": {
                "abs": abs, "max": max, "min": min, "round": round, 
                "pow": pow, "sum": sum, "len": len, "float": float
            },
            "math": __import__('math')
        }
        
        # Process ALL rows in the dataframe
        for idx in range(len(df)):
            try:
                # Create a safe evaluation environment with row data
                row_dict = df.iloc[idx].to_dict()
                
                # Safely evaluate the equation
                point = eval(equation, safe_globals, row_dict)
                
                # Make sure we have a numeric result
                if point is not None:
                    point = float(point)
                
                # Determine next-day price movement
                if idx < len(df) - 1:
                    next_row = df.iloc[idx + 1]
                    actual_movement = next_row['Close'] - next_row['Open']
                    movement = "+" if actual_movement > 0 else "-"
                    
                    # Check if prediction was correct
                    if point is not None:
                        valid_points += 1
                        signal_sum += abs(point)
                        
                        predicted_direction = "+" if point > 0 else "-"
                        if predicted_direction == movement:
                            correct_predictions += 1
                else:
                    movement = None  # Last row has no next day to predict
                
                # Add to results
                results.append({
                    'date': df.index[idx].strftime('%Y-%m-%d') if hasattr(df.index[idx], 'strftime') else str(df.index[idx]),
                    'point': point,
                    'movement': movement
                })
            except Exception as e:
                print(f"Error processing row {idx} for {ticker}: {str(e)}")
                # Skip this row if evaluation fails
                results.append({
                    'date': df.index[idx].strftime('%Y-%m-%d') if hasattr(df.index[idx], 'strftime') else str(df.index[idx]),
                    'point': None,
                    'movement': None
                })
        
        # Calculate accuracy consistently over all available data points
        accuracy = 0
        if valid_points > 0:
            accuracy = round((correct_predictions / valid_points) * 100, 2)
            
        # Calculate average signal strength
        avg_signal = None
        if valid_points > 0:
            avg_signal = round(signal_sum / valid_points, 4)
        
        print(f"Final stats for {ticker}: {valid_points} valid points, {correct_predictions} correct, {accuracy}% accuracy")
        
        return {
            'ticker': ticker,
            'accuracy': accuracy,
            'avg_signal': avg_signal,
            'results': results,
            'valid_points': valid_points,
            'correct_predictions': correct_predictions
        }
        
    except Exception as e:
        print(f"Error evaluating {ticker}: {str(e)}")
        return {"error": str(e), "ticker": ticker}

@app.route('/evaluate_sp500', methods=['POST'])
def evaluate_sp500():
    try:
        data = request.get_json()
        equation = data.get('equation')
        period = data.get('period', '1mo')
        interval = data.get('interval', '1d')
        limit = int(data.get('limit', 30))  # Add limit parameter with default of 30 stocks
        
        if not equation:
            return jsonify({"error": "Missing equation"}), 400
            
        # Get S&P 500 tickers using pandas_datareader
        try:
            # Try to get real S&P 500 components
            import pandas_datareader.data as web
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            sp500_table = pd.read_html(sp500_url)[0]
            sp500_tickers = sp500_table['Symbol'].tolist()
            
            # Limit the number of tickers to process
            sp500_tickers = sp500_tickers[:limit]
        except Exception as e:
            # Fallback to a sample of major stocks if we can't get the real S&P 500
            print(f"Error getting S&P 500 tickers: {str(e)}")
            sp500_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'PG', 'UNH', 
                             'JNJ', 'WMT', 'MA', 'HD', 'BAC', 'XOM', 'PFE', 'NVDA', 'DIS', 'NFLX',
                             'INTC', 'VZ', 'ADBE', 'CSCO', 'CRM', 'CMCSA', 'KO', 'PEP', 'ABT', 'MRK'][:limit]
        
        results = []
        total_valid_points = 0
        total_correct_predictions = 0
        successful_tickers = 0
        
        # Process each ticker using the existing function
        for ticker in sp500_tickers:
            try:
                print(f"Processing {ticker}...")
                result = evaluate_equation_for_ticker(ticker, period, interval, equation)
                
                if 'error' not in result and result.get('valid_points', 0) > 0:
                    # Add to overall counts for aggregate accuracy
                    total_valid_points += result.get('valid_points', 0)
                    total_correct_predictions += result.get('correct_predictions', 0)
                    successful_tickers += 1
                    
                    # Store individual ticker result with more details
                    ticker_result = {
                        'ticker': ticker,
                        'accuracy': result['accuracy'],
                        'valid_points': result['valid_points'],
                        'correct_predictions': result['correct_predictions'],
                        'avg_signal': result.get('avg_signal')
                    }
                    results.append(ticker_result)
                else:
                    print(f"Skipping {ticker}: {result.get('error', 'No valid predictions')}")
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
                
        # Calculate overall accuracy across all stocks
        overall_accuracy = 0
        if total_valid_points > 0:
            overall_accuracy = round((total_correct_predictions / total_valid_points) * 100, 2)
            
        print(f"Completed processing. Found {successful_tickers}/{len(sp500_tickers)} stocks with valid results.")
        return jsonify({
            'overall_accuracy': overall_accuracy,
            'total_valid_points': total_valid_points,
            'total_correct_predictions': total_correct_predictions,
            'stocks_tested': len(sp500_tickers),
            'stocks_with_data': successful_tickers,
            'results': results  # Individual stock results
        })
        
    except Exception as e:
        print(f"Error in evaluate_sp500: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=["POST"])
def reset_account():
    try:
        # Reset user data file
        df = pd.DataFrame(columns=["Ticker", "Date Bought", "Quantity", "Price Bought", "Current Value", "Earnings", "Change %"])
        df.to_csv(USER_DATA_FILE, index=False)
        
        # Reset sold data file
        pd.DataFrame(columns=["Ticker", "Date Bought", "Date Sold", "Quantity", "Buy Price", "Sell Price", "Profit"]).to_csv(SOLD_DATA_FILE, index=False)
        
        # Reset balance to starting amount
        with open(BALANCE_FILE, "w") as f:
            f.write(str(STARTING_BALANCE))
            
        # Log the reset transaction
        log_transaction("RESET", "SYSTEM", 0, 0, 0, STARTING_BALANCE)
        
        return jsonify({"success": "Account reset successfully", "balance": STARTING_BALANCE})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/download_transactions', methods=["GET"])
def download_transactions():
    try:
        # Check if transaction file exists
        if not os.path.exists(TRANSACTION_FILE):
            return jsonify({"error": "No transaction data available"}), 404

        # Read transaction data
        transactions_df = pd.read_csv(TRANSACTION_FILE)
        
        # Format any monetary values to have consistent decimal places
        money_columns = ['Price', 'Total', 'Balance After']
        for col in money_columns:
            if col in transactions_df.columns:
                transactions_df[col] = transactions_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
        
        # Create a string buffer to write CSV data
        from io import StringIO
        buffer = StringIO()
        transactions_df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Create the response with CSV data
        from flask import Response
        response = Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=transactions.csv"}
        )
        
        return response
    
    except Exception as e:
        print(f"Error downloading transactions: {str(e)}")
        return jsonify({"error": str(e)}), 500

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)