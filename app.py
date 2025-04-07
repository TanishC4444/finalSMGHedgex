import os
import threading
import webbrowser
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import requests

app = Flask(__name__)
app.secret_key = os.urandom(24)

USER_DATA_FILE = "users.csv"
SOLD_DATA_FILE = "sold.csv"
BALANCE_FILE = "balance.txt"
TRANSACTION_FILE = "transactions.csv"  # New transaction log file
STARTING_BALANCE = 100000.00  # Initial balance

# Initialize necessary files
def init_files():
    # Ensure user data file exists
    if not os.path.exists(USER_DATA_FILE) or os.stat(USER_DATA_FILE).st_size == 0:
        df = pd.DataFrame(columns=["Ticker", "Date Bought", "Quantity", "Price Bought", "Current Value", "Earnings", "Change %", "Status"])
        df.to_csv(USER_DATA_FILE, index=False)
    
    # Ensure sold data file exists
    if not os.path.exists(SOLD_DATA_FILE):
        pd.DataFrame(columns=["Ticker", "Date Bought", "Quantity", "Price Bought", "Sell Price", "Profit", "Sell Date"]).to_csv(SOLD_DATA_FILE, index=False)
    
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
        "Type": [transaction_type],  # "BUY" or "SELL"
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

@app.route("/sell_stock", methods=["POST"])
def sell_stock():
    try:
        ticker = request.args.get("ticker")
        if not ticker:
            return jsonify({"error": "Missing ticker"}), 400

        df = pd.read_csv("users.csv")

        # Check if stock exists
        if ticker not in df["Ticker"].values:
            return jsonify({"error": "Stock not found"}), 404

        # Get the stock data before removing
        stock_row = df[df["Ticker"] == ticker].iloc[0]
        quantity = float(stock_row["Quantity"])
        purchase_price = float(stock_row["Price Bought"])
        current_value = float(stock_row["Current Value"])
        price_per_share = current_value / quantity
        
        # Remove the stock from the user's holdings
        df = df[df["Ticker"] != ticker]
        df.to_csv("users.csv", index=False)
        
        # Update the balance
        new_balance = update_balance(current_value)
        
        # Log the transaction
        log_transaction("SELL", ticker, quantity, price_per_share, current_value, new_balance)
        
        return jsonify({
            "message": f"Sold all shares of {ticker}",
            "new_balance": new_balance
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/transactions", methods=["GET"])
def get_transactions():
    # Return all transactions
    if not os.path.exists(TRANSACTION_FILE):
        return jsonify([])
    
    transactions_df = pd.read_csv(TRANSACTION_FILE)
    return jsonify(transactions_df.to_dict(orient="records"))

def update_holdings():
    try:
        df = pd.read_csv("users.csv")

        if df.empty:
            return []  # Return empty list if no data

        for index, row in df.iterrows():
            ticker = row["Ticker"]
            quantity = row["Quantity"]
            
            # Fetch current stock price
            response = requests.get(f"http://localhost:5000/stock?ticker={ticker}")
            if response.status_code == 200:
                stock_data = response.json()
                if stock_data and stock_data["prices"]:
                    current_price = stock_data["prices"][-1]

                    # Ensure numerical values are handled correctly
                    quantity = float(quantity)
                    price_bought = float(row["Price Bought"])

                    # Update values
                    df.at[index, "Current Value"] = round(current_price * quantity, 2)
                    earnings = round((current_price - price_bought) * quantity, 2)
                    change_percent = round(((current_price - price_bought) / price_bought) * 100, 2)

                    df.at[index, "Earnings"] = earnings
                    df.at[index, "Change %"] = float(change_percent)

        df.to_csv("users.csv", index=False)
        return df.to_dict(orient="records")  # Return updated data
    
    except Exception as e:
        return {"error": str(e)}

@app.route("/update_holdings", methods=["GET"])
def get_updated_holdings():
    data = update_holdings()
    return jsonify(data)

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
        
        # Add fundamental data if needed
        fundamentals = stock.info
        
        # Function to evaluate the equation
        def evaluate_equation_points(data, equation):
            # Create a DataFrame with reset index for easier access
            df = data.reset_index()
            
            # Dictionary of column references
            column_replacements = {col: f"row['{col}']" for col in df.columns}
            
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
                    # Convert row to a dictionary
                    row_dict = row.to_dict()
                    result = eval(eq_eval, {"__builtins__": {}}, {"row": row_dict, "math": __import__('math')})
                    points.append(float(result))
                except Exception as e:
                    points.append(None)
                    print(f"Error evaluating equation at row {idx}: {e}")
            
            return points
            
        # Calculate next day movement
        def calculate_movement(data):
            movements = []
            for i in range(len(data) - 1):  # Exclude the last row
                next_open = data.iloc[i + 1]["Open"]
                next_close = data.iloc[i + 1]["Close"]
                movement = "+" if (next_close - next_open) > 0 else "-"
                movements.append(movement)
            
            movements.append(None)  # Last row has no movement data
            return movements
            
        # Calculate prediction accuracy
        def calculate_accuracy(points, movements):
            correct_predictions = 0
            total_valid = 0
            
            for i in range(len(points) - 1):  # Exclude last point as it has no movement data
                if points[i] is not None and movements[i] is not None:
                    predicted_sign = "+" if points[i] > 0 else "-"
                    if predicted_sign == movements[i]:
                        correct_predictions += 1
                    total_valid += 1
            
            accuracy = (correct_predictions / total_valid) * 100 if total_valid > 0 else 0
            return accuracy
        
        # Evaluate equation
        points = evaluate_equation_points(data, equation)
        
        # Calculate movements
        movements = calculate_movement(data)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(points, movements)
        
        # Prepare result
        result_data = []
        dates = data.index.strftime('%Y-%m-%d').tolist()
        
        for i in range(len(dates)):
            point_value = None if points[i] is None else float(points[i])
            
            result_data.append({
                "date": dates[i],
                "point": point_value,
                "movement": movements[i]
            })
        
        return jsonify({
            "results": result_data,
            "accuracy": round(accuracy, 2),
            "columns": sorted(data.reset_index().columns.tolist())
        })
        
    except Exception as e:
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
        
        # Add fundamental data
        fundamentals = stock.info
        
        # Return sorted list of columns
        columns = sorted(data.reset_index().columns.tolist())
        
        return jsonify({
            "columns": columns,
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
    ticker = request.args.get('ticker', '').upper()
    quantity = int(request.args.get('quantity', 0))
    trade_type = request.args.get('type', '').lower()

    if not ticker or quantity <= 0 or trade_type not in ['buy', 'sell']:
        return jsonify({"error": "Invalid trade parameters"}), 400

    stock = yf.Ticker(ticker)
    current_price = round(stock.history(period="1d", interval="1m").iloc[-1]['Close'], 2)
    total_cost = round(quantity * current_price, 2)

    df = pd.read_csv(USER_DATA_FILE)

    if trade_type == "buy":
        balance = get_balance()
        if total_cost > balance:
            return jsonify({"error": "Insufficient funds"}), 400

        new_trade = pd.DataFrame({
            "Ticker": [ticker], 
            "Date Bought": [datetime.now().strftime('%Y-%m-%d %H:%M')],
            "Quantity": [quantity], 
            "Price Bought": [current_price], 
            "Current Value": [total_cost],
            "Earnings": [0], 
            "Change %": [0], 
            "Status": ["OWNED"]
        })
        df = pd.concat([df, new_trade], ignore_index=True)
        new_balance = update_balance(-total_cost)
        
        # Log the buy transaction
        log_transaction("BUY", ticker, quantity, current_price, total_cost, new_balance)

    elif trade_type == "sell":
        stock_rows = df[(df["Ticker"] == ticker) & (df["Quantity"] > 0)]

        if stock_rows.empty:
            return jsonify({"error": "No shares available to sell"}), 400

        stock_row = stock_rows.iloc[0]  
        buy_price = stock_row["Price Bought"]
        profit = (current_price - buy_price) * quantity

        # Save sale to sold.csv
        sold_df = pd.read_csv(SOLD_DATA_FILE)
        new_sale = pd.DataFrame({
            "Ticker": [ticker], 
            "Date Bought": [stock_row["Date Bought"]],
            "Quantity": [quantity], 
            "Price Bought": [buy_price], 
            "Sell Price": [current_price],
            "Profit": [profit],
            "Sell Date": [datetime.now().strftime('%Y-%m-%d %H:%M')]
        })
        sold_df = pd.concat([sold_df, new_sale], ignore_index=True)
        sold_df.to_csv(SOLD_DATA_FILE, index=False)

        # Remove sold stocks from users.csv
        df = df[df["Ticker"] != ticker]
        new_balance = update_balance(total_cost)
        
        # Log the sell transaction
        log_transaction("SELL", ticker, quantity, current_price, total_cost, new_balance)

    df.to_csv(USER_DATA_FILE, index=False)
    return jsonify({"success": f"{trade_type.capitalize()} {quantity} shares of {ticker} at ${current_price:.2f}", "balance": get_balance()})

@app.route('/reset', methods=['GET'])
def reset_account():
    df = pd.DataFrame(columns=["Ticker", "Date Bought", "Quantity", "Price Bought", "Current Value", "Earnings", "Change %", "Status"])
    df.to_csv(USER_DATA_FILE, index=False)
    
    # Reset balance
    with open(BALANCE_FILE, "w") as f:
        f.write(str(STARTING_BALANCE))
    
    # Add a reset entry to the transaction log
    if os.path.exists(TRANSACTION_FILE):
        transactions_df = pd.read_csv(TRANSACTION_FILE)
        
        # Create reset transaction entry
        reset_transaction = pd.DataFrame({
            "Date": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            "Type": ["RESET"],
            "Ticker": ["SYSTEM"],
            "Quantity": [0],
            "Price": [0],
            "Total": [0],
            "Balance After": [STARTING_BALANCE]
        })
        
        # Add to transaction log
        transactions_df = pd.concat([transactions_df, reset_transaction], ignore_index=True)
        transactions_df.to_csv(TRANSACTION_FILE, index=False)
    
    return jsonify({"success": "Account reset successfully", "balance": STARTING_BALANCE})

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)