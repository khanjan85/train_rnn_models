import os

def fetch_database():
    """
    Clone the stocks_data_management repository and fetch the database.
    """
    db_repo = "https://github.com/chiragpalan/stocks_data_management.git"
    db_name = "nifty50_data_v1.db"
    local_db_path = os.path.join(os.getcwd(), db_name)

    if not os.path.exists(local_db_path):
        print("Fetching database from stocks_data_management repository...")
        os.system(f"git clone {db_repo} temp_repo")
        os.rename(f"temp_repo/{db_name}", local_db_path)
        os.system("rm -rf temp_repo")
    else:
        print("Database already exists locally.")

if __name__ == "__main__":
    fetch_database()
