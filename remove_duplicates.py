import pandas as pd


def remove_duplicates_from_csv():
    # File path for input and output
    file_path = "users_export.csv"
    output_file = "unduplicated_users.csv"

    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if 'did' column exists
        if 'did' not in df.columns:
            print("Error: Column 'did' not found in the CSV file.")
            return

        # Remove duplicate rows based on 'did'
        df_cleaned = df.drop_duplicates(subset=['did'])

        # Display the cleaned DataFrame
        print("Cleaned DataFrame:")
        print(df_cleaned)

        # Save the cleaned data
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    remove_duplicates_from_csv()
