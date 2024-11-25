#%%
import pandas as pd
from deep_translator import GoogleTranslator

def translate_dataset(df, columns_to_translate):
    """
    Translate specified columns from English to Spanish
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    columns_to_translate (list): List of column names to translate
    
    Returns:
    pandas.DataFrame: DataFrame with translated columns
    """
    # Create a copy of the dataframe
    df_translated = df.copy()
    
    # Initialize translator
    translator = GoogleTranslator(source='en', target='es')
    
    # Counter for translation progress
    total_items = len(df) * len(columns_to_translate)
    current_item = 0
    
    # Iterate through each column that needs translation
    for column in columns_to_translate:
        print(f"Traduciendo columna: {column}")
        
        # Iterate through each row in the column
        for idx, text in enumerate(df_translated[column]):
            try:
                # Skip empty or non-string values
                if pd.isna(text) or not isinstance(text, str):
                    continue
                
                # Translate text to Spanish
                translated_text = translator.translate(text)
                df_translated.at[idx, column] = translated_text
                
                # Update progress
                current_item += 1
                if idx % 100 == 0:  # Show progress every 100 items
                    progress = (current_item / total_items) * 100
                    print(f"Progreso: {progress:.1f}%")
                
                
            except Exception as e:
                print(f"Error translating row {idx} in column {column}: {str(e)}")
                continue
    
    return df_translated

# Example usage
if __name__ == "__main__":
    df_2 = pd.read_csv('../data/raw/GSM8K-Socratic.csv')

    # Specify which columns to translate
    columns_to_translate_df2 = ['question','answer']

    # Perform translation
    translated_df_2 = translate_dataset(df_2, columns_to_translate_df2)

    translated_df_2.to_csv('../data/processed/GSM8K-Socratic_spanish.csv', index=False)

    print("Traduccion completa!")