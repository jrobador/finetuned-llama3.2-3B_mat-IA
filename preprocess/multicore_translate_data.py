#%%
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def chunk_dataframe(df, chunk_size=1000):
    """Split dataframe into chunks"""
    return np.array_split(df, len(df) // chunk_size + 1)

def translate_texts(texts, source_lang='en', target_lang='es'):
    """Translate a batch of texts, retrying on failure."""
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translated_texts = []

    for text in texts:
        try:
            if pd.isna(text) or not isinstance(text, str):
                translated_texts.append(text)
            else:
                translated_texts.append(translator.translate(text))
        except Exception:
            translated_texts.append(text)  # Keep original text on error
    
    return translated_texts

def translate_chunk(chunk_data):
    """Process a chunk of the dataframe"""
    chunk, columns_to_translate, chunk_id = chunk_data

    for column in columns_to_translate:
        chunk[column] = translate_texts(chunk[column].tolist())
    
    return chunk_id, chunk

def translate_dataset(df, columns_to_translate, chunk_size=1000, num_processes=None):
    """
    Translate large dataset with multiprocessing and checkpoints
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    columns_to_translate (list): List of column names to translate
    chunk_size (int): Number of rows per chunk
    num_processes (int): Number of parallel processes to use
    
    Returns:
    pandas.DataFrame: DataFrame with translated columns
    """

    print(f"Using {num_processes} processes")

    # Split dataframe into chunks
    chunks = chunk_dataframe(df, chunk_size)
    total_chunks = len(chunks)
    chunk_args = [(chunk, columns_to_translate, i) for i, chunk in enumerate(chunks)]

    # Translate chunks in parallel with progress bar
    translated_chunks = []
    with Pool(processes=num_processes) as pool:
        for chunk_id, translated_chunk in tqdm(pool.imap_unordered(translate_chunk, chunk_args), total=total_chunks, desc="Processing chunks"):
            translated_chunks.append(translated_chunk)

    # Combine all chunks
    translated_df = pd.concat(translated_chunks, axis=0)
    translated_df.reset_index(drop=True, inplace=True)
    
    return translated_df

# Example usage
if __name__ == "__main__":
    try:
        df = pd.read_csv('../data/raw/MathInstruct.csv')
        
        columns_to_translate = ['instruction', 'output']
        
        optimal_chunk_size = 50
        optimal_processes = (cpu_count() - 1)

        print(f"Starting translation of {len(df)} rows...")

        start_time = time.time()
        translated_df = translate_dataset(
            df, 
            columns_to_translate,
            chunk_size=optimal_chunk_size,
            num_processes=optimal_processes
        )
        translated_df.to_csv('MathInstruct_spanish.csv', index=False)

        end_time = time.time()
        print(f"\nTranslation completed in {(end_time - start_time) / 60:.2f} minutes")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

#%%
