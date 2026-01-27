from pathlib import Path
import os
import pandas as pd

class BatchDataframe:
    """Load and store Dataframes in batches for more efficient RAM usage.
       Suitable for datetime as index Dataframes.
    """
    
    def __init__(
        self,
        dir_path: str,
        batch_size: str = '1D',
    ):
        self.dir_path = dir_path
        self.batch_size = batch_size
    
    def last_datetime(self) -> pd.Timestamp:
        """Get the last datetime stored in the batches.
        
        Returns:
            pd.Timestamp: The last datetime in the stored batches.
        """
        batch_files = sorted(Path(self.dir_path).glob('batch_*.parquet'))
        if not batch_files:
            return None
        
        last_batch_file = batch_files[-1]
        last_batch_df = pd.read_parquet(last_batch_file)
        
        if len(last_batch_df) > 0:
            return last_batch_df.index.max()
        else:
            return None
    
    def load_dataframe(
        self,
        from_datetime: pd.Timestamp,
        to_datetime: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load DataFrame from disk in batches based on datetime range.
        
        Args:
            from_datetime (pd.Timestamp): Start datetime for loading data.
            to_datetime (pd.Timestamp): End datetime for loading data.
        
        Returns:
            pd.DataFrame: Loaded DataFrame within the specified datetime range.
        """
        dataframes = []
        
        # List all batch files in the directory
        batch_files = sorted(Path(self.dir_path).glob('batch_*.parquet'))
        
        for batch_file in batch_files:
            try:
                batch_df = pd.read_parquet(batch_file)
                # Check if this batch has any data within the datetime range
                if len(batch_df) > 0:
                    batch_min = batch_df.index.min()
                    batch_max = batch_df.index.max()
                    # If batch overlaps with requested range, include it
                    if batch_max >= from_datetime and batch_min <= to_datetime:
                        dataframes.append(batch_df)
            except Exception:
                pass
        
        if dataframes:
            df = pd.concat(dataframes).sort_index()
            mask = (df.index >= from_datetime) & (df.index <= to_datetime)
            
            return df.loc[mask]
        else:
            return pd.DataFrame()
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
    ):
        """Save DataFrame in batches to disk.
        
        Args:
            df (pd.DataFrame): DataFrame to be saved.
        """
        df = df.sort_index()
        grouped = df.groupby(pd.Grouper(freq=self.batch_size))
        
        for group_time, group_df in grouped:
            batch_filename = f"{self.dir_path}/batch_{group_time.strftime('%Y%m%d%H%M%S')}.parquet"
            if not os.path.exists(self.dir_path):
                os.makedirs(self.dir_path)
            
            group_df.to_parquet(batch_filename)
