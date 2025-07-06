import backtrader as bt
import pandas as pd

class PandasDataFeed(bt.feeds.PandasData):
    """Custom Pandas Data Feed for backtrader"""
    
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
    )
    
    def __init__(self):
        super().__init__()
        
    @classmethod
    def create_from_dataframe(cls, df):
        """Create data feed from pandas DataFrame"""
        # Ensure the DataFrame has the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Create the data feed
        return cls(dataname=df)