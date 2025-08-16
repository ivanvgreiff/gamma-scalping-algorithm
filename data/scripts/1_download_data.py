from data_downloader import Downloader, configure_logging
from datetime import datetime, timedelta

def main():
    configure_logging('INFO')
    start_date = '2019-03-30'
    end_date = '2025-07-20'
    
    # Download all BTC options in one call
    print(f"\n=== Downloading BTC options from {start_date} to {end_date} ===")
    options_downloader = Downloader(
        base_path="./data/raw/options/",
        max_concurrent_requests=10,
        max_retries=3,
    )
    options_downloader.download(
        exchange='deribit',
        symbol='BTC',
        start_date=start_date,
        end_date=end_date,
        format='feather',
        kind='option',
        skip_existing=True
    )
    
    # Download all BTC futures in one call
    #print(f"\n=== Downloading BTC futures from {start_date} to {end_date} ===")
    #futures_downloader = Downloader(
    #    base_path="./downloads/futures/",
    #    max_concurrent_requests=10,
    #    max_retries=3,
    #)
    #futures_downloader.download(
    #    exchange='deribit',
    #    symbol='BTC',
    #    start_date=start_date,
    #    end_date=end_date,
    #    format='feather',
    #    kind='future'
    #)
    
    spot_downloader = Downloader(
        base_path="./data/raw/spot/",
        max_concurrent_requests=10,
        max_retries=3,
    )
    spot_downloader.download(
        exchange='binance',
        symbol='BTCUSDT',
        start_date=start_date,
        end_date=end_date,
        format='feather',
        data_type='klines',
        interval='1h',
        skip_existing=True
    )

    print("\n=== Download complete! ===")


if __name__ == '__main__':
    main()