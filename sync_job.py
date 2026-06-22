import time
import logging
import argparse
from brain import builder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sync_job")

def run_sync_loop(interval_minutes: int, full_reindex_first: bool = False):
    logger.info(f"Starting scheduled sync every {interval_minutes} minutes.")
    
    if full_reindex_first:
        logger.info("Running initial full sync as requested...")
        builder(full_reindex=True)
    
    while True:
        try:
            logger.info("Waking up to run incremental sync...")
            builder(full_reindex=False)
            logger.info("Incremental sync complete.")
        except Exception as e:
            logger.error(f"Error during scheduled sync: {e}", exc_info=True)
            
        logger.info(f"Sleeping for {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Interval in minutes between syncs")
    parser.add_argument("--full", action="store_true", help="Do a full sync on the first run")
    args = parser.parse_args()
    
    run_sync_loop(interval_minutes=args.interval, full_reindex_first=args.full)
