import importlib
import logging
import time
from src.config import LOG_FORMAT

# Configure root logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("AlphaTechMain")


def run_all():
    """
    Executes the entire quantitative investment pipeline end-to-end.
    """
    start_time = time.time()
    logger.info("=========================================================")
    logger.info("     ALPHATECH QUANTITATIVE PIPELINE - MASTER RUNNER      ")
    logger.info("=========================================================")

    # Dynamic imports for quantitative pipeline modules
    data_ingestion = importlib.import_module("src.01_data_ingestion")
    feature_eng = importlib.import_module("src.02_feature_eng")
    model_inference = importlib.import_module("src.03_model_inference")
    morning_brief = importlib.import_module("src.morning_brief")

    # Step 1: Data Ingestion
    logger.info("[Step 1/4] Ingesting Market OHLCV, Macro & Financial Data...")
    data_ingestion.run_ingestion_pipeline()

    # Step 2: Feature Engineering
    logger.info("[Step 2/4] Extracting Technical, GARCH Volatility & Alpha Features...")
    feature_eng.run_feature_engineering_pipeline()

    # Step 3: Model Training & Inference
    logger.info("[Step 3/4] Training Multi-Engine Ensemble & Optimizing Portfolio...")
    metrics, recommendations = model_inference.run_model_inference_pipeline()

    # Step 4: Morning Market Brief
    logger.info("[Step 4/4] Generating Institutional Daily Morning Brief via Gemini AI...")
    try:
        morning_brief.run_morning_brief_pipeline()
    except Exception as e:
        logger.warning(f"Morning Brief generation encountered an issue (non-fatal): {e}")

    elapsed = round(time.time() - start_time, 2)
    logger.info("=========================================================")
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed} SECONDS!")
    logger.info(f"GBDT ROC-AUC Score: {metrics.get('GBDT_ROC_AUC', metrics.get('ROC_AUC'))} | Precision@Top5: {metrics.get('Precision_Top5')}")
    logger.info("=========================================================")
    print("\n--- LATEST ALPHA RECOMMENDATIONS ---")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    run_all()
