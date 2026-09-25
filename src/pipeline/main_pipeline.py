import streamlit as st
from src.data.stock_data import fetch_stock_data
from src.features.feature_engineering import prepare_ml_data
from src.models.model_factory import get_models
from src.models.trainer import split_data_chronologically, train_models
from src.models.evaluator import evaluate_models
from src.models.selector import select_best_model
from src.prediction.predictor import extract_latest_features_for_prediction, predict_next_day

def execute_prediction_pipeline(ticker: str):
    """
    Executes the entire end-to-end ML pipeline and stores the results globally in st.session_state.
    This prevents duplicate execution when navigating between pages.
    """
    with st.spinner("Analyzing market data and generating prediction..."):
        df_raw, stock_info = fetch_stock_data(ticker)
        X, y = prepare_ml_data(df_raw)
        X_train, X_test, y_train, y_test = split_data_chronologically(X, y)
        models = get_models()
        trained_pipeline = train_models(models, X_train, y_train)
        evaluation_results = evaluate_models(trained_pipeline, X_test, y_test)
        best_model_name = select_best_model(evaluation_results)
        latest_features, latest_close = extract_latest_features_for_prediction(df_raw)
        prediction = predict_next_day(trained_pipeline[best_model_name], latest_features, latest_close, stock_info.native_currency)
        
        # Store all results globally
        st.session_state['df_raw'] = df_raw
        st.session_state['stock_info'] = stock_info
        st.session_state['evaluation_results'] = evaluation_results
        st.session_state['trained_pipeline'] = trained_pipeline
        st.session_state['best_model_name'] = best_model_name
        st.session_state['prediction'] = prediction
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['run_pipeline'] = True
        st.session_state['ticker'] = ticker
