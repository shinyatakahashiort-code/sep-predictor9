"""SE_p予測モデル用ユーティリティクラス"""
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SEPredictor:
    """SE_p予測のための統一インターフェース"""
    
    def __init__(self, model_name='MLP', model_dir='saved_models'):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        
        # メタデータの読み込み
        with open(self.model_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # データ統計の読み込み
        with open(self.model_dir / 'data_stats.json', 'r', encoding='utf-8') as f:
            self.data_stats = json.load(f)
        
        # モデル情報の取得
        self.model_info = self.metadata['models'][model_name]
        self.feature_columns = self.metadata['feature_columns']
        
        # モデルの読み込み
        model_file = self.model_info['model_file']
        self.model = joblib.load(self.model_dir / model_file)
        
        # スケーラーの読み込み
        self.scaler = None
        if self.model_info['needs_scaling'] and self.model_info['scaler_file']:
            scaler_file = self.model_info['scaler_file']
            self.scaler = joblib.load(self.model_dir / scaler_file)
    
    def predict(self, input_data):
        """単純な予測"""
        X = pd.DataFrame([input_data])[self.feature_columns]
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return float(self.model.predict(X)[0])
    
    def predict_with_details(self, input_data):
        """詳細情報付きの予測"""
        # 入力検証
        validation = self.validate_input(input_data)
        
        # 予測実行
        prediction = self.predict(input_data)
        
        # 性能指標の取得
        perf = self.model_info['performance']
        rmse = perf['outer_rmse_mean']
        mae = perf['outer_mae_mean']
        
        # 結果の構築
        result = {
            'prediction': round(prediction, 4),
            'confidence_interval_95': {
                'lower': round(prediction - 1.96 * rmse, 4),
                'upper': round(prediction + 1.96 * rmse, 4)
            },
            'expected_error': {
                'mae': round(mae, 4),
                'rmse': round(rmse, 4)
            },
            'model_performance': {
                'r2_mean': round(perf['outer_r2_mean'], 4),
                'r2_std': round(perf['outer_r2_std'], 4)
            },
            'validation': validation,
            'model_name': self.model_name
        }
        
        return result
    
    def validate_input(self, input_data):
        """入力データの検証"""
        result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # 必須フィールドのチェック
        for col in self.feature_columns:
            if col not in input_data:
                result['is_valid'] = False
                result['errors'].append(f"必須フィールド '{col}' がありません")
        
        if not result['is_valid']:
            return result
        
        # 値の範囲チェック
        for col in self.feature_columns:
            value = input_data[col]
            stats = self.data_stats['features'][col]
            min_val = stats['min']
            max_val = stats['max']
            
            if value < min_val * 0.5 or value > max_val * 1.5:
                result['warnings'].append(
                    f"{col}={value} は通常範囲外です (通常: {min_val:.2f} - {max_val:.2f})"
                )
        
        return result


class ModelEnsemble:
    """複数モデルのアンサンブル予測"""
    
    def __init__(self, model_names=None, model_dir='saved_models'):
        # メタデータの読み込み
        metadata_path = Path(model_dir) / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 利用可能なモデルの取得
        available_models = list(metadata['models'].keys())
        
        if model_names is None:
            model_names = available_models
        
        # 各モデルの予測器を初期化
        self.predictors = {}
        for name in model_names:
            if name in available_models:
                self.predictors[name] = SEPredictor(model_name=name, model_dir=model_dir)
        
        # 性能に基づく重みの計算
        r2_scores = {
            name: predictor.model_info['performance']['outer_r2_mean']
            for name, predictor in self.predictors.items()
        }
        total_r2 = sum(r2_scores.values())
        self.weights = {
            name: r2 / total_r2
            for name, r2 in r2_scores.items()
        }
    
    def predict(self, input_data):
        """加重平均によるアンサンブル予測"""
        predictions = {}
        for name, predictor in self.predictors.items():
            predictions[name] = predictor.predict(input_data)
        
        ensemble_prediction = sum(
            predictions[name] * self.weights[name]
            for name in predictions
        )
        
        return float(ensemble_prediction)
    
    def predict_with_details(self, input_data):
        """詳細情報付きアンサンブル予測"""
        # 各モデルで予測
        individual_predictions = {}
        individual_details = {}
        
        for name, predictor in self.predictors.items():
            individual_predictions[name] = predictor.predict(input_data)
            individual_details[name] = predictor.predict_with_details(input_data)
        
        # アンサンブル予測値
        ensemble_prediction = sum(
            individual_predictions[name] * self.weights[name]
            for name in individual_predictions
        )
        
        # 予測のばらつきを計算
        predictions_array = np.array(list(individual_predictions.values()))
        ensemble_std = np.std(predictions_array)
        
        # 結果の構築
        result = {
            'prediction': round(ensemble_prediction, 4),
            'ensemble_std': round(ensemble_std, 4),
            'confidence_interval_95': {
                'lower': round(ensemble_prediction - 1.96 * ensemble_std, 4),
                'upper': round(ensemble_prediction + 1.96 * ensemble_std, 4)
            },
            'individual_predictions': individual_predictions,
            'individual_details': individual_details,
            'weights': self.weights,
            'models_used': list(self.predictors.keys()),
            'validation': individual_details[list(individual_details.keys())[0]]['validation']
        }
        
        return result
