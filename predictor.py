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

"""線形回帰モデルの追加スクリプト"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# パスの設定
MODEL_DIR = Path('saved_models')
MODEL_DIR.mkdir(exist_ok=True)

# データの読み込み（あなたのデータセットに合わせて修正してください）
# この部分は実際のデータファイルに置き換えてください
print("データを読み込んでいます...")
# df = pd.read_csv('your_training_data.csv')  # 実際のファイル名に変更
# 以下はサンプルコード（実際のデータに置き換えてください）

# 特徴量とターゲットの設定
feature_columns = ['年齢', '性別', 'K（AVG）', 'AL', 'LT', 'ACD']
# X = df[feature_columns]
# y = df['SE_p']  # ターゲット変数名を確認してください

print("線形回帰モデルを学習しています...")

# スケーラーの作成と適用
scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# モデルの学習
model = LinearRegression()
# model.fit(X_scaled, y)

# クロスバリデーションでの性能評価
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# scoring = {
#     'r2': 'r2',
#     'neg_mse': 'neg_mean_squared_error',
#     'neg_mae': 'neg_mean_absolute_error'
# }
# 
# cv_results = cross_validate(
#     model, X_scaled, y,
#     cv=cv,
#     scoring=scoring,
#     return_train_score=False,
#     n_jobs=-1
# )

# 性能指標の計算
# r2_scores = cv_results['test_r2']
# mse_scores = -cv_results['test_neg_mse']
# mae_scores = -cv_results['test_neg_mae']
# rmse_scores = np.sqrt(mse_scores)

# サンプル値（実際のクロスバリデーション結果に置き換えてください）
r2_mean = 0.8500  # 実際の値に置き換え
r2_std = 0.0200
rmse_mean = 1.0500
rmse_std = 0.0500
mae_mean = 0.8200
mae_std = 0.0400

print(f"\n=== 線形回帰モデルの性能 ===")
print(f"R² Score: {r2_mean:.4f} ± {r2_std:.4f}")
print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")

# モデルとスケーラーの保存
print("\nモデルを保存しています...")
joblib.dump(model, MODEL_DIR / 'LinearRegression.pkl')
joblib.dump(scaler, MODEL_DIR / 'LinearRegression_scaler.pkl')

# metadata.jsonの更新
metadata_path = MODEL_DIR / 'metadata.json'
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 線形回帰の情報を追加
metadata['models']['LinearRegression'] = {
    'model_file': 'LinearRegression.pkl',
    'scaler_file': 'LinearRegression_scaler.pkl',
    'needs_scaling': True,
    'performance': {
        'outer_r2_mean': r2_mean,
        'outer_r2_std': r2_std,
        'outer_rmse_mean': rmse_mean,
        'outer_rmse_std': rmse_std,
        'outer_mae_mean': mae_mean,
        'outer_mae_std': mae_std
    },
    'description': '線形回帰モデル（解釈性が高い）'
}

# metadata.jsonを保存
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ 線形回帰モデルの追加が完了しました！")
print(f"\n保存されたファイル:")
print(f"- {MODEL_DIR / 'LinearRegression.pkl'}")
print(f"- {MODEL_DIR / 'LinearRegression_scaler.pkl'}")
print(f"- {metadata_path} (更新)")

# 係数の表示（オプション）
# print("\n=== 線形回帰の係数 ===")
# for feature, coef in zip(feature_columns, model.coef_):
#     print(f"{feature}: {coef:.4f}")
# print(f"切片: {model.intercept_:.4f}")
