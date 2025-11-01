"""線形回帰モデルの学習と保存"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.preprocessing import StandardScaler

def train_and_save_linear_regression(data_file='training_data.csv'):
    """
    線形回帰モデルを学習して保存
    
    Parameters:
    -----------
    data_file : str
        学習データのCSVファイルパス
    """
    
    # パスの設定
    MODEL_DIR = Path('saved_models')
    MODEL_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("線形回帰モデルの学習")
    print("=" * 60)
    
    # データの読み込み
    print(f"\n📂 データを読み込んでいます: {data_file}")
    try:
        df = pd.read_csv(data_file)
        print(f"✅ {len(df)} 件のデータを読み込みました")
    except FileNotFoundError:
        print(f"❌ エラー: {data_file} が見つかりません")
        print("\n使用方法:")
        print("  train_and_save_linear_regression('your_data.csv')")
        return
    
    # 特徴量とターゲットの設定
    feature_columns = ['年齢', '性別', 'K（AVG）', 'AL', 'LT', 'ACD']
    target_column = 'SE_p'
    
    # 列名の確認
    missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing_cols:
        print(f"❌ エラー: 以下の列が見つかりません: {missing_cols}")
        print(f"利用可能な列: {df.columns.tolist()}")
        return
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"\n📊 特徴量: {feature_columns}")
    print(f"📊 ターゲット: {target_column}")
    
    # スケーラーの作成と適用
    print("\n⚙️ データの標準化を実行中...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # モデルの学習
    print("🔧 線形回帰モデルを学習中...")
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # クロスバリデーションでの性能評価
    print("📈 クロスバリデーションで性能を評価中...")
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    
    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    
    cv_results = cross_validate(
        model, X_scaled, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    # 性能指標の計算
    r2_scores = cv_results['test_r2']
    mse_scores = -cv_results['test_neg_mse']
    mae_scores = -cv_results['test_neg_mae']
    rmse_scores = np.sqrt(mse_scores)
    
    r2_mean = float(np.mean(r2_scores))
    r2_std = float(np.std(r2_scores))
    rmse_mean = float(np.mean(rmse_scores))
    rmse_std = float(np.std(rmse_scores))
    mae_mean = float(np.mean(mae_scores))
    mae_std = float(np.std(mae_scores))
    
    print("\n" + "=" * 60)
    print("📊 線形回帰モデルの性能")
    print("=" * 60)
    print(f"R² Score: {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"RMSE:     {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"MAE:      {mae_mean:.4f} ± {mae_std:.4f}")
    
    # モデルとスケーラーの保存
    print("\n💾 モデルを保存しています...")
    joblib.dump(model, MODEL_DIR / 'LinearRegression.pkl')
    joblib.dump(scaler, MODEL_DIR / 'LinearRegression_scaler.pkl')
    
    # metadata.jsonの更新
    metadata_path = MODEL_DIR / 'metadata.json'
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("⚠️ metadata.json が見つかりません。新規作成します。")
        metadata = {
            'feature_columns': feature_columns,
            'models': {}
        }
    
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
    
    print("\n✅ 線形回帰モデルの追加が完了しました！")
    print(f"\n保存されたファイル:")
    print(f"  📄 {MODEL_DIR / 'LinearRegression.pkl'}")
    print(f"  📄 {MODEL_DIR / 'LinearRegression_scaler.pkl'}")
    print(f"  📄 {metadata_path} (更新)")
    
    # 係数の表示
    print("\n" + "=" * 60)
    print("🔍 線形回帰の係数")
    print("=" * 60)
    for feature, coef in zip(feature_columns, model.coef_):
        print(f"  {feature:12s}: {coef:>8.4f}")
    print(f"  {'切片':12s}: {model.intercept_:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # 使用例
    import sys
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # デフォルトのファイル名
        data_file = 'training_data.csv'
        print(f"使用方法: python train_linear_regression.py <データファイル>")
        print(f"デフォルトで '{data_file}' を使用します\n")
    
    train_and_save_linear_regression(data_file)
