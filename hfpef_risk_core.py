from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


TARGET_COLUMN = 'Event'
MODEL_RELATIVE_PATH = Path('HFpEF_saved_models/Random_Forest_best_model.pkl')
SCALER_RELATIVE_PATH = Path('HFpEF_saved_models/HFpEF_standard_scaler.pkl')
RUNTIME_BUNDLE_RELATIVE_PATH = Path('HFpEF_runtime_bundle.json')
SUMMARY_RELATIVE_PATH = Path('HFpEF_run_summary.json')
BATTERY_PYTHON_PATH = Path(r'C:\Users\ARCC\.conda\envs\battery\python.exe')


@dataclass(frozen=True)
class RuntimeAssets:
    project_dir: Path
    model: Any
    scaler: Any
    model_features: list[str]
    scaler_features: list[str]
    feature_specs: dict[str, dict[str, Any]]
    default_input_values: dict[str, float]
    reference_probabilities: np.ndarray
    lime_background_df: pd.DataFrame
    threshold: float


@dataclass(frozen=True)
class RuntimeExplainers:
    tree_explainer: Any
    lime_explainer: Any
    lime_predict_fn: Any


def load_runtime_assets(project_dir: str | Path) -> RuntimeAssets:
    base_dir = Path(project_dir).resolve()
    model = joblib.load(base_dir / MODEL_RELATIVE_PATH)
    scaler = joblib.load(base_dir / SCALER_RELATIVE_PATH)

    try:
        params = model.get_params(deep=False)
        if 'n_jobs' in params:
            model.set_params(n_jobs=1)
    except Exception:
        pass

    runtime_bundle = json.loads((base_dir / RUNTIME_BUNDLE_RELATIVE_PATH).read_text(encoding='utf-8'))

    model_features = list(getattr(model, 'feature_names_in_', []))
    scaler_features = list(getattr(scaler, 'feature_names_in_', []))
    if not model_features:
        raise ValueError('随机森林模型缺少 feature_names_in_，无法保证特征顺序。')
    if not scaler_features:
        raise ValueError('标准化器缺少 feature_names_in_，无法保证列映射。')
    if runtime_bundle.get('model_features') != model_features:
        raise ValueError('运行时数据包中的特征顺序与模型不一致。')
    lime_background = runtime_bundle.get('lime_background') or [runtime_bundle['default_input_values']]
    lime_background_df = pd.DataFrame(lime_background)
    lime_background_df = lime_background_df.loc[:, model_features].apply(pd.to_numeric, errors='coerce').astype(float)

    return RuntimeAssets(
        project_dir=base_dir,
        model=model,
        scaler=scaler,
        model_features=model_features,
        scaler_features=scaler_features,
        feature_specs=_normalize_runtime_feature_specs(runtime_bundle['feature_specs']),
        default_input_values={feature: float(value) for feature, value in runtime_bundle['default_input_values'].items()},
        reference_probabilities=np.asarray(runtime_bundle['reference_probabilities'], dtype=float),
        lime_background_df=lime_background_df,
        threshold=float(runtime_bundle.get('threshold', 0.5)),
    )


def _normalize_runtime_feature_specs(
    raw_specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for feature, spec in raw_specs.items():
        data_min = float(spec.get('data_min', spec.get('min', 0.0)))
        data_max = float(spec.get('data_max', spec.get('max', data_min)))
        slider_min = float(spec.get('slider_min', data_min))
        slider_max = float(spec.get('slider_max', data_max))
        default = float(spec.get('default', spec.get('median', data_min)))
        if slider_min > slider_max:
            slider_min, slider_max = data_min, data_max
        specs[feature] = {
            **spec,
            'data_min': data_min,
            'data_max': data_max,
            'slider_min': slider_min,
            'slider_max': slider_max,
            'default': default,
            'step': float(spec.get('step', max((slider_max - slider_min) / 200.0, 0.01))),
            'is_categorical': bool(spec.get('is_categorical', False)),
            'options': spec.get('options'),
        }
    return specs


def get_explainer_runtime_status() -> dict[str, Any]:
    missing_modules: list[str] = []
    sklearn_version = 'unknown'

    try:
        import shap  # noqa: F401
    except ModuleNotFoundError:
        missing_modules.append('shap')

    try:
        import lime.lime_tabular  # noqa: F401
    except ModuleNotFoundError:
        missing_modules.append('lime')

    try:
        import sklearn

        sklearn_version = getattr(sklearn, '__version__', 'unknown')
    except Exception:
        missing_modules.append('scikit-learn')

    python_executable = str(Path(sys.executable))
    is_battery_python = Path(sys.executable).as_posix().lower() == BATTERY_PYTHON_PATH.as_posix().lower()
    sklearn_compatible = sklearn_version.startswith('1.7.')
    is_ready = sklearn_compatible and not missing_modules

    if is_ready:
        message = '运行环境已满足当前样本 SHAP / LIME 实时解释要求。'
    else:
        message_parts = [
            '当前页面缺少 SHAP / LIME 依赖，或 scikit-learn 版本与模型文件不匹配。',
            f'当前 Python: {python_executable}',
            f'scikit-learn: {sklearn_version}',
        ]
        if missing_modules:
            message_parts.append(f'缺少依赖: {", ".join(missing_modules)}')
        if not sklearn_compatible:
            message_parts.append('当前环境的 scikit-learn 版本与模型文件不匹配，建议使用 1.7.x。')
        if not is_battery_python:
            message_parts.append(f'本地开发建议环境: {BATTERY_PYTHON_PATH}')
        message = ' '.join(message_parts)

    return {
        'is_ready': bool(is_ready),
        'message': message,
        'python_executable': python_executable,
        'expected_python': str(BATTERY_PYTHON_PATH),
        'is_battery_python': bool(is_battery_python),
        'sklearn_version': sklearn_version,
        'missing_modules': missing_modules,
    }


def load_runtime_explainers(
    assets: RuntimeAssets,
    display_feature_names: list[str] | None = None,
) -> RuntimeExplainers:
    status = get_explainer_runtime_status()
    if not status['is_ready']:
        raise RuntimeError(status['message'])

    import lime.lime_tabular
    import shap

    feature_names = list(display_feature_names or assets.model_features)
    train_raw = assets.lime_background_df.loc[:, assets.model_features].apply(pd.to_numeric, errors='coerce').astype(float)
    categorical_indices = [
        idx for idx, feature in enumerate(assets.model_features) if _is_categorical_like(train_raw[feature])
    ]
    categorical_names = {
        idx: [str(int(value)) for value in sorted(train_raw.iloc[:, idx].dropna().astype(int).unique().tolist())]
        for idx in categorical_indices
    }

    def lime_predict_fn(values: np.ndarray) -> np.ndarray:
        raw_df = pd.DataFrame(values, columns=assets.model_features)
        scaled_df = transform_dataset_for_model(raw_df, assets.scaler, assets.model_features)
        return np.asarray(assets.model.predict_proba(scaled_df), dtype=float)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_raw.to_numpy(dtype=float),
        feature_names=feature_names,
        class_names=['Negative', 'Positive'],
        mode='classification',
        discretize_continuous=True,
        categorical_features=categorical_indices or None,
        categorical_names=categorical_names or None,
        random_state=888,
    )

    return RuntimeExplainers(
        tree_explainer=shap.TreeExplainer(assets.model),
        lime_explainer=lime_explainer,
        lime_predict_fn=lime_predict_fn,
    )


def _is_categorical_like(series: pd.Series) -> bool:
    clean = pd.to_numeric(series, errors='coerce').dropna()
    if clean.empty:
        return False
    unique_count = clean.nunique()
    all_integer_like = np.all(np.isclose(clean.astype(float) % 1, 0.0))
    return bool(unique_count <= 10 and all_integer_like)


def _is_integer_range(series: pd.Series) -> bool:
    clean = pd.to_numeric(series, errors='coerce').dropna()
    if clean.empty:
        return False
    return bool(np.all(np.isclose(clean.astype(float) % 1, 0.0)))


def build_feature_specs(train_df: pd.DataFrame, model_features: list[str]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for feature in model_features:
        series = pd.to_numeric(train_df[feature], errors='coerce').dropna().astype(float)
        if series.empty:
            raise ValueError(f'训练集特征 {feature} 为空，无法构建页面控件。')

        default = float(np.median(series))
        f_min = float(np.min(series))
        f_max = float(np.max(series))
        q1 = float(np.percentile(series, 25))
        q3 = float(np.percentile(series, 75))

        is_categorical = _is_categorical_like(series)
        options = sorted({int(v) for v in series.tolist()}) if is_categorical else None

        slider_min = f_min
        slider_max = f_max
        if not is_categorical:
            iqr = q3 - q1
            if iqr > 0:
                bounded_min = q1 - 1.5 * iqr
                bounded_max = q3 + 1.5 * iqr
                slider_min = max(f_min, bounded_min)
                slider_max = min(f_max, bounded_max)
                if slider_min >= slider_max:
                    slider_min, slider_max = f_min, f_max

        step = 1.0 if _is_integer_range(series) else max((slider_max - slider_min) / 200.0, 0.01)

        specs[feature] = {
            'is_categorical': is_categorical,
            'options': options,
            'default': default,
            'data_min': f_min,
            'data_max': f_max,
            'slider_min': float(slider_min),
            'slider_max': float(slider_max),
            'step': float(step),
        }
    return specs


def scale_selected_features(
    raw_feature_values: Mapping[str, float],
    scaler: Any,
    model_features: list[str],
) -> pd.DataFrame:
    feature_index = {name: idx for idx, name in enumerate(scaler.feature_names_in_)}
    row: dict[str, float] = {}
    for feature in model_features:
        if feature not in raw_feature_values:
            raise KeyError(f'缺少特征输入: {feature}')
        if feature not in feature_index:
            raise KeyError(f'标准化器未找到特征: {feature}')
        idx = feature_index[feature]
        mean = float(scaler.mean_[idx])
        scale = float(scaler.scale_[idx]) if float(scaler.scale_[idx]) != 0 else 1.0
        raw_value = float(raw_feature_values[feature])
        row[feature] = (raw_value - mean) / scale

    return pd.DataFrame([row], columns=model_features)


def transform_dataset_for_model(
    df: pd.DataFrame,
    scaler: Any,
    model_features: list[str],
) -> pd.DataFrame:
    if df.empty:
        raise ValueError('输入数据集为空，无法执行标准化。')
    raw_part = df.loc[:, model_features].apply(pd.to_numeric, errors='coerce').astype(float)
    feature_index = {name: idx for idx, name in enumerate(scaler.feature_names_in_)}
    means = np.array([float(scaler.mean_[feature_index[f]]) for f in model_features], dtype=float)
    scales = np.array([float(scaler.scale_[feature_index[f]]) for f in model_features], dtype=float)
    scales = np.where(scales == 0.0, 1.0, scales)
    scaled_values = (raw_part.to_numpy(dtype=float) - means) / scales
    return pd.DataFrame(scaled_values, columns=model_features, index=raw_part.index)


def compute_risk_cutoffs(
    train_probabilities: np.ndarray,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> tuple[float, float]:
    low = float(np.quantile(train_probabilities, low_quantile))
    high = float(np.quantile(train_probabilities, high_quantile))
    if low >= high:
        low, high = 0.33, 0.67
    return low, high


def derive_runtime_cutoffs(assets: RuntimeAssets) -> tuple[float, float]:
    reference_probabilities = get_reference_probabilities(assets)
    low, high = compute_risk_cutoffs(reference_probabilities)

    if low <= 0.0:
        positive_probabilities = reference_probabilities[reference_probabilities > 0.0]
        if positive_probabilities.size >= 3:
            low, high = compute_risk_cutoffs(positive_probabilities)

    if low <= 0.0 or low >= high:
        low, high = 0.1, 0.3

    return float(low), float(high)


def get_reference_probabilities(assets: RuntimeAssets) -> np.ndarray:
    return np.asarray(assets.reference_probabilities, dtype=float)


def build_reference_feature_profile(
    train_df: pd.DataFrame,
    model_features: list[str],
) -> dict[str, float]:
    profile: dict[str, float] = {}
    for feature in model_features:
        series = pd.to_numeric(train_df[feature], errors='coerce').dropna().astype(float)
        if series.empty:
            raise ValueError(f'训练集特征 {feature} 为空，无法构建参考输入。')
        if _is_categorical_like(series):
            modes = series.mode(dropna=True)
            profile[feature] = float(modes.iloc[0]) if not modes.empty else float(np.median(series))
        else:
            profile[feature] = float(np.median(series))
    return profile


def build_local_feature_effects(
    model: Any,
    scaler: Any,
    train_df: pd.DataFrame,
    model_features: list[str],
    raw_input: Mapping[str, float],
) -> pd.DataFrame:
    actual_input = {feature: float(raw_input[feature]) for feature in model_features}
    actual_scaled = scale_selected_features(actual_input, scaler=scaler, model_features=model_features)
    actual_probability = float(model.predict_proba(actual_scaled)[0, 1])
    reference_profile = build_reference_feature_profile(train_df, model_features)

    rows: list[dict[str, float | str]] = []
    for feature in model_features:
        counterfactual_input = dict(actual_input)
        counterfactual_input[feature] = float(reference_profile[feature])
        counterfactual_scaled = scale_selected_features(
            counterfactual_input,
            scaler=scaler,
            model_features=model_features,
        )
        counterfactual_probability = float(model.predict_proba(counterfactual_scaled)[0, 1])
        effect = actual_probability - counterfactual_probability
        if effect > 1e-12:
            direction = 'increase'
        elif effect < -1e-12:
            direction = 'decrease'
        else:
            direction = 'neutral'
        rows.append(
            {
                'feature': feature,
                'current_value': float(actual_input[feature]),
                'reference_value': float(reference_profile[feature]),
                'effect': float(effect),
                'abs_effect': float(abs(effect)),
                'counterfactual_probability': float(counterfactual_probability),
                'direction': direction,
            }
        )

    return pd.DataFrame(rows).sort_values(['abs_effect', 'feature'], ascending=[False, True]).reset_index(drop=True)


def build_single_case_shap_explanation(
    runtime_explainers: RuntimeExplainers,
    model_features: list[str],
    raw_input: Mapping[str, float],
    scaled_input: pd.DataFrame,
    display_feature_names: list[str] | None = None,
) -> Any:
    import shap

    raw_values = np.array([float(raw_input[feature]) for feature in model_features], dtype=float)
    shap_output = runtime_explainers.tree_explainer(scaled_input)
    if len(np.shape(shap_output.values)) == 3:
        sample_output = shap_output[:, :, 1][0]
    else:
        sample_output = shap_output[0]

    return shap.Explanation(
        values=np.asarray(sample_output.values, dtype=float),
        base_values=float(np.asarray(sample_output.base_values).reshape(-1)[0]),
        data=raw_values,
        feature_names=list(display_feature_names or model_features),
    )


def build_single_case_shap_figures(
    runtime_explainers: RuntimeExplainers,
    model_features: list[str],
    raw_input: Mapping[str, float],
    scaled_input: pd.DataFrame,
    display_feature_names: list[str] | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import shap

    sample_explanation = build_single_case_shap_explanation(
        runtime_explainers=runtime_explainers,
        model_features=model_features,
        raw_input=raw_input,
        scaled_input=scaled_input,
        display_feature_names=display_feature_names,
    )

    plt.figure(figsize=(10.5, 5.8))
    shap.plots.waterfall(sample_explanation, show=False, max_display=min(10, len(model_features)))
    fig_waterfall = plt.gcf()
    plt.tight_layout()

    fig_force = shap.plots.force(sample_explanation, matplotlib=True, show=False)
    if not hasattr(fig_force, 'savefig'):
        fig_force = plt.gcf()
    fig_force.set_size_inches(15, 6)
    fig_force.set_facecolor('white')
    if fig_force.axes:
        fig_force.axes[0].set_facecolor('white')
        fig_force.subplots_adjust(bottom=0.28)

    return {
        'waterfall': fig_waterfall,
        'force': fig_force,
    }


def build_single_case_lime_figure(
    runtime_explainers: RuntimeExplainers,
    model_features: list[str],
    raw_input: Mapping[str, float],
    display_feature_names: list[str] | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    sample_values = np.array([float(raw_input[feature]) for feature in model_features], dtype=float)
    explanation = runtime_explainers.lime_explainer.explain_instance(
        sample_values,
        runtime_explainers.lime_predict_fn,
        num_features=min(10, len(model_features)),
    )
    figure = explanation.as_pyplot_figure()
    figure.set_size_inches(9.5, 5.6)
    plt.tight_layout()
    return figure


def compute_probability_percentile(reference_probabilities: np.ndarray, probability: float) -> float:
    if reference_probabilities.size == 0:
        raise ValueError('参考概率分布为空，无法计算百分位。')
    return float(np.mean(np.asarray(reference_probabilities, dtype=float) <= float(probability)) * 100.0)


def classify_risk(probability: float, cutoffs: tuple[float, float]) -> str:
    low, high = cutoffs
    if probability < low:
        return 'Low'
    if probability < high:
        return 'Medium'
    return 'High'


def predict_with_risk_stratification(
    model: Any,
    scaler: Any,
    model_features: list[str],
    raw_input: Mapping[str, float],
    cutoffs: tuple[float, float],
) -> dict[str, Any]:
    scaled_input = scale_selected_features(raw_input, scaler=scaler, model_features=model_features)
    probability = float(model.predict_proba(scaled_input)[0, 1])
    threshold = 0.5
    return {
        'probability': probability,
        'prediction': int(probability >= threshold),
        'risk_level': classify_risk(probability, cutoffs),
        'scaled_input': scaled_input,
        'raw_input': {feature: float(raw_input[feature]) for feature in model_features},
    }
