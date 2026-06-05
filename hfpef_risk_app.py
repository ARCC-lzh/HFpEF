from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import streamlit as st

from hfpef_risk_core import (
    build_single_case_lime_figure,
    build_single_case_shap_figures,
    compute_probability_percentile,
    derive_runtime_cutoffs,
    get_explainer_runtime_status,
    load_runtime_explainers,
    load_runtime_assets,
    predict_with_risk_stratification,
)


PROJECT_DIR = Path(__file__).resolve().parent

FEATURE_LABELS: dict[str, str] = {
    'NT_proBNP': 'NT-proBNP',
    'EF': '射血分数',
    'E_e': 'E/e′',
    'Age': '年龄',
    'AFc': '房颤史',
    'HGB': '血红蛋白',
    'ALB': '白蛋白',
    'FT3': '游离三碘甲状腺原氨酸',
    'CR': '肌酐',
    'BUN': '尿素氮',
    'FIB': '纤维蛋白原',
}

FEATURE_UNITS: dict[str, str] = {
    'NT_proBNP': 'pg/mL',
    'EF': '%',
    'E_e': '',
    'Age': '岁',
    'AFc': '0/1',
    'HGB': 'g/L',
    'ALB': 'g/L',
    'FT3': 'pmol/L',
    'CR': 'μmol/L',
    'BUN': 'mmol/L',
    'FIB': 'g/L',
}

CATEGORICAL_LABELS: dict[str, dict[int, str]] = {
    'AFc': {0: '无', 1: '有'},
}

RISK_LABELS = {'Low': '低风险', 'Medium': '中风险', 'High': '高风险'}
PREDICTION_LABELS = {0: '模型判断：Event=0', 1: '模型判断：Event=1'}
RECOMMENDATIONS = {
    'Low': '建议按常规流程管理，结合症状、超声和实验室指标进行常规随访。',
    'Medium': '建议重点复核 NT-proBNP、EF、肾功能和贫血相关指标，并加强近期复查。',
    'High': '建议作为重点预警人群，由医生尽快复核，必要时纳入强化评估与随访。',
}

CHINESE_FONT_CANDIDATES = [
    'Microsoft YaHei',
    'SimHei',
    'Noto Sans CJK SC',
    'Noto Sans CJK JP',
    'Noto Sans SC',
    'Source Han Sans SC',
    'PingFang SC',
    'WenQuanYi Zen Hei',
    'Arial Unicode MS',
]
CHINESE_FONT_SEARCH_DIRS = [
    PROJECT_DIR / 'fonts',
    Path('/usr/share/fonts/opentype/noto'),
    Path('/usr/share/fonts/truetype/noto'),
    Path('/usr/share/fonts/truetype/wqy'),
    Path('/usr/local/share/fonts'),
    Path('/home/appuser/.fonts'),
]

CURRENT_SAMPLE_SECTION_TITLES = [
    '当前样本结论',
    'SHAP 单例解释',
    'LIME 单例解释',
    '全局特征重要性',
]
BATTERY_STREAMLIT_COMMAND = (
    'cd D:\\Code\\HFpEF_v2\\web\n'
    '& "C:\\Users\\ARCC\\.conda\\envs\\battery\\python.exe" -m streamlit run hfpef_risk_app.py'
)


def register_chinese_font_candidates() -> None:
    seen_paths: set[str] = set()
    for font_dir in CHINESE_FONT_SEARCH_DIRS:
        if not Path(font_dir).exists():
            continue
        for pattern in ('*.ttf', '*.ttc', '*.otf'):
            for font_path in Path(font_dir).rglob(pattern):
                font_path_str = str(font_path.resolve())
                if font_path_str in seen_paths:
                    continue
                try:
                    font_manager.fontManager.addfont(font_path_str)
                    seen_paths.add(font_path_str)
                except Exception:
                    continue


def configure_matplotlib_for_chinese() -> str:
    register_chinese_font_candidates()
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = 'DejaVu Sans'
    for candidate in CHINESE_FONT_CANDIDATES:
        if candidate in available_fonts:
            selected_font = candidate
            break

    current_fonts = list(plt.rcParams.get('font.sans-serif', []))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [selected_font, *[font for font in current_fonts if font != selected_font]]
    plt.rcParams['axes.unicode_minus'] = False
    return selected_font


def _normalize_label_token(value: str) -> str:
    return ''.join(ch for ch in value.lower() if ch.isalnum())


def format_feature_label(feature: str) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    if _normalize_label_token(label) == _normalize_label_token(feature):
        return label
    return f'{label}（{feature}）'



@st.cache_resource(show_spinner=False)
def load_context() -> dict[str, Any]:
    font_name = configure_matplotlib_for_chinese()
    assets = load_runtime_assets(PROJECT_DIR)
    cutoffs = derive_runtime_cutoffs(assets)
    display_feature_names = [format_feature_label(feature) for feature in assets.model_features]
    explainability_status = get_explainer_runtime_status()
    runtime_explainers = None
    if explainability_status['is_ready']:
        try:
            runtime_explainers = load_runtime_explainers(
                assets,
                display_feature_names=display_feature_names,
            )
        except Exception as exc:
            explainability_status = {
                **explainability_status,
                'is_ready': False,
                'message': f"{explainability_status['message']} 解释器初始化失败: {exc}",
            }

    importance = pd.DataFrame(
        {
            'feature': assets.model_features,
            'importance': getattr(assets.model, 'feature_importances_', [0.0] * len(assets.model_features)),
        }
    ).sort_values('importance', ascending=False)
    importance['label'] = importance['feature'].map(format_feature_label)

    default_prediction = predict_with_risk_stratification(
        model=assets.model,
        scaler=assets.scaler,
        model_features=assets.model_features,
        raw_input=assets.default_input_values,
        cutoffs=cutoffs,
    )

    return {
        'assets': assets,
        'cutoffs': cutoffs,
        'reference_probabilities': assets.reference_probabilities,
        'feature_specs': assets.feature_specs,
        'feature_importance': importance.reset_index(drop=True),
        'default_input_values': assets.default_input_values,
        'default_prediction': default_prediction,
        'display_feature_names': display_feature_names,
        'explainability_status': explainability_status,
        'runtime_explainers': runtime_explainers,
        'font_name': font_name,
    }



def build_default_input_values(context: dict[str, Any]) -> dict[str, float]:
    return dict(context['default_input_values'])


def get_tab_labels() -> list[str]:
    return ['当前样本风险']


def _format_categorical_option(feature: str, option: int) -> str:
    return CATEGORICAL_LABELS.get(feature, {}).get(int(option), str(option))



def render_input_form(context: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    assets = context['assets']
    specs = context['feature_specs']
    default_values = build_default_input_values(context)
    user_input: dict[str, float] = {}

    st.subheader('患者信息录入')
    st.caption('当前默认值来自本次回顾性训练集的中位数或众数。分类变量使用下拉框，连续变量使用数值输入。')

    with st.form('predict_form', clear_on_submit=False):
        columns = st.columns(2)
        for idx, feature in enumerate(assets.model_features):
            spec = specs[feature]
            label = format_feature_label(feature)
            help_text = f"训练集范围：{spec['data_min']:.2f} ~ {spec['data_max']:.2f}"
            default_value = float(default_values.get(feature, spec['default']))
            with columns[idx % 2]:
                if spec['is_categorical']:
                    options = spec['options'] or [int(round(spec['default']))]
                    option_value = int(round(default_value))
                    default_index = options.index(option_value) if option_value in options else 0
                    selected = st.selectbox(
                        label=label,
                        options=options,
                        index=default_index,
                        help=help_text,
                        format_func=lambda opt, feature_name=feature: _format_categorical_option(feature_name, int(opt)),
                    )
                    user_input[feature] = float(selected)
                else:
                    clipped_default = float(min(max(default_value, spec['data_min']), spec['data_max']))
                    user_input[feature] = float(
                        st.number_input(
                            label=label,
                            min_value=float(spec['data_min']),
                            max_value=float(spec['data_max']),
                            value=clipped_default,
                            step=float(spec['step']),
                            help=help_text,
                        )
                    )

        submit = st.form_submit_button('预测 HFpEF Event 风险', type='primary')
    return submit, user_input


def render_probability_position_chart(context: dict[str, Any], probability: float) -> None:
    low_cutoff, high_cutoff = context['cutoffs']
    reference_probabilities = context['reference_probabilities']
    percentile = compute_probability_percentile(reference_probabilities, probability)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.hist(reference_probabilities, bins=30, color='#cbd5e1', edgecolor='white')
    ax.axvline(low_cutoff, color='#0ea5e9', linestyle='--', linewidth=1.5, label=f'低风险阈值 {low_cutoff:.3f}')
    ax.axvline(high_cutoff, color='#f59e0b', linestyle='--', linewidth=1.5, label=f'高风险阈值 {high_cutoff:.3f}')
    ax.axvline(probability, color='#dc2626', linewidth=2.2, label=f'当前输入 {probability:.3f}')
    ax.set_xlabel('Event 预测概率')
    ax.set_ylabel('参考样本数')
    ax.set_title(f'当前输入在参考队列中的位置（{percentile:.1f} 百分位）')
    ax.legend(fontsize=8.5)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_runtime_explainer_error(context: dict[str, Any]) -> None:
    st.error(context['explainability_status']['message'])
    st.caption('请使用下面的命令启动页面，以启用当前样本的实时 SHAP / LIME 解释。')
    st.code(BATTERY_STREAMLIT_COMMAND, language='powershell')


def render_current_sample_summary(context: dict[str, Any], result: dict[str, Any]) -> None:
    probability = float(result['probability'])
    risk_level = str(result['risk_level'])
    prediction = int(result['prediction'])

    c1, c2, c3 = st.columns(3)
    c1.metric('Event 概率', f'{probability:.2%}')
    c2.metric('风险分层', RISK_LABELS[risk_level])
    c3.metric('模型判断', PREDICTION_LABELS[prediction])
    st.progress(min(max(probability, 0.0), 1.0), text=f'风险概率：{probability:.2%}')
    st.info(RECOMMENDATIONS[risk_level])
    render_probability_position_chart(context, probability)


def render_shap_single_case_section(context: dict[str, Any], result: dict[str, Any]) -> None:
    if not context['explainability_status']['is_ready'] or context['runtime_explainers'] is None:
        render_runtime_explainer_error(context)
        return

    st.caption('下面两张图是基于你当前输入实时生成的 SHAP 单例解释，不是训练时保存的固定病例图。')
    figures = build_single_case_shap_figures(
        runtime_explainers=context['runtime_explainers'],
        model_features=context['assets'].model_features,
        raw_input=result['raw_input'],
        scaled_input=result['scaled_input'],
        display_feature_names=context['display_feature_names'],
    )
    st.markdown('**SHAP Waterfall**')
    st.pyplot(figures['waterfall'], clear_figure=True)
    st.markdown('**SHAP Force Plot**')
    st.pyplot(figures['force'], clear_figure=True)


def render_lime_single_case_section(context: dict[str, Any], result: dict[str, Any]) -> None:
    if not context['explainability_status']['is_ready'] or context['runtime_explainers'] is None:
        render_runtime_explainer_error(context)
        return

    st.caption('下面是基于你当前输入实时生成的 LIME 单例局部解释。')
    figure = build_single_case_lime_figure(
        runtime_explainers=context['runtime_explainers'],
        model_features=context['assets'].model_features,
        raw_input=result['raw_input'],
        display_feature_names=context['display_feature_names'],
    )
    st.pyplot(figure, clear_figure=True)


def render_global_feature_importance(context: dict[str, Any]) -> None:
    st.caption('这是全局视角的随机森林特征重要性图，不随当前输入变化。')
    importance_df = context['feature_importance']
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(importance_df['label'], importance_df['importance'], color='#2563eb')
    ax.invert_yaxis()
    ax.set_xlabel('特征重要性')
    ax.set_ylabel('变量')
    ax.set_title('随机森林特征重要性')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_current_sample_tab(context: dict[str, Any], result: dict[str, Any]) -> None:
    expander_renderers = [
        render_current_sample_summary,
        render_shap_single_case_section,
        render_lime_single_case_section,
    ]
    for idx, title in enumerate(CURRENT_SAMPLE_SECTION_TITLES):
        with st.expander(title, expanded=idx == 0):
            if idx < len(expander_renderers):
                expander_renderers[idx](context, result)
            else:
                render_global_feature_importance(context)



def main() -> None:
    st.set_page_config(
        page_title='HFpEF 风险预测平台',
        page_icon='🫀',
        layout='wide',
    )
    st.title('HFpEF 风险预测平台（Random Forest）')
    st.markdown('**纪芳杰**  |  天津医科大学')
    st.caption('基于重新纳入 E/e′ 的回顾性数据库训练，输入 10 个关键特征，输出 Event 风险概率、固定 0.5 阈值判断、分层和解释图谱。')

    context = load_context()
    default_prediction = context['default_prediction']

    submitted, user_input = render_input_form(context)
    state_key = 'hfpef_last_prediction_result'
    if submitted:
        assets = context['assets']
        st.session_state[state_key] = predict_with_risk_stratification(
            model=assets.model,
            scaler=assets.scaler,
            model_features=assets.model_features,
            raw_input=user_input,
            cutoffs=context['cutoffs'],
        )

    (tab_predict,) = st.tabs(get_tab_labels())
    with tab_predict:
        if state_key not in st.session_state:
            st.info('请先填写上方表单并点击“预测 HFpEF Event 风险”。')
            render_current_sample_tab(context, default_prediction)
        else:
            render_current_sample_tab(context, st.session_state[state_key])


if __name__ == '__main__':
    main()
