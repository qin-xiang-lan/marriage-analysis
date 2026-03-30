import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
import chardet
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# 页面配置
# ------------------------------
st.set_page_config(
    page_title="结婚离婚数据分析与预测",
    page_icon="💍",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>全国结婚离婚数据分析与预测</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>使用ARIMA模型对未来结婚/离婚登记数量进行预测</p>",
    unsafe_allow_html=True
)

# ==================== 重要：先定义所有函数 ====================

# ------------------------------
# 数据清洗和聚合函数
# ------------------------------
def clean_and_aggregate(df):
    """数据清洗和全国聚合"""
    # 地区名称规范化
    if '地区' in df.columns:
        df["地区"] = df["地区"].apply(
            lambda x: "内蒙古" if x == "内蒙古自治区" else
            "新疆" if x == "新疆维吾尔自治区" else
            "宁夏" if x == "宁夏回族自治区" else
            "广西" if x == "广西壮族自治区" else
            x[:-3] if str(x).endswith("自治区") else
            x[:-1] if str(x).endswith("省") or str(x).endswith("市") else
            x[:-5] if str(x).endswith("特别行政区") else
            x
        )

    # 年份转换
    df["年份"] = df["年份"].astype(str).str.replace("年", "").astype(int)

    # 数值列转换
    numeric_cols = ['结婚登记(万对)', '离婚登记(万对)', '内地居民初婚登记(万人)',
                    '内地居民再婚登记(万人)', '粗离婚率(‰)', '总人口(万人)',
                    '人口出生率(%)', 'GDP(亿元)', '全体居民人均可支配收入(元)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 全国数据聚合
    sum_cols = ['结婚登记(万对)', '离婚登记(万对)', 'GDP(亿元)', '总人口(万人)',
                '内地居民初婚登记(万人)', '内地居民再婚登记(万人)']
    existing_sum_cols = [col for col in sum_cols if col in df.columns]
    national = df.groupby('年份')[existing_sum_cols].sum().reset_index()

    # 加权平均函数
    def weighted_mean(series, weights):
        return (series * weights).sum() / weights.sum() if weights.sum() > 0 else np.nan

    # 人口出生率加权平均
    if '人口出生率(%)' in df.columns and '总人口(万人)' in df.columns:
        birth_rate = df.groupby('年份').apply(
            lambda x: weighted_mean(x['人口出生率(%)'], x['总人口(万人)'])
        ).reset_index(name='人口出生率(%)')
        national = national.merge(birth_rate, on='年份', how='left')

    # 人均可支配收入加权平均
    if '全体居民人均可支配收入(元)' in df.columns and '总人口(万人)' in df.columns:
        income_data = df[df['全体居民人均可支配收入(元)'].notna()].copy()
        if len(income_data) > 0:
            national_income = income_data.groupby('年份').apply(
                lambda x: weighted_mean(x['全体居民人均可支配收入(元)'], x['总人口(万人)'])
            ).reset_index(name='全体居民人均可支配收入(元)')
            national = national.merge(national_income, on='年份', how='left')

    return df, national


# ------------------------------
# 加载默认数据函数
# ------------------------------
def load_default_data():
    """加载默认数据文件"""
    try:
        # 尝试多种编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv("data.csv", encoding=enc)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            st.error("无法读取默认数据文件，请检查文件编码")
            return None, None

        return clean_and_aggregate(df)
    except FileNotFoundError:
        st.error("默认数据文件 data.csv 不存在，请上传文件")
        return None, None
    except Exception as e:
        st.error(f"加载默认数据失败: {e}")
        return None, None


# ------------------------------
# 加载上传文件函数
# ------------------------------
def load_uploaded_file(uploaded_file):
    """加载上传的文件"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='gbk')
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')
        else:
            st.error(f"不支持的文件格式：{file_extension}。请上传CSV或Excel文件。")
            return None, None
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        return None, None

    return clean_and_aggregate(df)


# ------------------------------
# 数据获取函数
# ------------------------------
@st.cache_data
def get_data(uploaded_file):
    if uploaded_file is not None:
        return load_uploaded_file(uploaded_file)
    else:
        return load_default_data()


# ==================== 主程序开始 ====================

# ------------------------------
# 文件上传选项（侧边栏）
# ------------------------------
st.sidebar.header("数据源设置")
uploaded_file = st.sidebar.file_uploader(
    "上传表格文件（CSV/Excel）",
    type=['csv', 'xlsx', 'xls']
)
use_default = st.sidebar.checkbox("使用默认数据", value=True if uploaded_file is None else False)

# 加载数据
df, national = get_data(uploaded_file)

if df is None or national is None:
    st.stop()

# ------------------------------
# 列映射设置（侧边栏）
# ------------------------------
st.sidebar.markdown("### 列映射设置")
# 年份列（必须）
year_options = df.columns.tolist()
year_col = st.sidebar.selectbox("年份列", year_options, index=year_options.index('年份') if '年份' in year_options else 0)
# 结婚登记列
marriage_options = ['无'] + [col for col in df.columns if '结婚' in col or 'marriage' in col.lower()]
marriage_col = st.sidebar.selectbox("结婚登记列", marriage_options, index=1 if len(marriage_options)>1 else 0)
# 离婚登记列
divorce_options = ['无'] + [col for col in df.columns if '离婚' in col or 'divorce' in col.lower()]
divorce_col = st.sidebar.selectbox("离婚登记列", divorce_options, index=1 if len(divorce_options)>1 else 0)
# GDP列
gdp_options = ['无'] + [col for col in df.columns if 'GDP' in col or 'gdp' in col.lower() or '国内生产总值' in col]
gdp_col = st.sidebar.selectbox("GDP列", gdp_options, index=1 if len(gdp_options)>1 else 0)
# 可支配收入列
income_options = ['无'] + [col for col in df.columns if '可支配' in col or 'income' in col.lower()]
income_col = st.sidebar.selectbox("居民可支配收入列", income_options, index=1 if len(income_options)>1 else 0)
# 总人口列
pop_options = ['无'] + [col for col in df.columns if '人口' in col or 'population' in col.lower()]
pop_col = st.sidebar.selectbox("总人口列", pop_options, index=1 if len(pop_options)>1 else 0)
# 人口出生率列
birth_options = ['无'] + [col for col in df.columns if '出生率' in col or 'birth' in col.lower()]
birth_col = st.sidebar.selectbox("人口出生率列", birth_options, index=1 if len(birth_options)>1 else 0)
# 地区列（可选）
region_options = ['无'] + [col for col in df.columns if '地区' in col or 'region' in col.lower()]
region_col = st.sidebar.selectbox("地区列（如无请选'无'）", region_options, index=0)

# 根据映射重命名列
rename_dict = {}
if marriage_col != '无': rename_dict[marriage_col] = '结婚登记(万对)'
if divorce_col != '无': rename_dict[divorce_col] = '离婚登记(万对)'
if gdp_col != '无': rename_dict[gdp_col] = 'GDP(亿元)'
if income_col != '无': rename_dict[income_col] = '全体居民人均可支配收入(元)'
if pop_col != '无': rename_dict[pop_col] = '总人口(万人)'
if birth_col != '无': rename_dict[birth_col] = '人口出生率(%)'
if region_col != '无': rename_dict[region_col] = '地区'
df = df.rename(columns=rename_dict)

# 处理年份列
df['年份'] = df[year_col].astype(str).str.extract(r'(\d{4})').astype(float).astype(int)
df = df.dropna(subset=['年份']).reset_index(drop=True)

# 数据预览
if st.sidebar.checkbox("预览上传数据"):
    st.sidebar.dataframe(df.head())

# 注意：这里有一个逻辑问题，需要修正
# 原代码中的 else 部分逻辑有问题，改为：
if uploaded_file is None and use_default:
    # 已经加载了默认数据，不需要再次加载
    st.sidebar.info("使用默认数据 data.csv")
else:
    if uploaded_file is not None:
        st.sidebar.info(f"使用上传文件: {uploaded_file.name}")
    else:
        st.warning("请上传文件或勾选'使用默认数据'")
        st.stop()

# ------------------------------
# 侧边栏：地区选择
# ------------------------------
if '地区' in df.columns:
    regions = ['全国'] + sorted(df['地区'].unique().tolist())
    selected_region = st.sidebar.selectbox("选择地区", regions, index=0)
else:
    selected_region = '全国'

# 根据地区筛选数据
if selected_region == '全国':
    if '地区' in df.columns:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if '年份' in numeric_cols:
            numeric_cols.remove('年份')
        df_region = df.groupby('年份')[numeric_cols].sum().reset_index()
    else:
        df_region = df.copy()
else:
    df_region = df[df['地区'] == selected_region].copy()

# 确保年份是整数
df_region['年份'] = df_region['年份'].astype(int)

# ------------------------------
# 数据预览选项卡
# ------------------------------
tab1, tab2 = st.tabs(["原始上传数据", "当前分析数据"])

with tab1:
    st.dataframe(df, use_container_width=True)
    st.caption("原始文件内容（未经过地区筛选和聚合）")

with tab2:
    st.dataframe(df_region, use_container_width=True)
    st.caption("处理后数据（已筛选、清洗，用于分析和预测）")


# ------------------------------
# 主要可视化区域
# ------------------------------
col1, col2 = st.columns(2)

# ----- 左侧：结婚/离婚历史趋势 -----
with col1:
    st.subheader("📈 结婚与离婚登记历史趋势")
    y_columns = ['结婚登记(万对)', '离婚登记(万对)']
    available_cols = [col for col in y_columns if col in df_region.columns]
    if available_cols:
        # 动态获取年份范围
        min_year = int(df_region['年份'].min())
        max_year = int(df_region['年份'].max())
        fig = px.line(df_region, x='年份', y=available_cols,
                      title=f"{min_year}-{max_year}年结婚与离婚登记变化",
                      markers=True)
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("数据中缺少结婚或离婚登记列")

# ----- 右侧：经济和人口指标趋势 -----
with col2:
    st.subheader("📊 经济与人口指标趋势")
    other_indicators = ['GDP(亿元)', '全体居民人均可支配收入(元)', '总人口(万人)', '人口出生率(%)']
    available_others = [col for col in other_indicators if col in df_region.columns]
    if available_others:
        selected = st.multiselect(
            "选择要显示的指标",
            options=available_others,
            default=[available_others[0]],
            key="economic_indicators"
        )
        if selected:
            # 动态获取年份范围
            min_year = int(df_region['年份'].min())
            max_year = int(df_region['年份'].max())
            fig2 = px.line(df_region, x='年份', y=selected,
                           title=f"{min_year}-{max_year}年经济与人口指标变化趋势",
                           markers=True,
                           labels={'value': '数值', 'variable': '指标'})
            fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig2, use_container_width=True)

            if st.checkbox("显示指标变化详情"):
                latest_year = df_region['年份'].max()
                earliest_year = df_region['年份'].min()
                latest_data = df_region[df_region['年份'] == latest_year][selected].iloc[0]
                earliest_data = df_region[df_region['年份'] == earliest_year][selected].iloc[0]
                change = ((latest_data - earliest_data) / earliest_data * 100).round(2)
                change_df = pd.DataFrame({
                    '指标': selected,
                    f'{latest_year}年数值': latest_data.values,
                    f'较{earliest_year}年变化率(%)': change.values
                })
                st.dataframe(change_df, use_container_width=True)
        else:
            st.info("👆 请至少选择一个指标")
    else:
        st.warning("数据中缺少经济或人口指标列")



# ------------------------------
# ARIMA预测函数
# ------------------------------

st.subheader("🔮 ARIMA模型预测未来结婚/离婚登记数量")

# 可修改参数
steps = st.slider("选择预测未来年数", min_value=1, max_value=5, value=5)
min_marriage = 60   # 结婚登记最小合理值（万对）
min_divorce = 80    # 离婚登记最小合理值（万对）
max_annual_decline = 0.08  # 最大年下降率（8%）

# 存储预测结果供结论使用
forecast_mar = None
forecast_div = None
best_order_mar = None
best_order_div = None

# 创建两列分别显示结婚和离婚预测
pred_col1, pred_col2 = st.columns(2)


def exponential_smoothing_forecast(series, steps=3, min_value=50, max_decline_rate=0.08):
    """
    指数平滑预测（备选方案）
    """
    try:
        model = ExponentialSmoothing(series, trend='add', seasonal=None)
        fit = model.fit()
        forecast = fit.forecast(steps)
        forecast_values = forecast.values if hasattr(forecast, 'values') else np.array(forecast)
        forecast_values = forecast_values.copy()  # 创建副本
        method = "Holt指数平滑"
    except:
        # 降级：保守预测
        last_value = series.iloc[-1]
        forecast_values = []
        current = last_value
        for i in range(steps):
            current = current * (1 - max_decline_rate)
            current = max(current, min_value)
            forecast_values.append(current)
        forecast_values = np.array(forecast_values)
        method = "保守下降率预测"

    # 应用最小值约束
    forecast_values = np.maximum(forecast_values, min_value)

    return None, forecast_values, method


def arima_forecast_safe(series, steps=5, min_value=50, max_decline_rate=0.08):
    """
    带约束的ARIMA预测函数（修复版）
    - 限制差分阶数不超过1（避免过度差分）
    - 应用下降率约束
    - 确保预测值非负且合理
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    # 网格搜索，限制差分阶数 d <= 1
    for p in range(4):
        for d in range(2):  # d最大为1，避免过度差分
            for q in range(4):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit(method_kwargs={'disp': False})
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except:
                    continue

    # 如果所有模型都失败 → 使用指数平滑兜底
    if best_model is None:
        _, forecast_values, method = exponential_smoothing_forecast(series, steps, min_value, max_decline_rate)
        return None, forecast_values, f"指数平滑({method})"

    # 执行预测
    forecast_result = best_model.forecast(steps=steps)
    forecast_values = forecast_result.values if hasattr(forecast_result, 'values') else np.array(forecast_result)

    # ========== 关键修复：创建可修改的副本 ==========
    forecast_values = forecast_values.copy()  # 创建副本，避免只读错误

    # 应用约束
    last_value = series.iloc[-1]

    for i in range(len(forecast_values)):
        # 约束1：年下降率不超过 max_decline_rate
        max_allowed = last_value * (1 - max_decline_rate) ** (i + 1)
        if forecast_values[i] < max_allowed:
            forecast_values[i] = max_allowed

        # 约束2：不低于最小值
        forecast_values[i] = max(forecast_values[i], min_value)

        # 约束3：确保单调递减（如果历史是递减的）
        if len(series) >= 3 and series.iloc[-1] < series.iloc[-3]:
            if i > 0 and forecast_values[i] > forecast_values[i - 1]:
                forecast_values[i] = forecast_values[i - 1] * 0.98

    # 最后再确保无负数
    forecast_values = np.maximum(forecast_values, 0)

    return best_model, forecast_values, best_order


# ---------- 结婚登记预测 ----------
with pred_col1:
    st.markdown("#### 💒 结婚登记预测")
    if '结婚登记(万对)' in df_region.columns:
        series_marriage = df_region.set_index('年份')['结婚登记(万对)'].sort_index()
        if len(series_marriage) >= 4:
            try:
                # 训练集（全部数据，因为年份少）
                train_mar = series_marriage

                # ✅ 调用修复后的函数
                model_mar, forecast_mar, best_order_mar = arima_forecast_safe(
                    train_mar, steps=steps, min_value=min_marriage, max_decline_rate=max_annual_decline
                )

                last_year = series_marriage.index[-1]
                future_years = list(range(last_year + 1, last_year + steps + 1))

                # 绘图
                fig_marriage = go.Figure()
                fig_marriage.add_trace(go.Scatter(
                    x=series_marriage.index, y=series_marriage.values,
                    mode='lines+markers', name='历史数据', line=dict(color='#1f77b4')
                ))
                fig_marriage.add_trace(go.Scatter(
                    x=future_years, y=forecast_mar,
                    mode='lines+markers', name='ARIMA预测', line=dict(color='#ff4b5c', dash='dash')
                ))

                # 添加下限参考线
                fig_marriage.add_hline(
                    y=min_marriage, line_dash="dot", line_color="gray",
                    annotation_text=f"下限参考线({min_marriage}万对)", annotation_position="bottom right"
                )

                fig_marriage.update_layout(
                    title=f"结婚登记数量预测 | 最优阶数: {best_order_mar if best_order_mar else '指数平滑'}",
                    xaxis_title="年份", yaxis_title="万对",
                    yaxis=dict(range=[0, max(series_marriage.max(), forecast_mar.max()) * 1.1]),
                    height=400
                )
                st.plotly_chart(fig_marriage, use_container_width=True)

                # 预测表格
                pred_df = pd.DataFrame({
                    "年份": future_years,
                    "预测结婚登记(万对)": np.round(forecast_mar, 2)
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

                # 显示预测合理性检查
                with st.expander("查看预测合理性检查"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("历史最后值", f"{series_marriage.iloc[-1]:.2f} 万对")
                        st.metric("预测最后值", f"{forecast_mar[-1]:.2f} 万对")
                    with col2:
                        annual_change = (1 - (forecast_mar[-1] / series_marriage.iloc[-1]) ** (1 / steps)) * 100
                        st.metric("年均下降率", f"{annual_change:.1f}%",
                                  delta="-下降" if annual_change > 0 else "+上升")
                        if best_order_mar:
                            st.info(f"最优ARIMA阶数: {best_order_mar}")

            except Exception as e:
                st.error(f"ARIMA模型预测失败：{str(e)}")
                # 指数平滑兜底
                _, forecast_mar, method = exponential_smoothing_forecast(
                    series_marriage, steps=steps, min_value=min_marriage, max_decline_rate=max_annual_decline
                )
                best_order_mar = f"指数平滑({method})"

                last_year = series_marriage.index[-1]
                future_years = list(range(last_year + 1, last_year + steps + 1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series_marriage.index, y=series_marriage.values, name="历史"))
                fig.add_trace(go.Scatter(x=future_years, y=forecast_mar, name="指数平滑预测", line=dict(dash='dash')))
                fig.add_hline(y=min_marriage, line_dash="dot", line_color="gray",
                              annotation_text=f"下限参考线({min_marriage}万对)")
                fig.update_layout(title="结婚登记（备用方案）", yaxis=dict(range=[0, None]), height=400)
                st.plotly_chart(fig, use_container_width=True)

                # 显示预测表格
                pred_df = pd.DataFrame({
                    "年份": future_years,
                    "预测结婚登记(万对)": np.round(forecast_mar, 2)
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"数据点不足，需要≥4年，当前仅{len(series_marriage)}年")
    else:
        st.warning("缺少「结婚登记(万对)」字段")

# ---------- 离婚登记预测 ----------
with pred_col2:
    st.markdown("#### 💔 离婚登记预测")
    if '离婚登记(万对)' in df_region.columns:
        series_divorce = df_region.set_index('年份')['离婚登记(万对)'].sort_index()
        if len(series_divorce) >= 4:
            try:
                train_div = series_divorce
                # ✅ 调用修复后的函数
                model_div, forecast_div, best_order_div = arima_forecast_safe(
                    train_div, steps=steps, min_value=min_divorce, max_decline_rate=max_annual_decline
                )

                last_year = series_divorce.index[-1]
                future_years = list(range(last_year + 1, last_year + steps + 1))

                fig_divorce = go.Figure()
                fig_divorce.add_trace(go.Scatter(
                    x=series_divorce.index, y=series_divorce.values,
                    mode='lines+markers', name='历史数据', line=dict(color='#2ca02c')
                ))
                fig_divorce.add_trace(go.Scatter(
                    x=future_years, y=forecast_div,
                    mode='lines+markers', name='ARIMA预测', line=dict(color='#ff4b5c', dash='dash')
                ))

                # 添加下限参考线
                fig_divorce.add_hline(
                    y=min_divorce, line_dash="dot", line_color="gray",
                    annotation_text=f"下限参考线({min_divorce}万对)", annotation_position="bottom right"
                )

                fig_divorce.update_layout(
                    title=f"离婚登记数量预测 | 最优阶数: {best_order_div if best_order_div else '指数平滑'}",
                    xaxis_title="年份", yaxis_title="万对",
                    yaxis=dict(range=[0, max(series_divorce.max(), forecast_div.max()) * 1.1]),
                    height=400
                )
                st.plotly_chart(fig_divorce, use_container_width=True)

                pred_df2 = pd.DataFrame({
                    "年份": future_years,
                    "预测离婚登记(万对)": np.round(forecast_div, 2)
                })
                st.dataframe(pred_df2, use_container_width=True, hide_index=True)

                # 显示预测合理性检查
                with st.expander("查看预测合理性检查"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("历史最后值", f"{series_divorce.iloc[-1]:.2f} 万对")
                        st.metric("预测最后值", f"{forecast_div[-1]:.2f} 万对")
                    with col2:
                        annual_change = ((forecast_div[-1] / series_divorce.iloc[-1]) ** (1 / steps) - 1) * 100
                        st.metric("年均变化率", f"{annual_change:+.1f}%",
                                  delta="上升" if annual_change > 0 else "下降")
                        if best_order_div:
                            st.info(f"最优ARIMA阶数: {best_order_div}")

            except Exception as e:
                st.error(f"ARIMA模型预测失败：{str(e)}")
                # 指数平滑兜底
                _, forecast_div, method = exponential_smoothing_forecast(
                    series_divorce, steps=steps, min_value=min_divorce, max_decline_rate=max_annual_decline
                )
                best_order_div = f"指数平滑({method})"

                last_year = series_divorce.index[-1]
                future_years = list(range(last_year + 1, last_year + steps + 1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series_divorce.index, y=series_divorce.values, name="历史"))
                fig.add_trace(go.Scatter(x=future_years, y=forecast_div, name="指数平滑预测", line=dict(dash='dash')))
                fig.add_hline(y=min_divorce, line_dash="dot", line_color="gray",
                              annotation_text=f"下限参考线({min_divorce}万对)")
                fig.update_layout(title="离婚登记（备用方案）", yaxis=dict(range=[0, None]), height=400)
                st.plotly_chart(fig, use_container_width=True)

                # 显示预测表格
                pred_df2 = pd.DataFrame({
                    "年份": future_years,
                    "预测离婚登记(万对)": np.round(forecast_div, 2)
                })
                st.dataframe(pred_df2, use_container_width=True, hide_index=True)
        else:
            st.warning(f"数据点不足，需要≥4年，当前仅{len(series_divorce)}年")
    else:
        st.warning("缺少「离婚登记(万对)」字段")

# ------------------------------
# 相关性分析
# ------------------------------
st.subheader("📌 指标相关性分析")

# 筛选数值列（排除年份）
numeric_cols = df_region.select_dtypes(include='number').columns.tolist()
if '年份' in numeric_cols:
    numeric_cols.remove('年份')

# 进一步确保列存在且为数值类型
numeric_cols_corr = [col for col in numeric_cols if col in df_region.columns and col != '年份']

if len(numeric_cols_corr) > 1:
    # 计算相关系数矩阵
    corr_df = df_region[numeric_cols_corr].corr()

    # 使用plotly绘制热图（与原代码风格一致）
    import plotly.express as px

    fig = px.imshow(
        corr_df,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="变量相关性热图"
    )

    fig.update_layout(
        width=900,
        height=700,
        font=dict(size=12),
        title_font_size=14
    )

    st.plotly_chart(fig, use_container_width=True)

    # 显示与结婚登记的相关性详情
    if '结婚登记(万对)' in corr_df.columns:
        with st.expander("📊 与结婚登记的相关性详情"):
            corr_with_marriage = corr_df['结婚登记(万对)'].sort_values(ascending=False)

            # 创建DataFrame并格式化显示
            result_df = pd.DataFrame({
                '指标': corr_with_marriage.index,
                '相关系数': corr_with_marriage.values
            })
            result_df['相关系数'] = result_df['相关系数'].round(4)
            result_df = result_df.set_index('指标')

            st.dataframe(result_df, use_container_width=True)

            # 添加说明
            st.caption("💡 相关系数范围：-1（完全负相关）到 1（完全正相关），0表示无相关")
else:
    st.warning(f"没有足够的数值列计算相关性，当前只有 {len(numeric_cols_corr)} 个数值列")

# ------------------------------
# 动态生成分析结论
# ------------------------------
st.subheader("📝 分析结论")

conclusion_text = ""

# 1. 结婚登记趋势分析
if '结婚登记(万对)' in df_region.columns:
    series_mar = df_region.set_index('年份')['结婚登记(万对)'].sort_index()
    last_hist = series_mar.iloc[-1]
    first_hist = series_mar.iloc[0]
    hist_change = ((last_hist - first_hist) / first_hist * 100) if first_hist != 0 else 0

    if forecast_mar is not None and len(forecast_mar) > 0:
        last_pred = forecast_mar[-1]
        pred_change = ((last_pred - last_hist) / last_hist * 100) if last_hist != 0 else 0

        if pred_change > 5:
            trend_desc = "显著上升"
        elif pred_change > 0:
            trend_desc = "小幅上升"
        elif pred_change > -5:
            trend_desc = "小幅下降"
        else:
            trend_desc = "显著下降"

        conclusion_text += f"- **结婚登记**：从{first_hist:.0f}万对（{series_mar.index[0]}年）变化至{last_hist:.0f}万对（{series_mar.index[-1]}年），预计{series_mar.index[-1] + steps}年将达{last_pred:.0f}万对，**{trend_desc}**（变化{pred_change:+.1f}%）。\n"
    else:
        if hist_change > 5:
            trend_desc = "显著上升"
        elif hist_change > 0:
            trend_desc = "小幅上升"
        elif hist_change > -5:
            trend_desc = "小幅下降"
        else:
            trend_desc = "显著下降"
        conclusion_text += f"- **结婚登记**：{series_mar.index[0]}-{series_mar.index[-1]}年间从{first_hist:.0f}万对变化至{last_hist:.0f}万对，整体呈**{trend_desc}**趋势（变化{hist_change:+.1f}%）。\n"

# 2. 离婚登记趋势分析
if '离婚登记(万对)' in df_region.columns:
    series_div = df_region.set_index('年份')['离婚登记(万对)'].sort_index()
    last_hist_div = series_div.iloc[-1]
    first_hist_div = series_div.iloc[0]
    hist_change_div = ((last_hist_div - first_hist_div) / first_hist_div * 100) if first_hist_div != 0 else 0

    if forecast_div is not None and len(forecast_div) > 0:
        last_pred_div = forecast_div[-1]
        pred_change_div = ((last_pred_div - last_hist_div) / last_hist_div * 100) if last_hist_div != 0 else 0

        if pred_change_div > 5:
            trend_desc_div = "显著上升"
        elif pred_change_div > 0:
            trend_desc_div = "小幅上升"
        elif pred_change_div > -5:
            trend_desc_div = "小幅下降"
        else:
            trend_desc_div = "显著下降"

        conclusion_text += f"- **离婚登记**：从{first_hist_div:.0f}万对（{series_div.index[0]}年）变化至{last_hist_div:.0f}万对（{series_div.index[-1]}年），预计{series_div.index[-1] + steps}年将达{last_pred_div:.0f}万对，**{trend_desc_div}**（变化{pred_change_div:+.1f}%）。\n"
    else:
        if hist_change_div > 5:
            trend_desc_div = "显著上升"
        elif hist_change_div > 0:
            trend_desc_div = "小幅上升"
        elif hist_change_div > -5:
            trend_desc_div = "小幅下降"
        else:
            trend_desc_div = "显著下降"
        conclusion_text += f"- **离婚登记**：{series_div.index[0]}-{series_div.index[-1]}年间从{first_hist_div:.0f}万对变化至{last_hist_div:.0f}万对，整体呈**{trend_desc_div}**趋势（变化{hist_change_div:+.1f}%）。\n"

# 3. 与GDP的关系分析
if 'GDP(亿元)' in df_region.columns and '结婚登记(万对)' in df_region.columns:
    corr_gdp = df_region['结婚登记(万对)'].corr(df_region['GDP(亿元)'])
    if corr_gdp > 0.5:
        relation_gdp = "强正相关"
    elif corr_gdp > 0.3:
        relation_gdp = "弱正相关"
    elif corr_gdp > -0.3:
        relation_gdp = "无明显相关"
    elif corr_gdp > -0.5:
        relation_gdp = "弱负相关"
    else:
        relation_gdp = "强负相关"
    conclusion_text += f"- **与GDP的关系**：结婚登记与GDP呈**{relation_gdp}**（相关系数{corr_gdp:.2f}）。\n"

# 4. 与人均可支配收入的关系
if '全体居民人均可支配收入(元)' in df_region.columns and '结婚登记(万对)' in df_region.columns:
    valid_data = df_region[df_region['全体居民人均可支配收入(元)'].notna()]
    if len(valid_data) > 5:
        corr_income = valid_data['结婚登记(万对)'].corr(valid_data['全体居民人均可支配收入(元)'])
        if corr_income > 0.5:
            relation_inc = "强正相关"
        elif corr_income > 0.3:
            relation_inc = "弱正相关"
        elif corr_income > -0.3:
            relation_inc = "无明显相关"
        elif corr_income > -0.5:
            relation_inc = "弱负相关"
        else:
            relation_inc = "强负相关"
        conclusion_text += f"- **与人均可支配收入的关系**：结婚登记与收入呈**{relation_inc}**（相关系数{corr_income:.2f}）。\n"

st.markdown(conclusion_text)

# ------------------------------
# 固定分析结论
# ------------------------------
with st.expander("📋 详细分析报告"):
    st.markdown("""
    ### 核心发现

    #### 1. 结婚登记趋势
    - **2001-2013年**：结婚登记数量持续上升，2013年达到历史峰值
    - **2013-2024年**：结婚登记数量持续下降，2024年降至历史低位
    - **影响因素**：适婚人口减少、初婚年龄推迟、婚育成本上升

    #### 2. 离婚登记趋势
    - **2001-2019年**：离婚登记数量持续上升，2019年达到峰值
    - **2020-2024年**：离婚登记数量略有回落，趋于平稳
    - **影响因素**：离婚冷静期政策、婚姻观念变化、经济压力

    #### 3. 经济与婚姻的关系
    - **GDP增长与结婚率**：呈负相关，经济发展伴随结婚意愿降低
    - **人均收入与结婚率**：近十年呈现负相关趋势
    - **房价因素**：高房价是影响结婚决策的重要因素

    ### 政策建议

    1. **降低婚育成本**
       - 提供住房补贴或优惠贷款
       - 完善托育服务体系
       - 扩大教育资源供给

    2. **营造婚育友好环境**
       - 延长婚假、产假、陪产假
       - 保障女性就业权益
       - 推广灵活工作制度

    3. **加强婚姻家庭服务**
       - 婚前辅导和婚姻咨询
       - 家庭矛盾调解机制
       - 离婚冷静期配套服务
    """)

# ------------------------------
# 脚注
# ------------------------------
st.markdown("---")
st.caption("数据来源：国家统计局 | 分析工具：Streamlit + ARIMA")