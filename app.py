import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
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

# ------------------------------
# 文件上传选项（侧边栏）
# ------------------------------
st.sidebar.header("数据源设置")
uploaded_file = st.sidebar.file_uploader(
    "上传表格文件（CSV/Excel）",
    type=['csv', 'xlsx', 'xls']
)
use_default = st.sidebar.checkbox("使用默认数据", value=True if uploaded_file is None else False)

# ------------------------------
# 数据加载（支持上传和默认）
# ------------------------------
@st.cache_data
def load_default_data():
    df = pd.read_csv("data.csv", encoding='gbk')
    df['年份'] = df['年份'].astype(str).str.replace('年', '', regex=False).astype(int)
    return df


if uploaded_file is not None:
    # 判断文件后缀
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
            st.stop()
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

    st.sidebar.success("文件上传成功！")

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

else:
    if use_default:
        df = load_default_data()
        st.sidebar.info("使用默认数据 data.csv")
        # 默认数据已经有标准列名，无需映射
        rename_dict = {}  # 空字典
    else:
        st.warning("请上传文件或勾选'使用默认数据'")
        st.stop()

# ------------------------------
# 数据清洗（数值转换、缺失值处理）
# ------------------------------
numeric_std_cols = ['结婚登记(万对)', '离婚登记(万对)', 'GDP(亿元)', '全体居民人均可支配收入(元)', '总人口(万人)', '人口出生率(%)']
for col in numeric_std_cols:
    if col in df.columns:
        # 如果是字符串，移除逗号、百分号
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
        # 转换为数值
        df[col] = pd.to_numeric(df[col], errors='coerce')


# 检查缺失值
# missing = df[numeric_std_cols].isnull().sum()
# if missing.sum() > 0:
#     st.warning(f"数据中存在缺失值：{missing[missing>0].to_dict()}。将删除包含缺失值的行。")
#     df = df.dropna(subset=numeric_std_cols)


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
# 显示原始数据
# ------------------------------
# with st.expander("📋 查看原始数据（可筛选）"):
#     st.dataframe(df_region, use_container_width=True)


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
# ARIMA预测部分
# ------------------------------
st.subheader("🔮 ARIMA模型预测未来结婚/离婚登记数量")

# 模型参数（可从Jupyter中获得）
MARRIAGE_ORDER = (1, 1, 1)   # 请根据实际结果修改
DIVORCE_ORDER = (0, 1, 2)    # 请根据实际结果修改

steps = st.slider("选择预测未来年数", min_value=1, max_value=10, value=5)

pred_col1, pred_col2 = st.columns(2)


# 结婚登记预测
with pred_col1:
    st.markdown("#### 💒 结婚登记预测")
    if '结婚登记(万对)' in df_region.columns:
        series_marriage = df_region.set_index('年份')['结婚登记(万对)'].sort_index()
        if len(series_marriage) > max(MARRIAGE_ORDER[0], MARRIAGE_ORDER[2]):
            try:
                model_marriage = ARIMA(series_marriage, order=MARRIAGE_ORDER)
                model_marriage_fit = model_marriage.fit()
                forecast_marriage = model_marriage_fit.forecast(steps=steps)

                # 获取历史年份范围
                min_hist = int(series_marriage.index.min())
                max_hist = int(series_marriage.index.max())
                # 生成未来年份
                last_year = max_hist
                future_years = list(range(last_year + 1, last_year + steps + 1))
                future_start = future_years[0]
                future_end = future_years[-1]

                # 绘制历史+预测
                fig_marriage = go.Figure()
                fig_marriage.add_trace(go.Scatter(
                    x=series_marriage.index, y=series_marriage.values,
                    mode='lines+markers', name='历史', line=dict(color='blue')
                ))
                fig_marriage.add_trace(go.Scatter(
                    x=future_years, y=forecast_marriage.values,
                    mode='lines+markers', name='预测', line=dict(color='red', dash='dash')
                ))
                # 动态标题
                title = f"{min_hist}-{max_hist}年结婚登记历史与{future_start}-{future_end}年预测"
                fig_marriage.update_layout(
                    title=title,
                    xaxis_title="年份",
                    yaxis_title="万对"
                )
                st.plotly_chart(fig_marriage, use_container_width=True)

                # 显示预测值表格
                pred_df = pd.DataFrame({
                    '年份': future_years,
                    '预测结婚登记(万对)': forecast_marriage.values.round(2)
                })
                st.dataframe(pred_df, use_container_width=True)
            except Exception as e:
                st.error(f"结婚登记预测失败：{e}")
        else:
            st.warning("数据点太少，无法拟合所选ARIMA模型")
    else:
        st.warning("数据中缺少'结婚登记(万对)'列")


# 离婚登记预测
with pred_col2:
    st.markdown("#### 💔 离婚登记预测")
    if '离婚登记(万对)' in df_region.columns:
        series_divorce = df_region.set_index('年份')['离婚登记(万对)'].sort_index()
        if len(series_divorce) > max(DIVORCE_ORDER[0], DIVORCE_ORDER[2]):
            try:
                model_divorce = ARIMA(series_divorce, order=DIVORCE_ORDER)
                model_divorce_fit = model_divorce.fit()
                forecast_divorce = model_divorce_fit.forecast(steps=steps)

                # 获取历史年份范围
                min_hist = int(series_divorce.index.min())
                max_hist = int(series_divorce.index.max())
                # 生成未来年份
                last_year = max_hist
                future_years = list(range(last_year + 1, last_year + steps + 1))
                future_start = future_years[0]
                future_end = future_years[-1]

                fig_divorce = go.Figure()
                fig_divorce.add_trace(go.Scatter(
                    x=series_divorce.index, y=series_divorce.values,
                    mode='lines+markers', name='历史', line=dict(color='green')
                ))
                fig_divorce.add_trace(go.Scatter(
                    x=future_years, y=forecast_divorce.values,
                    mode='lines+markers', name='预测', line=dict(color='orange', dash='dash')
                ))
                # 动态标题
                title = f"{min_hist}-{max_hist}年离婚登记历史与{future_start}-{future_end}年预测"
                fig_divorce.update_layout(
                    title=title,
                    xaxis_title="年份",
                    yaxis_title="万对"
                )
                st.plotly_chart(fig_divorce, use_container_width=True)

                pred_df2 = pd.DataFrame({
                    '年份': future_years,
                    '预测离婚登记(万对)': forecast_divorce.values.round(2)
                })
                st.dataframe(pred_df2, use_container_width=True)
            except Exception as e:
                st.error(f"离婚登记预测失败：{e}")
        else:
            st.warning("数据点太少，无法拟合所选ARIMA模型")
    else:
        st.warning("数据中缺少'离婚登记(万对)'列")


# ------------------------------
# 相关性分析
# ------------------------------
st.subheader("📌 指标相关性分析")
numeric_cols = df_region.select_dtypes(include='number').columns.tolist()
if '年份' in numeric_cols:
    numeric_cols.remove('年份')
if len(numeric_cols) > 1:
    corr = df_region[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                         title="各指标相关系数矩阵",
                         color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("没有足够的数值列计算相关性")


# 根据预测结果生成趋势描述（如果可用）
conclusion_text = "根据2001-2024年数据及ARIMA模型预测，可以得出以下初步结论：\n\n"

if '结婚登记(万对)' in df_region.columns and 'forecast_marriage' in locals():
    last_hist = series_marriage.values[-1]
    last_pred = forecast_marriage.values[-1]
    marriage_trend = "上升" if last_pred > last_hist else "下降" if last_pred < last_hist else "持平"
    conclusion_text += f"- **结婚登记**：整体呈现[上升/下降/波动]趋势，预计未来5年将继续**{marriage_trend}**。\n"
else:
    conclusion_text += "- **结婚登记**：数据不足，无法判断趋势。\n"

if '离婚登记(万对)' in df_region.columns and 'forecast_divorce' in locals():
    last_hist_div = series_divorce.values[-1]
    last_pred_div = forecast_divorce.values[-1]
    divorce_trend = "上升" if last_pred_div > last_hist_div else "下降" if last_pred_div < last_hist_div else "持平"
    conclusion_text += f"- **离婚登记**：预计未来5年将**{divorce_trend}**。\n"
else:
    conclusion_text += "- **离婚登记**：数据不足，无法判断趋势。\n"

# 与GDP、收入的关系
if 'GDP(亿元)' in df_region.columns and '结婚登记(万对)' in df_region.columns:
    corr_gdp = df_region['结婚登记(万对)'].corr(df_region['GDP(亿元)'])
    relation = "正相关" if corr_gdp > 0.3 else "负相关" if corr_gdp < -0.3 else "无明显相关"
    conclusion_text += f"- **与GDP的关系**：结婚登记与GDP呈**{relation}**（相关系数{corr_gdp:.2f}）。\n"
else:
    conclusion_text += "- **与GDP的关系**：数据不足，无法分析。\n"

if '全体居民人均可支配收入(元)' in df_region.columns and '结婚登记(万对)' in df_region.columns:
    corr_income = df_region['结婚登记(万对)'].corr(df_region['全体居民人均可支配收入(元)'])
    relation_inc = "正相关" if corr_income > 0.3 else "负相关" if corr_income < -0.3 else "无明显相关"
    conclusion_text += f"- **与人均可支配收入的关系**：结婚登记与收入呈**{relation_inc}**（相关系数{corr_income:.2f}）。\n"
else:
    conclusion_text += "- **与人均可支配收入的关系**：数据不足，无法分析。\n"


# ------------------------------
# 分析结论（可自定义）
# ------------------------------
st.subheader("📝 分析结论")
st.markdown("""
根据2001-2024年数据及ARIMA模型预测，可以得出以下初步结论：
- **结婚登记**：整体呈现**波动下降**趋势，2001-2013年缓慢上升，2013年达到峰值1347万对后持续下降，2024年已降至610万对。预计未来5年将继续下降。
- **离婚登记**：离婚登记数量在2001-2024年间整体呈**先升后稳**态势。2001-2019年持续上升，2019年达到峰值470万对，2020年后略有回落并趋于平稳。
- **GDP与结婚率**：相关性分析显示，结婚登记与GDP呈**显著负相关**（相关系数-0.85），表明经济发展伴随结婚意愿的**降低**。近十年人均可支配收入的提高与结婚登记下降趋势同步。
- **政策建议**：1、**鼓励适龄婚育**：针对结婚率下降，可出台住房补贴、税收优惠、延长婚假等政策，降低年轻人结婚成本。
              2、**加强婚姻辅导**：针对离婚率较高问题，推广婚前辅导、婚姻危机干预，帮助家庭维系稳定。
              3、**关注经济影响**：在经济发展规划中纳入家庭发展视角，平衡工作与家庭生活，营造鼓励婚育的社会氛围。
""")

# ------------------------------
# 脚注
# ------------------------------
st.markdown("---")
st.caption("数据来源：国家统计局 | 分析工具：Streamlit + ARIMA")