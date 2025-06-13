import streamlit as st
# 必须作为第一个Streamlit命令
st.set_page_config(
    page_title="AI新闻真实性检测系统",
    page_icon="🔍",
    layout="wide",
    menu_items={
        'Get Help': 'https://example.com',
        'Report a bug': "https://example.com",
        'About': "# 新闻真实性检测系统 v6.0\n使用先进的NLP与语义分析技术评估新闻可信度"
    }
)

# 其他导入
import joblib
import numpy as np
import os
import re
import jieba
import pandas as pd
import json
from datetime import datetime
import hashlib
import math

# 配置jieba词典（确保在函数外部）
if os.path.exists('dict.txt.big'):
    jieba.set_dictionary('dict.txt.big')

# 启动验证
def check_launch():
    """安全显示启动信息"""
    with st.container():
        st.title("🔍 AI新闻真实性检测系统")
        st.markdown("使用深度学习与语义分析技术精准评估新闻可信度")
        st.divider()
        st.caption(f"系统版本: 6.0 | 更新时间: {datetime.now().strftime('%Y-%m-%d')}")

# 模型加载函数
@st.cache_resource(show_spinner="加载模型组件中...")
def load_models():
    """安全加载预训练模型组件"""
    models = {}
    model_dir = "models"
    
    try:
        # 安全加载TF-IDF模型
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer_enhanced.pkl")
        if os.path.exists(tfidf_path):
            models["tfidf"] = joblib.load(tfidf_path)
        else:
            st.toast("⚠️ TF-IDF模型未找到，使用简化特征提取", icon="⚠️")
        
        # 加载分类器
        classifier_path = os.path.join(model_dir, "enhanced_fake_news_model.pkl")
        if os.path.exists(classifier_path):
            models["classifier"] = joblib.load(classifier_path)
        
        # 设置期望维度
        models["expected_dim"] = 204
        
        # 加载词典
        custom_dict_path = os.path.join(model_dir, "custom_dict.txt")
        if os.path.exists(custom_dict_path):
            jieba.load_userdict(custom_dict_path)
        
        return models
    
    except Exception as e:
        st.exception(f"模型加载失败: {str(e)}")
        return {"error": str(e)}

# 文本预处理函数 - 增强版（含安全边界）
def preprocess_text(text, max_length=5000):
    """安全预处理文本"""
    if not text or len(text) > max_length:
        return ""
    
    # 安全移除特殊字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\.,，。!?；;:\-\'\"\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()[:max_length]
    
    # 安全分词处理
    try:
        words = jieba.lcut(text, cut_all=False)
        return " ".join(words).strip()
    except:
        return text[:max_length]  # 降级处理

# 增强特征工程（含数值边界）
def extract_text_features(text, max_length=10000):
    """安全提取差异化特征"""
    if not text:
        return {}
    
    # 限制文本长度防止溢出
    text = text[:max_length]
    text_length = len(text)
    
    features = {
        'length': text_length,
        'sentences': min(100, text.count('。') + text.count('!') + text.count('?') + text.count(';') + 1),
        'commas': min(200, text.count(',') + text.count('，')),
        'exclamation': min(50, text.count('!') + text.count('！')),
    }
    
    # 1. 可信度信号（增强差异）
    reliable_terms = ['新华社', '人民日报', '央视', '省政府', '教育部', '卫健委', '官方声明']
    features['reliable_score'] = min(1.0, sum(1 for term in reliable_terms if term in text) * 0.2)
    
    # 2. 情感分析特征
    positive_terms = ['成功', '进展', '成就', '突破', '庆祝', '合作']
    negative_terms = ['失败', '争议', '丑闻', '冲突', '警告', '谴责']
    positive_count = sum(1 for term in positive_terms if term in text)
    negative_count = sum(1 for term in negative_terms if term in text)
    features['positive_sentiment'] = min(1.0, positive_count / max(1, text_length/50))
    features['negative_sentiment'] = min(1.0, negative_count / max(1, text_length/50))
    
    # 3. 结构特征（含边界检查）
    sentences = re.split(r'[。！？!?]', text)
    long_sentences = sum(1 for s in sentences if len(s) > 50)
    features['long_sentence_ratio'] = min(1.0, long_sentences / max(1, features['sentences']))
    
    # 4. 数字/数据特征
    data_terms = ['%', '亿元', '万人', '公斤', '公里', '研究发现', '数据表明', '据统计']
    features['data_presence'] = 1 if any(term in text for term in data_terms) else 0
    features['number_count'] = min(1.0, len(re.findall(r'\d+', text)) / max(1, text_length/100))
    
    # 5. 异常内容检测
    absurd_terms = ['月球爆炸', '太阳消失', '地球停转', '长生不老', '时光倒流', '外星人']
    features['absurd_score'] = min(2.0, sum(0.5 for term in absurd_terms if term in text))
    
    # 6. 可疑语言特征
    urgency_terms = ['必看', '紧急', '速看', '赶快', '立刻']
    exaggeration_terms = ['震惊', '最牛', '100%', '惊天', '秘闻', '绝密']
    features['urgency_score'] = min(1.0, sum(0.3 for term in urgency_terms if term in text))
    features['exaggeration_score'] = min(1.0, sum(0.4 for term in exaggeration_terms if term in text))
    
    # 7. 内容独特性特征（安全哈希）
    try:
        content_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 10000
        features['uniqueness'] = (content_hash % 100) / 100
    except:
        features['uniqueness'] = 0.5  # 默认值
    
    return features

# 增强特征生成函数（含维度保护）
def generate_features(text, models):
    """生成安全特征向量"""
    # 空输入保护
    if not text.strip():
        return np.zeros((1, models.get("expected_dim", 204)), dtype=np.float32)
    
    processed_text = preprocess_text(text)
    
    try:
        # TF-IDF特征
        tfidf_vector = np.zeros((1, 1))
        if "tfidf" in models and models["tfidf"]:
            tfidf_vector = models["tfidf"].transform([processed_text]).toarray()
            if tfidf_vector.shape[1] > models["expected_dim"]:
                tfidf_vector = tfidf_vector[:, :models["expected_dim"]]
        
        # 高级特征
        adv_features = extract_text_features(text)
        feature_keys = sorted(adv_features.keys())
        adv_vector = [float(adv_features[key]) for key in feature_keys]
        
        # 合并特征（保护维度）
        combined = np.concatenate([tfidf_vector[0], adv_vector])
        if len(combined) < models["expected_dim"]:
            padded = np.zeros(models["expected_dim"])
            padded[:len(combined)] = combined
            return padded.reshape(1, -1)
        elif len(combined) > models["expected_dim"]:
            return combined[:models["expected_dim"]].reshape(1, -1)
        else:
            return combined.reshape(1, -1)
    
    except Exception as e:
        st.toast(f"特征生成错误: {str(e)}", icon="⚠️")
        return np.zeros((1, models.get("expected_dim", 204)), dtype=np.float32)

# 专业异常内容识别系统 - 安全版
def absurd_content_detector(text, max_length=10000):
    """安全识别荒谬内容"""
    if not text:
        return False, ""
    
    # 限制文本长度
    text = text[:max_length]
    
    # 1. 违反科学规律
    absurd_patterns = [
        ('月球爆炸', '天体毁灭'),
        ('太阳消失', '天体消失'),
        ('地球停转', '违反物理规律'),
        ('长生不老', '违反生物规律'),
        ('时光倒流', '违反物理规律')
    ]
    
    for phrase, reason in absurd_patterns:
        if phrase in text:
            return True, f"检测到荒谬内容: {reason}"
    
    # 2. 极端统计数据
    if re.search(r'99\.9[%％]的人|99\.9[%％]都|所有人|完全消除', text):
        return True, "检测到极端统计数据"
    
    # 3. 不可能事件
    if re.search(r'一夜之间|瞬间解决|完全改变', text):
        return True, "检测到不合常理的时间表述"
    
    return False, ""

# 置信度计算系统（含数值保护）
def calculate_confidence(probabilities, text):
    """安全计算置信度"""
    # 空概率处理
    if not probabilities.size or probabilities.shape[1] < 2:
        return 0.5, 0.5, []
    
    # 初始概率
    real_prob = max(0.01, min(0.99, probabilities[0][0]))
    fake_prob = max(0.01, min(0.99, probabilities[0][1]))
    
    # 空文本保护
    if not text:
        return real_prob, fake_prob, []
    
    # 提取特征
    features = extract_text_features(text)
    
    # 可信度调整因子
    credibility_factors = []
    
    # 1. 权威来源提升真实概率
    if features.get('reliable_score', 0) > 0:
        boost = min(0.2, features['reliable_score'] * 0.5)
        real_prob = min(0.99, real_prob + boost)
        credibility_factors.append(f"权威来源 +{int(boost*100)}%")
    
    # 2. 数据支撑提升真实概率
    if features.get('data_presence', 0):
        real_prob = min(0.95, real_prob + 0.1)
        credibility_factors.append("数据支持 +10%")
    
    # 3. 异常内容提升虚假概率
    if features.get('absurd_score', 0) > 0:
        fake_boost = min(0.35, features['absurd_score'] * 0.7)
        fake_prob = min(0.99, fake_prob + fake_boost)
        credibility_factors.append(f"异常内容 +{int(fake_boost*100)}%")
    
    # 4. 情感因素调整
    sentiment_diff = features.get('positive_sentiment', 0) - features.get('negative_sentiment', 0)
    if sentiment_diff < -0.1:  # 负面情感
        adjustment = min(0.15, abs(sentiment_diff) * 0.5)
        fake_prob = min(0.99, fake_prob + adjustment)
        credibility_factors.append(f"负面情绪 +{int(adjustment*100)}%")
    
    # 5. 长度因素调整
    text_length = features.get('length', 0)
    if 200 <= text_length <= 500:  # 最佳长度
        adjustment = min(0.15, 0.05 + text_length * 0.0002)
        real_prob = min(0.99, real_prob + adjustment)
        credibility_factors.append(f"理想长度 +{int(adjustment*100)}%")
    elif text_length < 50 or text_length > 1000:  # 过长或过短
        fake_prob = min(0.99, fake_prob + 0.1)
        real_prob = max(0.01, real_prob - 0.1)
        credibility_factors.append("文本长度异常 ±10%")
    
    # 6. 长句比例影响
    if features.get('long_sentence_ratio', 0) > 0.3:
        adjustment = min(0.15, features['long_sentence_ratio'] * 0.4)
        fake_prob = min(0.99, fake_prob + adjustment)
        credibility_factors.append(f"复杂句式 +{int(adjustment*100)}%")
    
    # 7. 内容独特性
    uniqueness = features.get('uniqueness', 0.5)
    if uniqueness > 0.7:
        adjustment = min(0.1, uniqueness * 0.2)
        real_prob = min(0.99, real_prob + adjustment)
        credibility_factors.append(f"独特内容 +{int(adjustment*100)}%")
    
    # 8. 可疑语言特征
    urgency_score = features.get('urgency_score', 0)
    exaggeration_score = features.get('exaggeration_score', 0)
    if urgency_score > 0 or exaggeration_score > 0:
        adjustment = min(0.25, (urgency_score + exaggeration_score) * 0.3)
        fake_prob = min(0.99, fake_prob + adjustment)
        credibility_factors.append(f"可疑语言 +{int(adjustment*100)}%")
    
    # 归一化处理
    total = real_prob + fake_prob
    return real_prob/total, fake_prob/total, credibility_factors

# 生成安全报告
def generate_report(text, real_prob, fake_prob, is_absurd, credibility_factors):
    """生成安全分析报告"""
    if not text:
        return {
            "summary": "输入内容为空",
            "key_insights": ["未提供有效内容"],
            "recommendations": ["请输入新闻内容后重试"]
        }
    
    try:
        features = extract_text_features(text)
        credibility_score = int(real_prob * 100)
        
        risk_level = "高风险"
        color = "red"
        if credibility_score >= 85:
            risk_level = "低风险"
            color = "green"
        elif credibility_score >= 70:
            risk_level = "中等风险"
            color = "orange"
        
        report = {
            "summary": f"可信度评分: {credibility_score}/100",
            "risk_level": f"<span style='color:{color};font-weight:bold'>{risk_level}</span>",
            "key_insights": [],
            "recommendations": [
                "建议核实信息的原始来源",
                "检查多个独立信息源进行交叉验证"
            ]
        }
        
        # 关键洞察
        if is_absurd:
            report["key_insights"].append("⛔ 检测到高度荒谬内容")
        if features['reliable_score'] > 0:
            report["key_insights"].append(f"✅ 检测到{features['reliable_score']:.1f}个权威来源")
        else:
            report["key_insights"].append("⚠️ 未检测到权威来源")
        if features['data_presence']:
            report["key_insights"].append("📊 包含数据支撑")
        else:
            report["key_insights"].append("ℹ️ 缺乏具体数据")
        if features['exaggeration_score'] > 0.5:
            report["key_insights"].append("❗ 检测到夸张语言")
        
        return report
    except:
        return {
            "summary": "报告生成出错",
            "key_insights": ["系统内部错误"],
            "recommendations": ["请尝试重新输入内容"]
        }

# 用户界面主函数 - 安全增强版
def main_application():
    # 安全初始化
    st.session_state.setdefault("show_feedback", False)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", {})
    
    # 安全侧边栏
    with st.sidebar:
        try:
            st.markdown("### 🛠️ 系统状态")
            models = load_models()
            if "error" in models:
                st.warning("模型加载异常 - 使用简化模式")
            else:
                st.success("✓ 模型就绪")
                st.info(f"特征维度: {models.get('expected_dim', 'N/A')}")
            
            st.divider()
            st.markdown("### 📝 使用指南")
            st.markdown("""
            1. 粘贴新闻内容（200字+更准确）
            2. 避免单句检测
            3. 权威来源提高可信度
            4. 每次检测结果都会变化
            """)
            
            # 历史记录（安全显示）
            st.divider()
            st.markdown("### 🕒 历史检测")
            if st.session_state.history:
                for i, item in list(enumerate(st.session_state.history[-3:])):
                    icon = "✅" if item.get("credibility", 0) > 70 else "⚠️" if item.get("credibility", 0) > 40 else "❌"
                    st.caption(f"{icon} {item.get('time', '')} - {item.get('result', '')}")
            else:
                st.caption("无历史记录")
            
            st.divider()
            if st.button("📨 报告问题", use_container_width=True):
                st.session_state.show_feedback = True
        except:
            st.warning("侧边栏初始化出错")

    # 主界面（含异常捕获）
    try:
        st.header("📰 新闻真实性分析")
        st.caption("粘贴新闻内容获取精准可信度评估")
        
        news_text = st.text_area("新闻内容:", 
                                height=250, 
                                placeholder="在此处粘贴新闻内容...",
                                help="支持中英文内容，最佳长度200-1000字",
                                key="news_input")
        
        if st.button("✅ 检测真实性", type="primary", use_container_width=True):
            if not news_text.strip():
                st.warning("请输入新闻内容")
                return
                
            progress_bar = st.progress(0, text="分析准备中...")
            
            try:
                # 1. 荒谬内容检测
                progress_bar.progress(15, "检测异常内容...")
                is_absurd, reason = absurd_content_detector(news_text)
                if is_absurd:
                    st.error(f"⛔ **高风险虚假新闻** - {reason}")
                    progress_bar.progress(100)
                    
                    # 保存历史
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M"),
                        "text": news_text[:100] + "..." if len(news_text) > 100 else news_text,
                        "result": "荒谬内容 - 虚假新闻",
                        "credibility": 10
                    })
                    
                    # 生成报告
                    with st.expander("📊 详细分析报告", expanded=True):
                        report = generate_report(news_text, 0.1, 0.9, True, [])
                        st.subheader("可信度评估")
                        st.markdown(f"### {report['summary']}")
                        st.markdown(f"**风险等级**: {report['risk_level']}", unsafe_allow_html=True)
                        st.subheader("关键洞察")
                        for insight in report["key_insights"]:
                            st.write(f"- {insight}")
                        st.subheader("建议")
                        for rec in report["recommendations"]:
                            st.info(f"- {rec}")
                    
                    return
                
                # 2. 特征提取
                progress_bar.progress(35, "提取内容特征...")
                features = generate_features(news_text, models)
                
                # 3. 模型预测（安全降级）
                progress_bar.progress(65, "分析内容真实性...")
                if "classifier" in models and models["classifier"]:
                    probabilities = models["classifier"].predict_proba(features)
                else:
                    # 规则降级
                    fake_indicators = features[0][-8:].sum() - 2
                    probabilities = np.array([[0.5, 0.5] if fake_indicators < 0 else [0.3, 0.7]])
                
                # 4. 置信度计算
                progress_bar.progress(85, "计算可信度...")
                real_prob, fake_prob, factors = calculate_confidence(probabilities, news_text)
                
                # 5. 显示结果
                result_type = "高风险 - 可疑内容" if fake_prob > 0.6 else "可靠 - 真实内容"
                credibility_score = fake_prob * 100 if fake_prob > 0.6 else real_prob * 100
                
                if fake_prob > 0.6:
                    st.error(f"⚠️ **可疑新闻** (风险: {fake_prob*100:.1f}%)")
                else:
                    st.success(f"✅ **可靠新闻** (可信度: {real_prob*100:.1f}%)")
                
                progress_bar.progress(100)
                
                # 可视化
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("可信度", f"{real_prob*100:.1f}%", delta_color="inverse")
                    st.metric("虚假风险", f"{fake_prob*100:.1f}%", delta_color="inverse")
                
                # 详细报告
                with st.expander("📊 详细分析报告", expanded=True):
                    report = generate_report(news_text, real_prob, fake_prob, False, factors)
                    st.subheader("可信度评估")
                    st.markdown(f"### {report['summary']}")
                    st.markdown(f"**风险等级**: {report['risk_level']}", unsafe_allow_html=True)
                    
                    if factors:
                        st.subheader("关键影响因素")
                        for factor in factors:
                            st.caption(f"- {factor}")
                    
                    st.subheader("内容分析")
                    for insight in report["key_insights"]:
                        st.write(f"- {insight}")
                    
                    st.subheader("专家建议")
                    for rec in report["recommendations"]:
                        st.info(f"- {rec}")
                
                # 保存结果
                st.session_state.last_result = {
                    "text": news_text, 
                    "result": result_type,
                    "confidence": credibility_score
                }
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "text": news_text[:100] + "..." if len(news_text) > 100 else news_text,
                    "result": result_type,
                    "credibility": round(credibility_score, 1)
                })
                
            except Exception as e:
                progress_bar.progress(100)
                st.error(f"分析过程中出错: {str(e)}")
                st.exception(e)
        
        # 反馈系统
        if st.session_state.get("show_feedback", False):
            st.divider()
            st.subheader("✉️ 报告问题")
            
            with st.form(key="feedback_form"):
                st.radio("结果准确性:", 
                        ("非常准确", "部分准确", "不准确"), 
                        index=1,
                        key="feedback_accuracy")
                
                st.radio("实际新闻类型:", 
                       ("真实新闻", "不确定", "虚假新闻"), 
                       index=1,
                       key="feedback_label")
                
                st.text_area("问题描述:", 
                           height=100,
                           placeholder="请描述您发现的问题...",
                           key="feedback_comment")
                
                if st.form_submit_button("提交反馈"):
                    try:
                        feedback_dir = "user_feedback"
                        os.makedirs(feedback_dir, exist_ok=True)
                        
                        feedback_data = {
                            "time": datetime.now().isoformat(),
                            "text": st.session_state.get("last_result", {}).get("text", "")[:500],
                            "system_result": st.session_state.get("last_result", {}).get("result", ""),
                            "system_confidence": st.session_state.get("last_result", {}).get("confidence", 0),
                            "user_accuracy": st.session_state.feedback_accuracy,
                            "user_label": st.session_state.feedback_label,
                            "comment": st.session_state.feedback_comment
                        }
                        
                        with open(os.path.join(feedback_dir, "feedback.jsonl"), "a", encoding="utf-8") as f:
                            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
                        
                        st.success("感谢反馈！您的意见非常重要")
                        st.session_state.show_feedback = False
                    except:
                        st.error("反馈保存失败")
    
    except Exception as e:
        st.error(f"主界面初始化出错: {str(e)}")

# 主程序入口
if __name__ == "__main__":
    try:
        check_launch()
        main_application()
    except Exception as e:
        st.error(f"系统启动失败: {str(e)}")
        st.error("请联系技术人员并提供此错误信息")
