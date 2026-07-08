# 项目展示说明：RAG智能问答系统评测与自动化测试框架

## 项目定位

本项目面向 **大模型测试工程师、AI测试开发工程师、RAG评测工程师、算法测试工程师** 等岗位，展示如何对RAG问答系统进行自动化评测、回归测试和失败案例分析。

## 可写入简历的项目名称

**RAG智能问答系统评测与自动化测试框架**

## 简历描述示例

基于Python构建面向客户支持知识库的RAG问答系统与自动化评测框架，实现文档加载、文本切分、向量检索、证据抽取式回答、评测集批量测试、来源匹配、关键词召回、上下文召回、回答依据性、幻觉风险代理指标、不可回答问题安全拒答、失败案例分类与pytest回归测试。项目支持Streamlit演示、Docker部署和GitHub Actions持续集成。

## 对应岗位关键词

- 大模型测试工程师
- AI测试开发工程师
- RAG评测工程师
- 算法测试工程师
- 大模型评测
- 幻觉检测
- 不可回答问题测试
- 自动化回归测试
- Python / Pytest / CI/CD / Docker
- LangChain / RAG / 向量数据库 / 检索评测

## 技术亮点

1. **RAG评测闭环**：从知识库构建到评测结果输出，形成完整评测流程。
2. **不可回答问题安全处理**：当问题中的关键数字、付款方式、国际范围等信息未出现在检索上下文中时，系统拒绝猜测。
3. **可解释失败分析**：将失败分为检索失败、上下文召回失败、回答关键词缺失、依据性不足、安全拒答失败等类型。
4. **测试开发能力展示**：使用pytest测试核心模块，并通过GitHub Actions实现持续集成。
5. **工程化部署能力**：提供Streamlit交互界面和Dockerfile。

## 面试讲解顺序

1. 为什么RAG系统需要专门评测：检索错误、幻觉、上下文不完整、不可回答问题。
2. 项目架构：Documents → Chunks → Vector Store → Retrieval → Answer → Evaluation → Failed Cases。
3. 关键指标：source match、keyword recall、context keyword recall、answer groundedness、hallucination risk、unanswerable safety。
4. 测试策略：单元测试、回归测试、失败案例分析。
5. 可改进方向：接入RAGAS、DeepEval、OpenCompass或真实LLM API。
