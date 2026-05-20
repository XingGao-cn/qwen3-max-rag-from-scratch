# build RAG from scratch

## 项目概述

本项目是一个基于 LangChain 的 RAG（Retrieval-Augmented Generation，检索增强生成）示例。Notebook 以 qwen3-max 作为生成模型，以 DashScope OpenAI-compatible 接口调用 text-embedding-v4 作为向量嵌入模型，并使用 FAISS 构建本地向量库。整体流程参考 LangChain 的 RAG from Scratch 1–4 教程，将完整 RAG 过程拆解为 Overview、Indexing、Retrieval、Generation 和 Full RAG chain 五个模块。

该项目的目标不是只展示RAG与API模型调用，而是完整展示从网页数据加载、文本切分、向量化、向量库构建、语义检索、Prompt 组织，到大模型生成答案的完整数据链路。

每个模块均包含对应的jupyter代码以及.py脚本.


## 环境构建

建议单独创建一个名为 rag_qwen3max 的 conda 环境，并使用 Python 3.10。避免直接在 base 环境中运行该项目，避免和已有机器学习、可视化、RAG 或大模型工具包产生版本冲突。

环境构建建议分为四步：

第一步，创建并激活 conda 环境。该环境用于隔离本项目所需依赖，保证 Notebook 的 Python Kernel、LangChain、FAISS 和向量模型调用处于同一运行环境。

第二步，更新 pip、setuptools 和 wheel。这一步用于避免老版本构建工具导致依赖安装失败。

第三步，安装数值计算和向量检索依赖。项目使用 numpy 1.26.4，并使用 faiss-cpu 1.8.0 构建本地向量索引。由于 FAISS 在不同平台上的 pip 兼容性不完全一致，推荐通过 conda-forge 安装 faiss-cpu。

第四步，安装 LangChain 生态相关依赖。核心依赖包括 langchain、langchain-core、langchain-community、langchain-openai、langchain-text-splitters、tiktoken、beautifulsoup4 和 requests。Notebook 运行时还需要 ipykernel 和 jupyter，并需要将 rag_qwen3max 注册为 Jupyter Kernel。

完成环境创建后，在 Notebook 界面中选择 Python (rag_qwen3max) 作为 Kernel。环境是否选择正确，主要看 Notebook 中的 Python executable 是否指向 rag_qwen3max 环境路径。

## 主要依赖及作用

Python 3.10：作为基础解释器版本，兼顾 LangChain、FAISS、tiktoken 等依赖的兼容性。

numpy 1.26.4：用于向量计算，例如余弦相似度计算。

faiss-cpu 1.8.0：用于构建本地向量库，并根据问题向量检索最相似的文档块。

langchain 0.3.27：提供 RAG 流程编排、Prompt 模板、Runnable 链式组合等基础能力。

langchain-core 0.3.80：提供 Document、Runnable、输出解析器等核心抽象。

langchain-community 0.3.31：提供 WebBaseLoader、FAISS vectorstore 等社区集成组件。

langchain-openai 0.3.35：通过 OpenAI-compatible 接口调用 DashScope 的 qwen3-max 和 text-embedding-v4。

langchain-text-splitters 0.3.11：提供 RecursiveCharacterTextSplitter，用于将网页长文本切分为可检索的文档块。

tiktoken 0.8.0：用于 token 统计，并在文本切分时按 token 粒度控制 chunk 大小。

beautifulsoup4 4.12.3：用于解析网页 HTML，并从博客页面中提取标题、正文和头部内容。

requests：用于网页请求和底层 HTTP 访问。

ipykernel 与 jupyter：用于将 conda 环境注册为 Notebook Kernel，并运行 .ipynb 文件。

## 模型与外部服务

生成模型使用 qwen3-max。该模型通过 DashScope 的 OpenAI-compatible base_url 接入，负责根据检索到的上下文生成最终答案。

向量模型使用 text-embedding-v4。该模型同样通过 DashScope OpenAI-compatible 接口调用，负责把用户问题和网页文本块编码为向量。

LangSmith / LangChain tracing 用于记录链路执行过程，便于调试和观察 RAG 中的输入、检索结果、Prompt 和模型输出。Notebook 中配置了 tracing 相关变量，因此运行后可以在 LangSmith 项目中查看执行轨迹。

安全提醒：Notebook 中如果写入了真实 API Key，不建议将 Notebook 直接上传到公开仓库。公开分享前应删除密钥，并在平台后台轮换已暴露的 Key。

## 代码模块逻辑

### 0. 全局配置模块

全局配置模块负责导入依赖、设置 DashScope 和 LangSmith 相关环境变量、指定模型名称、指定 embedding 维度、指定博客 URL 和默认问题。

该模块中最关键的配置包括生成模型 qwen3-max、向量模型 text-embedding-v4、向量维度 1024、博客 URL，以及默认问题 What is Task Decomposition?。

### 1. Overview 模块

Overview 模块用于解释 RAG 的基础组成，包括 token 统计、向量嵌入和余弦相似度。

该模块会统计问题文本的 token 数量，将问题和一段示例文档分别编码成向量，然后计算两者的余弦相似度。这个模块的作用是帮助理解“为什么相似语义文本可以通过向量距离被检索出来”。

在完整 RAG 中，这一思想会被进一步扩展为：用户问题被编码为问题向量，网页文档块被编码为文档向量，两者通过向量相似度完成语义检索。

### 2. Indexing 模块

Indexing 是 RAG 的知识库构建阶段。该阶段不直接生成答案，而是负责把外部网页数据加工成可检索的向量库。

该模块首先通过 WebBaseLoader 加载 Lilian Weng 的 Agent 博客页面，然后通过 BeautifulSoup 只提取博客标题、头部和正文内容，避免将网页导航栏、页脚和无关 HTML 内容混入知识库。

加载后的网页内容会被转换为 LangChain Document 对象。由于博客正文通常很长，不能直接整体参与检索和生成，因此代码使用 RecursiveCharacterTextSplitter 将网页长文档切分为多个较短的文档块。当前 Notebook 使用较大的 chunk_size，以便每个文档块保留更完整的上下文，并通过 chunk_overlap 减少切分边界导致的信息丢失。

切分后的文档块会被 text-embedding-v4 编码成向量，并写入 FAISS 向量库。FAISS 向量库保存的是“文档块向量 + 文档块原文 + metadata”。它不负责生成答案，只负责后续根据问题向量快速找到相似文档块。

### 3. Retrieval 模块

Retrieval 是 RAG 的检索阶段。该阶段接收用户问题，并从向量库中找出与问题最相关的若干网页片段。

当用户输入问题后，retriever 会先把问题编码成向量，再用该问题向量到 FAISS 中进行相似度搜索。FAISS 会比较问题向量与所有文档块向量之间的距离，并返回 top-k 个最相似的文档块。

Notebook 中的 k 控制每次检索返回多少个相关文档块。k 越大，模型看到的网页上下文越多；但同时 Prompt 会更长，调用成本和推理时间也会增加。如果要回答更复杂的问题，可以适当提高 k。

### 4. Generation 模块

Generation 是“显式上下文生成”阶段。该阶段不重新检索，而是使用 Retrieval 阶段已经得到的 retrieved_docs。

首先，代码会将 retrieved_docs 格式化成 context 字符串。这个 context 是大模型真正能看到的外部知识内容。然后，Prompt 模板会把 context 和 question 合并为大模型输入。Prompt 的作用是约束 qwen3-max 只能根据给定上下文回答，如果上下文中没有答案，就应说明无法从上下文判断。

最后，qwen3-max 接收完整 Prompt，并生成回答。生成结果经过 StrOutputParser 转换为普通字符串，方便输出和后续处理。

这一模块适合调试，因为它可以清楚地看到检索出来的上下文是什么，以及这些上下文如何被传入模型。

### 5. Full RAG Chain 模块

Full RAG Chain 是完整端到端链路。它将 Retrieval 和 Generation 串联起来，让用户只需要输入问题，即可自动完成检索、上下文拼接、Prompt 构造和答案生成。

在该链路中，用户问题会被分成两条数据路径：

第一条路径用于检索。问题进入 retriever，retriever 根据问题向量从 FAISS 中找出相关文档块，然后将这些文档块格式化为 context。

第二条路径用于生成。原始问题通过 RunnablePassthrough 原样保留，并作为 question 输入 Prompt。

两条路径汇合后，Prompt 同时接收 context 和 question，随后将格式化结果传给 qwen3-max，最终由 StrOutputParser 输出字符串答案。

## RAG 数据流程

### Indexing 数据流

网页 URL 进入 WebBaseLoader 后被解析为原始 Document。Document 中包含网页正文和 metadata 信息。

原始 Document 进入文本切分器后被切分为多个较小的 Document chunks。每个 chunk 保留一段网页内容，并继承来源 URL 等 metadata。

Document chunks 进入 embedding 模型后，每个 chunk 被转换为固定维度向量。当前项目中向量维度为 1024。

chunk 向量和对应文本被写入 FAISS。至此，网页数据完成从 HTML 到向量库的转换。

### Retrieval 数据流

用户问题首先进入 embedding 模型，被转换为问题向量。

问题向量进入 FAISS 检索，与所有文档块向量进行相似度比较。

FAISS 返回最相关的 top-k 文档块。这些文档块构成后续生成阶段的外部知识来源。

### Generation 数据流

检索到的文档块进入 format_docs，被转换为纯文本 context。

context 与用户问题一起填入 Prompt 模板。Prompt 由任务指令、检索上下文和用户问题三部分组成。

格式化后的 Prompt 被发送给 qwen3-max。qwen3-max 不直接访问网页，也不直接访问 FAISS；它只能看到 Prompt 中包含的 context 和 question。

qwen3-max 生成回答后，输出解析器将模型消息转换为普通字符串，作为最终答案。

## 参数理解与调整建议

embedding 维度影响向量表达能力、索引大小和检索速度。1024 是 text-embedding-v4 的常用默认配置，适合当前项目。

chunk_size 决定每个文档块包含多少文本。更大的 chunk_size 可以保留更长上下文，适合总结类或复杂问答；更小的 chunk_size 检索更精细，但可能丢失跨段信息。

chunk_overlap 决定相邻文档块之间的重叠量。适当重叠可以避免重要句子被切断，但过大会增加冗余。

retriever 的 k 决定每个问题返回多少个文档块。k 越大，大模型看到的上下文越多；但 Prompt 越长，成本和响应时间也会提高。

temperature 控制 qwen3-max 输出的随机性。RAG 问答推荐使用较低 temperature，以保证答案稳定并减少幻觉。

max_tokens 控制模型最多输出多少 token。若需要更完整的长答案或一次回答多个问题，可以适当提高该值。

timeout 控制 API 最长等待时间。若检索上下文较长或输出较长，建议同步提高 timeout。

max_retries 控制请求失败后的重试次数。网络不稳定或服务偶发超时时，可以适当增加。

## 构建过程总结

本项目的构建过程可以概括为：先构建知识库，再基于知识库回答问题。

知识库构建阶段从博客网页开始，经过网页加载、正文提取、文本切分、向量化和 FAISS 入库，形成可检索的本地向量库。

问答阶段从用户问题开始，经过问题向量化、FAISS 检索、上下文拼接、Prompt 构造和 qwen3-max 生成，得到基于网页内容的回答。

这种设计将“知识检索”和“语言生成”分离开来。FAISS 负责找到相关资料，Prompt 负责组织资料和问题，qwen3-max 负责基于资料生成自然语言答案。相比直接向大模型提问，RAG 可以让答案更贴近指定网页内容，并降低无依据生成的风险。

## 运行顺序建议

在 Notebook 中建议按以下顺序执行：

首先确认 Kernel 使用 rag_qwen3max 环境。然后运行环境检查单元，确认 Python executable 指向正确 conda 环境。接着运行全局配置和导入模块。随后依次运行 Overview、Indexing、Retrieval、Generation 和 Full RAG chain 模块。

如果修改了 chunk_size、chunk_overlap、embedding 维度或网页数据源，应重新运行 Indexing 模块并重建 FAISS 向量库。如果只修改问题或 Prompt，可以从 Retrieval 或 Generation 模块重新运行。

## 常见问题

如果 Notebook 中显示 Conda env 为 None，但 Python executable 指向 rag_qwen3max 环境路径，通常说明 Kernel 已经选对，只是 CONDA_DEFAULT_ENV 环境变量没有传入，不影响运行。

如果 LangChain Hub 拉取公共 Prompt 失败，可以使用 Notebook 中的本地 Prompt。该问题只影响 Prompt 来源，不代表 qwen3-max 没有被调用。

如果想确认 qwen3-max 是否被调用，可以观察 Generation 或 Full RAG 模块是否成功输出模型生成答案。如果出现鉴权失败、模型不存在或网络超时，则需要检查 DashScope API Key、模型名、base_url 和网络环境。

如果检索结果与问题关系不强，应优先调整 chunk_size、chunk_overlap 和 k，也可以检查网页加载内容是否包含目标知识点。

## 参考资料

LangChain RAG from Scratch 1–4 notebook: https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_1_to_4.ipynb

LangChain RAG documentation: https://docs.langchain.com/oss/python/langchain/rag

OpenAI Cookbook tiktoken token counting example: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

Lilian Weng Agent Blog: https://lilianweng.github.io/posts/2023-06-23-agent/
