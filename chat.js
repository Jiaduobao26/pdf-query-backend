// 用于把文本按字符递归切分成小块，便于后续处理。
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// 用于生成文本的向量嵌入（embedding），通常用于语义搜索或相似度计算。
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
// 在内存中存储和检索向量数据的工具，常用于小规模的向量检索任务。
import { MemoryVectorStore } from "langchain/vectorstores/memory";
// 用于构建检索式问答链，将检索和问答模型结合起来。
import { RetrievalQAChain } from "langchain/chains";
// 用于与 OpenAI 的聊天模型（如 GPT-3/4）进行交互。
import { ChatOpenAI } from "langchain/chat_models/openai";
// 用于构建和管理 prompt 模板，方便与大模型交互时动态生成 prompt。
import { PromptTemplate } from "langchain/prompts";
// 用于加载 PDF 文件并将其内容提取为文本，便于后续处理。
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

const chat = async (filePath = "./uploads/hbs-lean-startup.pdf", query) => {
  // step 1:
  const loader = new PDFLoader(filePath);
  const data = await loader.load();
  // step 2:
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500, // (in terms of number of characters) 
    chunkOverlap: 0,
  });
  const splitDocs = await textSplitter.splitDocuments(data);
  // step 3
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.REACT_APP_OPENAI_API_KEY,
  });
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
  // step 4: retrieval
  // const relevantDocs = await vectorStore.similaritySearch(
  // "What is task decomposition?"
  //);
  // step 5: qa w/ customzie the prompt
  // 创建一个 OpenAI 聊天大模型（如 gpt-3.5-turbo）的实例，并配置好 API 密钥。
  // 这样后续就可以用这个 model 对象，向 OpenAI 的大模型发送问题和上下文，让它生成智能回答。
  // 简单说，就是“准备好一个可以和 OpenAI 聊天模型对话的接口”。
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.REACT_APP_OPENAI_API_KEY,
  });
  // template 是一个 prompt 模板，用来告诉大语言模型（如 GPT-3.5-turbo）如何回答用户问题。
  // 它会把检索到的相关内容（{context}）和用户的问题（{question}）插入到模板里，引导模型用简洁、准确的方式作答，并且如果不知道答案就直接说不知道。
  // 这样可以让模型输出更符合预期的风格和要求。
  const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}
Question: {question}
Helpful Answer:`;

  // 创建一个“检索增强问答链”（RetrievalQAChain）对象。
  // 它把大语言模型（model）、向量检索器（vectorStore.asRetriever()）和 prompt 模板（template）组合起来。
  // 这样，后续只需要调用 chain.call({ query })，就能自动完成“检索相关内容 + 让大模型基于这些内容回答问题”的整个流程。
  /**
   * 创建 chain 时（RetrievalQAChain.fromLLM），
   * 传入了 model、retriever（vectorStore.asRetriever()）和 prompt 模板。
   * 此时只是把这些组件组合起来，方便后续统一调用。
   * 
   * 真正“把 prompt 结合起来、检索 context 并生成答案”是在调用 chain.call({ query }) 这一行时发生的
   **/
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
    prompt: PromptTemplate.fromTemplate(template),
    // returnSourceDocuments: true,
  });

  /**
   * chain.call({ query }) 被调用时，
   * RetrievalQAChain 会先用 retriever
   * （也就是 vectorStore.asRetriever()）根据 query 检索出最相关的内容（context）。
   * 检索到 context 和用户的 query 会被一起填充进定义的 prompt 模板（PromptTemplate.fromTemplate(template)）。
   * 填充好的 prompt 会被送到大语言模型（model）去生成最终的回答（response）。
   */
  const response = await chain.call({
    query,
  });
  return response;
};

export default chat;
