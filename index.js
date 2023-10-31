import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import {
  JSONLoader,
  JSONLinesLoader,
} from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { OpenAIEmbeddings } from "langchain/embeddings/openai";

import { PineconeClient } from "@pinecone-database/pinecone";
import * as dotenv from "dotenv";
import { Document } from "langchain/document";
import { PineconeStore } from "langchain/vectorstores/pinecone";

import { ConversationalRetrievalQAChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";
import { BufferMemory } from "langchain/memory";
import express from "express";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.post("/api/answer", async (req, res) => {
  try {
    const memoryUsedBeforeModel = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log(`Memory used before creating model: ${memoryUsedBeforeModel} MB`);

    const model = new OpenAI({ temperature: 0 });

    const memoryUsedBeforeClient = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log(`Memory used before creating PineconeClient: ${memoryUsedBeforeClient} MB`);

    const client = new PineconeClient();
    await client.init({
      apiKey: process.env.PINECONE_API_KEY,
      environment: process.env.PINECONE_ENVIRONMENT,
    });
    const pineconeIndex = client.Index(process.env.PINECONE_INDEX);

    // ... (rest of your logic)
    const loader = new DirectoryLoader("./docs", {
      ".json": (path) => new JSONLoader(path, "/texts"),
      ".jsonl": (path) => new JSONLinesLoader(path, "/html"),
      ".txt": (path) => new TextLoader(path),
      ".csv": (path) => new CSVLoader(path, "text"),
      ".docx": (path) => new DocxLoader(path),
      ".pdf": (path) => new PDFLoader(path),
    });
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 75,
    });

    const output = await splitter.splitDocuments(docs);

    const embeddings = new OpenAIEmbeddings();

    const vectorStore = await PineconeStore.fromDocuments(output, embeddings, {
      pineconeIndex,
    });

    const question = req.body.question;
    const results = await vectorStore.similaritySearch(question);

    const vectorStoreRetriever = vectorStore.asRetriever();

    const chain = ConversationalRetrievalQAChain.fromLLM(
      model,
      vectorStoreRetriever
    );
    const chat_history = req.body.chat_history || [];
    const answer = await chain.call({ question, chat_history });
    chat_history.push({ question, answer: answer.text });

    const memoryUsedBeforeAnswer = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log(`Memory used before sending the answer: ${memoryUsedBeforeAnswer} MB`);

    res.json({ answer: answer.text, chat_history });
    
    const memoryUsed = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log(`The current heap memory usage is approximately ${memoryUsed} MB.`);
  } catch (error) {
    console.log("error", error);
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
