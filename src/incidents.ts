import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { Document } from 'langchain/document';
import jsonData from './incidents.json' with { type: 'json' };
import {
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  ChatPromptTemplate,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnablePassthrough,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { formatDocumentsAsString } from 'langchain/util/document';
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression';
import { EmbeddingsFilter } from 'langchain/retrievers/document_compressors/embeddings_filter';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

const testSimilarity = async () => {
  const embeddings = new OpenAIEmbeddings();
  const baseCompressor = new EmbeddingsFilter({
    embeddings: new OpenAIEmbeddings(),
    similarityThreshold: 0.8,
    k: 50,
  });

  let vectorStore = await HNSWLib.load('./incidents_vector_string', embeddings);

  const retriever = new ContextualCompressionRetriever({
    baseCompressor,
    baseRetriever: vectorStore.asRetriever(),
  });

  // const retrievedDocs = await retriever.getRelevantDocuments(
  //   'Please find all incidents for operator name "Latitude 33"',
  // );
  // console.log({ retrievedDocs });

  // const retrievedDocs = await vectorStore.similaritySearch(
  //   `every Incidents for Hop-A-Jet operator with root cause, date, time, tailnumber,` +
  //     `Departure airport, Destination airport`,

  //   //  (document) => document.metadata.operatorName == '45 North Aviation',
  // );

  const retrievedDocs = await vectorStore.similaritySearch(
    `Most recent date of incident for Hop-A-Jet operator with root cause, tailnumber,` +
      `Departure airport, Destination airport`,

    //  (document) => document.metadata.operatorName == '45 North Aviation',
  );

  const pageContents = retrievedDocs.map((doc) => doc.pageContent);
  console.log(pageContents);
};

const prepareTextVector = async () => {
  const embeddings = new OpenAIEmbeddings();
  let allDocumentString = '';
  for (let i = 0; i < jsonData.length; i++) {
    if (jsonData[i].incidentDetails.length == 0) {
      const details = 'none';
      // allDocumentString += `Operator name:${jsonData[i].operatorName},Tail number:${jsonData[i].tailNumber},Incident Details:[${details}]\n`;
    } else {
      for (let j = 0; j < jsonData[i].incidentDetails.length; j++) {
        let details = '';
        let rootCause = '';
        details += `Operator name: "${jsonData[i].operatorName}",`;
        // details += `"Aircraft Tail number":"${jsonData[i].tailNumber}",`;
        const detailsData = jsonData[i].incidentDetails[j];
        for (const [key, value] of Object.entries(detailsData)) {
          const keyValue = key.toLowerCase();
          const owner = 'Owner/operator'.toLowerCase();
          const operator = 'operator'.toLowerCase();
          if (keyValue == owner || keyValue == operator) {
            //  details += `"Operator Name": "${jsonData[i].operatorName}",`;
          } else if (keyValue == 'details') {
            // details += `"Root cause": "${value.trim()}",`;
            rootCause = value.trim();
          } else {
            details += `"${key}": "${value.trim()}",`;
          }
        }
        allDocumentString +=
          `Operator name:"${jsonData[i].operatorName}",` +
          `Tail number:"${jsonData[i].tailNumber}",` +
          `Root cause:"${rootCause}",` +
          `Incident Details for operator ${jsonData[i].operatorName}:"[${details}]"\n`;
      }
    }
  }
  console.log(allDocumentString);
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 8000,
    chunkOverlap: 3000,
  });

  const rawDocuments = await splitter.createDocuments([allDocumentString]);
  let vectorStore = new HNSWLib(embeddings, { space: 'cosine' });
  await vectorStore.addDocuments(rawDocuments);
  await vectorStore.save('./incidents_vector_string');

  testSimilarity();
};

const runQA = async () => {
  const embeddings = new OpenAIEmbeddings();

  let vectorStore = await HNSWLib.load('./incidents_vector_string', embeddings);
  const model = new ChatOpenAI({
    modelName: 'gpt-3.5-turbo-1106',
    maxTokens: -1,
    verbose: true,
    callbacks: [
      {
        handleLLMEnd(output) {
          console.log(JSON.stringify(output, null, 2));
        },
      },
    ],
  });

  // Create a system & human prompt for the chat model
  const SYSTEM_TEMPLATE =
    `As an AI assistant, your role now includes finding airline incident information. ` +
    `Use the following pieces of context to answer the question at the end.` +
    `If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    {context}`;
  const messages = [
    SystemMessagePromptTemplate.fromTemplate(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.fromTemplate('{question}'),
  ];
  const prompt = ChatPromptTemplate.fromMessages(messages);

  const chain = RunnableSequence.from([
    {
      context: vectorStore.asRetriever().pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
  ]);
  // const response = await chain.invoke(
  //   `all incident details with their date, time, tailnumber, type, Departure airport, Destination airport, for operator with name "Hop-A-Jet"`,
  // );

  // const response = await chain.invoke(
  //   `Incident with latest date, their reason, date, tailnumber, Departure airport, Destination airport, for operator with name "Hop-A-Jet", ignore time`,
  // );

  const response = await chain.invoke(
    `five most recent incidents for Latitude 33 operator with root cause, date, time, tailnumber, type, ` +
      `Departure airport, Destination airport`,
  );

  // const response = await chain.invoke(
  //   `Most recent incident date for Hop-A-Jet operator with root cause, tailnumber, incident date, ` +
  //     `Departure airport, Destination airport`,
  // );

  console.log(response);
};

runQA();
//testSimilarity();
//prepareTextVector();
