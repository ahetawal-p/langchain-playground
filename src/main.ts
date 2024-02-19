import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { ChatOpenAI, OpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { RetrievalQAChain } from 'langchain/chains';
import { HumanMessage, SystemMessage } from 'langchain/schema';
import {
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  ChatPromptTemplate,
} from 'langchain/prompts';
import { RunnableSequence, RunnablePassthrough } from 'langchain/runnables';
import { StringOutputParser } from 'langchain/schema/output_parser';
import { formatDocumentsAsString } from 'langchain/util/document';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

const embeddings = new OpenAIEmbeddings();

// const loader = new PDFLoader('./sheet.pdf', { parsedItemSeparator: '' });

// const splitter = new RecursiveCharacterTextSplitter({
//   chunkSize: 10000,
//   chunkOverlap: 2000,
// });

// const rawDocuments = await loader.loadAndSplit(splitter);
// let vectorStore = new HNSWLib(embeddings, { space: 'cosine' });
// await vectorStore.addDocuments(rawDocuments);
// await vectorStore.save('./output');

// const retrievedDocs = await vectorStore.similaritySearch(
//   'Please extract and format the aircraft inventory information from the provided fleet sheet text, focusing on key details like operator, aircraft model',
//   1,
// );

// const pageContents = retrievedDocs.map((doc) => doc.pageContent);

// console.log(pageContents);

let vectorStore = await HNSWLib.load('./output', embeddings);
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
const SYSTEM_TEMPLATE = `As an AI assistant, your role now includes processing and organizing fleet sheet data for private charter flight operator companies. This task involves extracting specific details about each aircraft in the fleet, such as operator name, aircraft model, year of manufacture and refurbishment, tail number, homebase, availability of wifi, pet policy, and hourly rate. Your objective is to convert these details into a structured JSON format that aligns with the updated schema, ensuring accurate and efficient data management for the fleet inventory.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
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
//   'Please extract all the states where the fleet is located.',
// );

const schema = {
  name: 'add_aircraft_inventory',
  description:
    "Add aircraft inventory data to the database, detailing each aircraft in a private charter flight operator's fleet",
  parameters: {
    type: 'object',
    properties: {
      fleet_data: {
        type: 'object',
        description: 'Object containing the fleet data',
        properties: {
          operator: {
            type: 'string',
            description: 'Name of the flight operator company',
          },
          aircrafts: {
            type: 'array',
            description:
              'Array of objects, each representing an aircraft in the fleet',
            items: {
              type: 'object',
              properties: {
                model: {
                  type: 'string',
                  description: 'Model of the aircraft',
                },
                year_manufactured: {
                  type: 'integer',
                  description: 'Year the aircraft was manufactured',
                },
                year_refurbished: {
                  type: 'integer',
                  description:
                    'Year the aircraft was last refurbished, if applicable',
                },
                tail_number: {
                  type: 'string',
                  description: 'The tail number of the aircraft',
                },
                homebase: {
                  type: 'string',
                  description:
                    'The homebase airport or location of the aircraft',
                },
                has_wifi: {
                  type: 'boolean',
                  description:
                    'Indicates if the aircraft is equipped with wifi',
                },
                allows_pets: {
                  type: 'boolean',
                  description: 'Indicates if pets are allowed on the aircraft',
                },
                passenger_count: {
                  type: 'integer',
                  description:
                    'Number of passengers/seats/pax the aircraft can accommodate',
                },
                hourly_rate: {
                  type: 'string',
                  description: 'Hourly rate for chartering the aircraft',
                },
              },
              required: [
                'model',
                'year_manufactured',
                'tail_number',
                'homebase',
                'hourly_rate',
              ],
            },
          },
        },
        required: ['operator', 'aircrafts'],
      },
    },
    required: ['fleet_data'],
  },
};
const response = await chain.invoke(
  `Please extract and format the aircraft inventory information from the provided fleet sheet text, focusing on key details like operator, aircraft model, year manufactured, year refurbished, tail number, homebase, wifi availability, pet allowance, and hourly rate. Organize this information JSON schema`,
);
console.log(response);
