import { pipeline } from "@Xenova/transformers";

// const classifier = await pipeline("sentiment-analysis");

async function embedder() {
  const embedding = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
  );
  const response = await embedding("i love xenova", {
    pooling: "mean",
    normalize: true,
  });
  console.log(response);
}

async function textGeneration() {
  const generator = await pipeline(
    "text-generation",
    "Xenova/gpt2",
  );
  const response = await generator("i love onnx runtime", {
    temperature: 0.7,
    max_new_tokens: 100,
    repetition_penalty: 2.0,
  });
  console.log(response);
}
textGeneration();
