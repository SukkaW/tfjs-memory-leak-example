const tf = require('@tensorflow/tfjs-node');
const automl = require('@tensorflow/tfjs-automl');
const { promises: fsPromises } = require('fs');
const path = require('path');

(async () => {
  const graphModel = await tf.loadGraphModel(`file://${path.resolve(__dirname, 'models/model.json')}`);
  const model = new automl.ImageClassificationModel(graphModel, ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']);

  const input = await fsPromises.readFile(path.resolve(__dirname, 'daisy.jpg'));

  for (let i = 0; i < 10000; i++) {
    const tensor = tf.node.decodeJpeg(input);
    const predictions = await model.classify(tensor);

    tensor.dispose();
    console.log(predictions[0].label, process.memoryUsage().rss, tf.memory().numBytes);
  }
})();
