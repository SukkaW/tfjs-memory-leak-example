const tf = require('@tensorflow/tfjs-node');
const automl = require('@tensorflow/tfjs-automl');
const { promises: fsPromises } = require('fs');
const path = require('path');

(async () => {
  printPlatformInformation();

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

function printPlatformInformation() {
  console.log('Platform info:');

  const os = require('os');
  const { node, v8 } = process.versions;
  const plat = `OS: ${os.type()} ${os.release()} ${os.arch()}\nNode.js: ${node}\nV8: ${v8}`;
  const cpus = os.cpus().map(cpu => cpu.model).reduce((o, model) => {
    o[model] = (o[model] || 0) + 1;
    return o;
  }, {});
  const cpusInfo = Object.keys(cpus).map((key) => {
    return `${key} x ${cpus[key]}`;
  }).join('\n');

  console.log(`${plat}\nCPU: ${cpusInfo}\nMemory: ${os.totalmem() / (1024 * 1024)} MiB\n`);
}
