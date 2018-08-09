package samediff;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Create by manlier 2018/8/2 20:47
 */
public class Ex2LambdaLayer {

    public static void main(String[] args) {
        int networkNumInputs = 28 * 28;          // For MNIST - 28x28 pixels
        int networkNumOutputs = 10;              // FOR MNIST - 10 classes
        int layerSize = 128;                     // 128 units for the SameDiff layers

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1e-1))
                .seed(12345)
                .activation(Activation.TANH)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                // Add some standard DL4J layers
                .layer(new ConvolutionLayer.Builder().nIn(1).nOut(16).kernelSize(2, 2).stride(1, 1)
                        .build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).build())
                .layer(new ConvolutionLayer.Builder().nIn(16).nOut(8).kernelSize(2, 2).stride(1, 1)
                        .build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).build())
                // Add custom SameDiff lambda layer:
                .layer(new L2NormalizeLambdaLayer(1, 2, 3))
                // Add standard DL4J output layer:
                .layer(new OutputLayer.Builder().nIn(7 * 7 * 8).nOut(networkNumOutputs)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(28, 28, 1))
                .inputPreProcessor(5, new CnnToFeedForwardPreProcessor(7, 7, 8))
                .build();
    }
}
