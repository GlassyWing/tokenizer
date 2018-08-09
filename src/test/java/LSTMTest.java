import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.tensorflow.distruntime.DistributedRuntimeProtos;

/**
 * Create by manlier 2018/7/30 11:54
 */
public class LSTMTest {

    @Test
    public void testLSTM() {


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .l2(0.001)
                .list()
                .layer(0, new Bidirectional(new LSTM.Builder().nIn(30).nOut(100)
                        .activation(Activation.TANH)
                        .build()))
                .setInputType(InputType.recurrent(200))
                .layer(1, new DenseLayer.Builder().nIn(200).nOut(4).build())
                .layer(2, new RnnOutputLayer.Builder().nIn(4).nOut(4).activation(Activation.SOFTMAX).build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        System.out.println(net.summary());

        INDArray indArray = Nd4j.randn(new int[]{1, 30, 5});
        INDArray labels = Nd4j.create(new double[][][]{{
                {0, 0, 1, 0},
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 0, 1},
                {0, 1, 0, 0}
        }}).permute(0, 2, 1);

        System.out.println(labels);

        DataSet dataSet = new DataSet(indArray, labels);
        net.fit(dataSet);


    }

    @Test
    public void justMath() {
        INDArray array = Nd4j.arange(1, 17).reshape(2, 2, 4);
        System.out.println(array.permute(2, 0, 1));
        System.out.println(array.size(2));
        System.out.println(array.tensorAlongDimension((int)array.size(2) - 1,0, 1));
        System.out.println(array.tensorAlongDimension((int)array.size(2) - 1,1, 0));
    }
}
